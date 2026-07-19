import numpy as np
import time
import pybullet as p
import pybullet_data
from sim.utils.CONSTANTS import *


class ThermalMaterialLibrary:
    """Preset emissivity and thermal response for common object types."""
    DEFAULT = {
        "alpha": 5e-3,     # solar absorption
        "beta": 1e-3,      # convective cooling
        "gamma": 1e-10,    # radiative coupling
        "emiss": 0.95,     # IR emissivity
        "T": 288.8,        # temperature (kelvin)
        "cp": 900.0,       # specific heat capacity
        "density": 1000.0,  # density
        "conductivity": 0.7, # thermal conductivity
        "absorpt": 0.85    # solar absorptivity
    }

    MATERIALS = {
        "tree":     {"alpha": 4e-3, "beta": 1e-3, "gamma": 8e-11, "emiss": 0.98, "T": 294.3, "cp": 1700.0, "density": 700.0, "conductivity": 0.18, "absorpt": 0.75},
        "cloud":    {"alpha": 1e-3, "beta": 2e-3, "gamma": 1e-10, "emiss": 0.90, "T": 260.0, "cp": 1005.0, "density": 0.8, "conductivity": 0.025, "absorpt": 0.25},
        "metal":    {"alpha": 2e-3, "beta": 2e-3, "gamma": 5e-11, "emiss": 0.20, "T": 291.0, "cp": 900.0, "density": 2700.0, "conductivity": 180.0, "absorpt": 0.45},
        "robot":    {"alpha": 3e-3, "beta": 1e-3, "gamma": 9e-11, "emiss": 0.80, "T": 285.0, "cp": 850.0, "density": 1200.0, "conductivity": 15.0, "absorpt": 0.65},
        "terrain":  {"alpha": 5e-3, "beta": 1e-3, "gamma": 1e-10, "emiss": 0.92, "T": 290.0, "cp": 800.0, "density": 1600.0, "conductivity": 1.5, "absorpt": 0.80},
        "generic":  DEFAULT
    }

    @staticmethod
    def match_material_from_filename(filename: str):
        f = filename.lower()
        if "tree" in f:
            return ThermalMaterialLibrary.MATERIALS["tree"]
        if "cloud" in f:
            return ThermalMaterialLibrary.MATERIALS["cloud"]
        if "metal" in f or "aluminum" in f:
            return ThermalMaterialLibrary.MATERIALS["metal"]
        if "robot" in f or "agent" in f:
            return ThermalMaterialLibrary.MATERIALS["robot"]
        if "terrain" in f or "ground" in f:
            return ThermalMaterialLibrary.MATERIALS["terrain"]
        return ThermalMaterialLibrary.DEFAULT
    
class ThermalManager:
    def __init__(self, time_of_day, ambient_K=293.0, T_sky = 260.0):
        # objects now hold ThermalObject instances, not just coefficient dicts.
        # public methods stayed the same so cameras/sensors can still call
        # get_temperature(body_id) without knowing about the richer model.
        self.objects = {}
        self.ambient = ambient_K
        self.T_sky = T_sky
        self.time_of_day = time_of_day
        self.wind_speed = 0.0

    def add_object(self, 
                   body_id, 
                   link_id=-1, 
                   init_T=None, 
                   alpha=None, 
                   beta=None, 
                   emiss=None, 
                   gamma=None, 
                   material=None, 
                   area=None, 
                   volume=None, 
                   mass=None, 
                   cp=None, 
                   absorpt=None, 
                   conductivity=None, 
                   contact_area=None, 
                   heat_watts=0.0):
    # accepted extras:
        # area/volume/mass/cp: manual size and thermal mass overrides
        # absorpt: direct solar absorption override
        # conductivity/contact_area: contact conduction tuning
        # heat_watts: internal object heat generation
        from sim.Environment.ThermalObject.ThermalObject import ThermalObject

        if material is None:
            material = dict(ThermalMaterialLibrary.DEFAULT)
            material.update({"alpha": 1e-3 if alpha is None else alpha, "beta": 1e-3 if beta is None else beta, "gamma": 5E-10 if gamma is None else gamma, "emiss": 0.95 if emiss is None else emiss})
        else:
            material = dict(material)
            if alpha is not None: material["alpha"] = alpha
            if beta is not None: material["beta"] = beta
            if gamma is not None: material["gamma"] = gamma
            if emiss is not None: material["emiss"] = emiss

        self.objects[(body_id, link_id)] = ThermalObject(body_id, link_id, material=material, sigma=STEFAN_BOLTZMANN_CONSTANT, init_T=init_T or material.get("T", self.ambient), client_id=self.physics_client, area=area, volume=volume, mass=mass, cp=cp, absorpt=absorpt, conductivity=conductivity, contact_area=contact_area, heat_watts=heat_watts)

    def sun_dir(self):
        # simple sun vector from time-of-day; caller can pass a better one.
        # returns None at night so ThermalObject skips shadow rays/sun heating.
        # elevation peaks at noon, azimuth rotates once per day.
        tod = self.time_of_day
        if hasattr(tod, "hour"): hour = tod.hour + tod.minute/60.0 + tod.second/3600.0
        elif isinstance(tod, (int, float)):
            parts = [float(x) for x in tod.split(":")]
            hour = parts[0] + (parts[1] if len(parts) > 1 else 0)/60.0 + (parts[2] if len(parts) > 2 else 0)/3600.0
        else: hour = float(tod)
        elev = max(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))  # simple sine curve from 6am to 6pm
        horiz = np.sqrt(max(1.0 - elev**2, 0.0))
        azim = (hour / 24.0) * 2.0 * np.pi  # full rotation over 24 hours
        return [None, np.asarray([horiz*np.cos(azim), horiz*np.sin(azim), elev])][elev > 0.0]
    
    def temp_map(self):
        # contact heat needs the neighbor temperatures from the start of tick.
        # without this snapshot, whichever object updated first would slightly
        # change the result for objects updated later.
        return {k: (v.T if hasattr(v, "T") else v["T"]) for k, v in self.objects.items()}

    def update(self, dt, irradiance, wind=None, sun_dir=None):
        # irradiance may be normalized 0..1 or W/m^2; object handles both.        
        # update order:        
        # 1. snapshot current temps for contact heat        
        # 2. choose wind and sun vector for this tick        
        # 3. let each ThermalObject integrate one Euler step
        temps = self.temp_map()
        wind = self.wind_speed if wind is None else wind
        sun_dir = self.sun_dir() if sun_dir is None else sun_dir

        for k, obj in self.objects.items():
            if hasattr(obj, "get_temp"):
                obj.get_temp(dt, irradiance, self.ambient, self.T_sky, sun_dir=sun_dir, temps=temps, wind=wind)
            else:
                radiative_term = self.compute_radiative(obj)
                dT = obj["alpha"] * irradiance - obj["beta"] * (obj["T"] - self.ambient) + radiative_term
                obj["T"] += dT * dt
    
    def compute_radiative(self, obj):
        if hasattr(obj, "T"): return -obj.gamma*obj.emiss*obj.sigma*(obj.T**4 - self.T_sky**4)
        return -obj["gamma"]*obj["emiss"]*obj["sigma"]*(obj["T"]**4 - self.T_sky**4)

    def get_temperature(self, body_id, link_id=-1):
        # sensors only need the current scalar temperature
        obj = self.objects.get((body_id, link_id), None)
        if obj is None: return self.ambient
        return obj.T if hasattr(obj, "T") else obj["T"]
    
    def register_body(self, body_id: int, filename: str, per_link=False):
        """Automatically register a URDF-loaded body with reasonable material settings."""
        # filename decides the material preset, so tree/cloud/robot assets
        # get different heat capacity, emissivity, density, and so on.

        material = ThermalMaterialLibrary.match_material_from_filename(filename)
        # base link is always registered; per-link registration is optional
        self.add_object(body_id, -1, alpha=material["alpha"], beta=material["beta"], gamma=material["gamma"], emiss=material["emiss"], init_T=material["T"], material=material)
    
    # per-link makes articulated/compound objects heat separately, but it only works when pybullet is available to tell us how many links exist.
        if per_link and p:
            n = p.getNumJoints(body_id)
            for link_id in range(n):
                self.add_object(body_id, link_id, alpha=material["alpha"], beta=material["beta"], gamma=material["gamma"], emiss=material["emiss"], init_T=material["T"], material=material)