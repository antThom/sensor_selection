import numpy as np
import time
import pybullet as p
import pybullet_data
from sim.Constants import *


class ThermalMaterialLibrary:
    """Preset emissivity and thermal response for common object types."""
    DEFAULT = {
        "alpha": 5e-3,     # solar absorption
        "beta": 1e-3,      # convective cooling
        "gamma": 1e-10,    # radiative coupling
        "emiss": 0.95,     # IR emissivity
        "T": 288.8
    }

    MATERIALS = {
        "tree":     {"alpha": 4e-3, "beta": 1e-3, "gamma": 8e-11, "emiss": 0.98, "T": 294.3},
        "cloud":    {"alpha": 1e-3, "beta": 2e-3, "gamma": 1e-10, "emiss": 0.90, "T": 260.0},
        "metal":    {"alpha": 2e-3, "beta": 2e-3, "gamma": 5e-11, "emiss": 0.20, "T": 291.0},
        "robot":    {"alpha": 3e-3, "beta": 1e-3, "gamma": 9e-11, "emiss": 0.80, "T": 285.0},
        "terrain":  {"alpha": 5e-3, "beta": 1e-3, "gamma": 1e-10, "emiss": 0.92, "T": 290.0},
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
    def __init__(self,time_of_day, ambient_K=293.0, T_sky=260.0):
        self.objects     = {}
        self.ambient     = ambient_K
        self.T_sky       = T_sky
        self.time_of_day = time_of_day

    def add_object(self, body_id, link_id=-1, init_T=None, alpha=1e-3, beta=1e-3, emiss=0.95, gamma=5E-10):
        # Beta: Cooling Coefficient
        # I(t): Normalized solar irradiance (0 -> 1 by the time-of-day)
        # Alpha: Heating Coefficient (absorptivity x area / heat capacity)
        # T_air: Ambient air temperature
        # gamma: Radiative Coupling Coefficient -> Surface Area / (mass * specific heat capacity)
        # epsilon: Emissivity (0 -> 1)
        # sigma: Stefan-Boltzmann constant
        # T_sky: Effective sky temperature
        self.objects[(body_id, link_id)] = {
            "T": init_T or self.ambient,
            "alpha": alpha,
            "beta": beta,
            "emiss": emiss,
            "sigma": stefan_boltzmann_constant,
            "gamma": gamma
        }

    def update(self, dt, irradiance):
        for k, obj in self.objects.items():
            radiative_term = self.compute_radiative(obj)
            dT = obj["alpha"] * irradiance - obj["beta"] * (obj["T"] - self.ambient) + radiative_term
            obj["T"] += dT * dt

    def compute_radiative(self, obj):
        return -obj["gamma"]*obj["emiss"]*obj["sigma"]*(obj["T"]**4 - self.T_sky**4)

    def get_temperature(self, body_id, link_id=-1):
        return self.objects.get((body_id, link_id), {"T": self.ambient})["T"]

    def register_body(self, body_id: int, filename: str, per_link=False):
        """Automatically register a URDF-loaded body with reasonable material settings."""

        material = ThermalMaterialLibrary.match_material_from_filename(filename)

        # Base link
        self.add_object(
            body_id, -1,
            alpha=material["alpha"],
            beta=material["beta"],
            gamma=material["gamma"],
            emiss=material["emiss"],
            init_T=material["T"]
        )

        # Optional: register each link individually
        if per_link:
            n = p.getNumJoints(body_id)
            for link_id in range(n):
                self.add_object(
                    body_id, link_id,
                    alpha=material["alpha"],
                    beta=material["beta"],
                    gamma=material["gamma"],
                    emiss=material["emiss"]
                )