from __future__ import annotations
import math
import numpy as np
from typing import Optional

from sim.environment.thermal.thermal_manager import ThermalManager, ThermalMaterialLibrary

try:
    import pybullet as p
except Exception:
    p = None




class ThermalObject:
    # This is a lumped thermal model: every object/link has one temperature.
    # It is not resolving surface-by-surface heat flow.  Instead, it estimates
    # the largest drivers that matter for this sim: solar gain, air cooling,
    # long-wave radiation to the sky, contact heat, and internal heat.
    def __init__(
        self,
        body_id: int,
        link_id: int = -1,
        *,
        material: Optional[dict] = None,
        sigma: float = 5.670374419e-8,
        init_T: Optional[float] = None,
        client_id: Optional[int] = None,
        area: Optional[float] = None,
        volume: Optional[float] = None,
        mass: Optional[float] = None,
        cp: Optional[float] = None,
        absorpt: Optional[float] = None,
        conductivity: Optional[float] = None,
        contact_area: Optional[float] = None,
        heat_watts: float = 0.0,
    ) -> None:
        self.body_id = body_id
        self.link_id = link_id
        self.client_id = client_id
        material = ThermalMaterialLibrary.DEFAULT if material is None else material

        # old coefficients still drive the rough rate:
        # alpha: sun heating response, larger means faster warming in light
        # beta: convective response, larger means faster pull toward air temp
        # gamma: radiative response, larger means stronger sky radiation
        # emiss: IR emissivity, mostly affects thermal radiation
        self.alpha = float(material.get("alpha", 1e-3))
        self.beta = float(material.get("beta", 1e-3))
        self.gamma = float(material.get("gamma", 5e-10))
        self.emiss = float(material.get("emiss", 0.95))
        self.sigma = float(sigma)

        # extra physical knobs:
        # absorpt: solar absorption fraction; dark/matte objects heat faster
        # cp: specific heat capacity in J/(kg*K), controls thermal inertia
        # mass: kg; if missing, pybullet mass or density*volume is used
        # area: m^2-ish exposed area; if missing, estimated from AABB
        # volume: m^3-ish volume; if missing, estimated from AABB
        # k: thermal conductivity, used only for contact heat
        # contact_area: guessed contact patch area per contact point
        # heat_watts: internal generated heat, useful for robots/electronics
        # lapse: air temp drop per meter altitude
        # shadow_min: sunlight fraction that remains while shaded
        # wind_coef: how much wind increases convective cooling
        self.absorpt = float(material.get("absorpt", self.emiss if absorpt is None else absorpt))
        self.cp = float(material.get("cp", 900.0 if cp is None else cp))
        self.mass = mass if mass is not None else material.get("mass", None)
        self.area = area if area is not None else material.get("area", None)
        self.volume = volume if volume is not None else material.get("volume", None)
        self.k = float(material.get("conductivity", 0.7 if conductivity is None else conductivity))
        self.contact_area = float(material.get("contact_area", 0.02 if contact_area is None else contact_area))
        self.heat_watts = float(material.get("heat_watts", heat_watts))
        self.lapse = float(material.get("lapse", 0.0065))
        self.shadow_min = float(material.get("shadow_min", 0.18))
        self.wind_coef = float(material.get("wind_coef", 0.08))
        self.density = float(material.get("density", 1000.0))
        self.T = float(material.get("T", 293.0) if init_T is None else init_T)
        self.last_terms = {}
        self.update_geom()

    def opts(self):
        # keeps pybullet calls pinned to the correct physics client
        return {} if self.client_id is None else {"physicsClientId": self.client_id}

<<<<<<< HEAD
    def get_temp(
        self, dt: float, irradiance: float, ambient: float, T_sky: float
    ) -> float:
        # change in temperature due to radiative heat transfer
        dT_rad = -self.gamma * self.emiss * self.sigma * (self.T**4 - T_sky**4)

        # total temperature rate, utilizing the same formula as in ThermalManager.update()
        dTdt = self.alpha * irradiance - self.beta * (self.T - ambient) + dT_rad

        # euler integration to update the temperature
=======
    def pos(self):
        # base links and normal links use different pybullet APIs
        if p is None: return np.zeros(3)
        try:
            if self.link_id >= 0: return np.asarray(p.getLinkState(self.body_id, self.link_id, **self.opts())[0], dtype=float)
            return np.asarray(p.getBasePositionAndOrientation(self.body_id, **self.opts())[0], dtype=float)
        except Exception:
            return np.zeros(3)

    def update_geom(self):
        # pybullet gives rough geometry; manual values override these.
        # AABB is cheap and stable, but it overestimates irregular meshes.
        # That is okay here because we only need a consistent exposed-area
        # estimate for the statistical thermal trend.
        if p is not None:
            try:
                lo, hi = p.getAABB(self.body_id, self.link_id, **self.opts())
                side = np.maximum(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float), 0.05)
                self.area = float(self.area or 2*(side[0]*side[1] + side[0]*side[2] + side[1]*side[2]))
                self.volume = float(self.volume or np.prod(side))
            except Exception:
                pass
            try:
                dynmass = float(p.getDynamicsInfo(self.body_id, self.link_id, **self.opts())[0])
                if self.mass is None and dynmass > 0: self.mass = dynmass
            except Exception:
                pass
        self.area = float(self.area or 1.0)
        self.volume = float(self.volume or 1.0)
        if self.mass is None: self.mass = max(self.volume*self.density, 0.1)
        # thermal mass is the energy needed to move this object by 1 Kelvin
        self.therm_mass = max(float(self.mass)*self.cp, 1.0)

    def sun_area(self, sun_dir):
        # project a rough cube-like area onto the sun direction
        side = math.sqrt(max(self.area/6.0, 1e-9))
        d = np.abs(np.asarray(sun_dir, dtype=float))
        return max(side*side*(d[0] + d[1] + d[2]), 0.05)

    def shadow(self, sun_dir, raydist=500.0):
        # ray toward the sun; hit means mostly shaded, not totally dark.
        # This handles clouds/trees/terrain blocking sunlight in the sim.
        # We keep shadow_min above zero so diffuse light still warms objects.
        if p is None or sun_dir is None: return 1.0
        d = np.asarray(sun_dir, dtype=float)
        n = np.linalg.norm(d)
        if n == 0: return 1.0
        d = d/n
        start = self.pos() + d*0.2
        end = start + d*raydist
        try:
            hit = p.rayTest(start.tolist(), end.tolist(), **self.opts())[0]
            if hit[0] >= 0 and not (hit[0] == self.body_id and (hit[1] == self.link_id or self.link_id < 0)): return self.shadow_min
        except Exception:
            pass
        return 1.0

    def contact_term(self, temps=None):
        # contact area is unknown, so use a small configurable patch per contact.
        # The manager passes temps from the start of the tick so heat exchange
        # does not depend on update order.
        if p is None: return 0.0
        temps = temps or {}
        total = 0.0
        try:
            pts = p.getContactPoints(bodyA=self.body_id, linkIndexA=self.link_id, **self.opts())
        except Exception:
            return 0.0
        for pt in pts:
            other = (pt[2], pt[4])
            otherT = temps.get(other, temps.get((pt[2], -1), None))
            if otherT is not None: total += self.k*self.contact_area*(float(otherT) - self.T)/self.therm_mass
        return total

    def get_temp(self, dt: float, irradiance: float, ambient: float, T_sky: float, sun_dir=None, temps=None, wind=0.0) -> float:
        # irradiance accepts either normalized 0..1 sunlight or W/m^2.
        # if it is small, assume normalized and scale to about full sun.
        self.update_geom()
        irr = irradiance*1000.0 if irradiance <= 5 else irradiance

        # local environment around the object
        shade = self.shadow(sun_dir) if irr > 0 else 0.0
        air = ambient - self.lapse*max(self.pos()[2], 0.0)
        therm_mass_value = max((float(self.therm_mass) if self.therm_mass is not None else 1.0)/1000.0, 0.1)
        scale = float(self.area or 1.0)/therm_mass_value
        sunarea = self.sun_area([0, 0, 1] if sun_dir is None else sun_dir)
        area = float(self.area or 1.0)
        therm_mass = max(self.therm_mass/1000.0, 0.1)

        # each term is K/s:
        # dT_sun: short-wave solar gain, reduced by shadow and thermal mass
        # dT_conv: convection toward local air temp, stronger with wind/area
        # dT_rad: long-wave radiation exchange with the effective sky temp
        # dT_contact: conduction to touching objects already in temp map
        # dT_internal: object heat generation such as motors/electronics
        dT_sun = self.alpha*(irr/1000.0)*self.absorpt*sunarea*shade/therm_mass
        dT_conv = -self.beta*(1 + self.wind_coef*max(float(wind), 0.0))*(self.T - air)*scale
        dT_rad = -self.gamma*self.emiss*self.sigma*area*(self.T**4 - T_sky**4)/therm_mass
        dT_contact = self.contact_term(temps)
        dT_internal = self.heat_watts/self.therm_mass
        dTdt = dT_sun + dT_conv + dT_rad + dT_contact + dT_internal

>>>>>>> upstream/panda3_env
        self.T += dTdt * dt
        # saved for debugging/tuning, e.g. inspect why one object got hotter
        self.last_terms = {"sun": dT_sun, "conv": dT_conv, "rad": dT_rad, "contact": dT_contact, "internal": dT_internal, "shade": shade, "air": air}
        return self.T

    def as_dict(self) -> dict:
        # keeps compatibility with old ThermalManager.add_object style
        return {
            "body_id": self.body_id,
            "link_id": self.link_id,
            "init_T": self.T,
            "alpha": self.alpha,
            "beta": self.beta,
            "emiss": self.emiss,
            "gamma": self.gamma,
            "area": self.area,
            "volume": self.volume,
            "mass": self.mass,
            "cp": self.cp,
            "absorpt": self.absorpt,
            "conductivity": self.k,
            "contact_area": self.contact_area,
            "heat_watts": self.heat_watts,
        }