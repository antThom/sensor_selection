from __future__ import annotations
import numpy as np
from typing import Optional

from sim.environment.thermal.thermal_manager import ThermalMaterialLibrary


class ThermalObject:
    """
    Represents a single thermally‑active entity in the simulation.
    material : dict, optional
        A dictionary with the keys alpha, beta, gamma, emiss, temperature (initial temperature).  If omitted the
        ``ThermalMaterialLibrary.DEFAULT`` entry is used.
    """

    def __init__(
        self,
        body_id: int,
        link_id: int = -1,
        *,
        material: Optional[dict] = None,
        sigma: float = 5.670374419e-8,  # SI value, identical to the one in
    ) -> None:
        self.body_id = body_id
        self.link_id = link_id

        # Pick a material
        if material is None:
            material = ThermalMaterialLibrary.DEFAULT

        # Store the material coefficients
        self.alpha: float = float(material.get("alpha", 1e-3))
        self.beta: float = float(material.get("beta", 1e-3))
        self.gamma: float = float(material.get("gamma", 5e-10))
        self.emiss: float = float(material.get("emiss", 0.95))
        self.sigma: float = float(sigma)

        # Temperature (Kelvin)
        self.T: float = float(material.get("T", 293.0))

    def get_temp(
        self, dt: float, irradiance: float, ambient: float, T_sky: float
    ) -> float:
        # change in temperature due to radiative heat transfer
        dT_rad = -self.gamma * self.emiss * self.sigma * (self.T**4 - T_sky**4)

        # total temperature rate, utilizing the same formula as in ThermalManager.update()
        dTdt = self.alpha * irradiance - self.beta * (self.T - ambient) + dT_rad

        # euler integration to update the temperature
        self.T += dTdt * dt

        return self.T

    # Helper that the ThermalManager can call to expose the internal dict
    def as_dict(self) -> dict:
        """
        Return a dictionary in the exact shape that ``ThermalManager`` expects
        for ``add_object``.  This makes it easy to register the object
        directly:

        >>> manager.add_object(**obj.as_dict())
        """
        return {
            "body_id": self.body_id,
            "link_id": self.link_id,
            "init_T": self.T,
            "alpha": self.alpha,
            "beta": self.beta,
            "emiss": self.emiss,
            "gamma": self.gamma,
            # ``sigma`` is not supplied to ``add_object``
        }
