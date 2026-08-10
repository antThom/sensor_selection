from __future__ import annotations

import math
from typing import Optional, Sequence

try:
    import pybullet as p
except ImportError:
    p = None


SIGMA = 5.670374419e-8


class ThermalObject:
    """A lumped thermal model for one simulation body or link."""

    def __init__(
        self,
        body_id: Optional[int] = None,
        link_id: int = -1,
        *,
        client_id: Optional[int] = None,
        material: Optional[dict] = None,
        temperature: Optional[float] = None,
        dimensions: Optional[Sequence[float]] = None,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        area: Optional[float] = None,
        volume: Optional[float] = None,
        mass: Optional[float] = None,
        specific_heat: Optional[float] = None,
        density: Optional[float] = None,
        conductivity: Optional[float] = None,
        emissivity: Optional[float] = None,
        absorptivity: Optional[float] = None,
        contact_area: Optional[float] = None,
        contact_length: Optional[float] = None,
        natural_h: Optional[float] = None,
        diffuse_shade: Optional[float] = None,
        internal_heat: Optional[float] = None,
    ):
        mat = {} if material is None else dict(material)

        def pick(value, key, fallback):
            return mat.get(key, fallback) if value is None else value

        self.body_id = body_id
        self.link_id = int(link_id)
        self.client_id = client_id
        self.temperature = float(pick(temperature, "T", 293.15))
        self.cp = float(pick(specific_heat, "cp", mat.get("specific_heat", 900.0)))
        self.density = float(pick(density, "density", 1000.0))
        self.k = float(pick(conductivity, "conductivity", 0.7))
        self.emiss = float(pick(emissivity, "emiss", mat.get("emissivity", 0.95)))
        self.absorpt = float(
            pick(absorptivity, "absorpt", mat.get("absorptivity", 0.85))
        )
        self.natural_h = float(pick(natural_h, "natural_h", 5.0))
        self.diffuse_shade = float(pick(diffuse_shade, "diffuse_shade", 0.10))
        self.internal_heat = float(
            pick(internal_heat, "heat_watts", mat.get("internal_heat", 0.0))
        )

        self._manual_dimensions = dimensions is not None
        self.dimensions = tuple(
            float(value) for value in (dimensions or (1.0, 1.0, 1.0))
        )
        self._position = tuple(float(value) for value in position)
        self._area = None if area is None else float(area)
        self._volume = None if volume is None else float(volume)
        self.mass = pick(mass, "mass", None)
        self.mass = None if self.mass is None else float(self.mass)

        contact_area_value = pick(contact_area, "contact_area", None)
        contact_length_value = pick(contact_length, "contact_length", None)
        self.contact_area = (
            None if contact_area_value is None else float(contact_area_value)
        )
        self.contact_length = (
            None if contact_length_value is None else float(contact_length_value)
        )

        self.air_k = 0.0257
        self.air_nu = 1.46e-5
        self.air_pr = 0.71
        self.last_rates = self._zero_rates()
        self.last_terms = self._zero_rates()

        self.refresh_geometry()
        if self.contact_area is None:
            self.contact_area = self.dimensions[0] * self.dimensions[1]
        if self.contact_length is None:
            self.contact_length = min(self.dimensions) / 2.0
        self._validate()

    def _validate(self):
        if any(side <= 0 for side in self.dimensions):
            raise ValueError("dimensions must be positive")
        if self.mass <= 0 or self.cp <= 0:
            raise ValueError("mass and specific heat must be positive")
        if not 0 <= self.emiss <= 1 or not 0 <= self.absorpt <= 1:
            raise ValueError("emissivity and absorptivity must be between 0 and 1")

    def _options(self):
        return {} if self.client_id is None else {"physicsClientId": self.client_id}

    @staticmethod
    def _zero_rates():
        return {
            "conduction": 0.0,
            "convection": 0.0,
            "longwave": 0.0,
            "solar": 0.0,
            "radiation": 0.0,
            "internal": 0.0,
            "total": 0.0,
        }

    @property
    def T(self):
        return self.temperature

    @T.setter
    def T(self, value):
        self.temperature = float(value)

    @property
    def volume(self):
        return math.prod(self.dimensions) if self._volume is None else self._volume

    @property
    def surface_area(self):
        if self._area is not None:
            return self._area
        x, y, z = self.dimensions
        return 2.0 * (x * y + x * z + y * z)

    @property
    def thermal_mass(self):
        return self.mass * self.cp

    def refresh_geometry(self):
        if p is not None and self.body_id is not None:
            if not self._manual_dimensions:
                try:
                    low, high = p.getAABB(self.body_id, self.link_id, **self._options())
                    self.dimensions = tuple(
                        max(float(high[index] - low[index]), 1e-4) for index in range(3)
                    )
                except Exception:
                    pass
            if self.mass is None:
                try:
                    dynamic_mass = float(
                        p.getDynamicsInfo(
                            self.body_id, self.link_id, **self._options()
                        )[0]
                    )
                    if dynamic_mass > 0:
                        self.mass = dynamic_mass
                except Exception:
                    pass
        if self.mass is None:
            self.mass = max(self.density * self.volume, 0.01)

    def position(self):
        if p is None or self.body_id is None:
            return self._position
        try:
            if self.link_id >= 0:
                return tuple(
                    float(value)
                    for value in p.getLinkState(
                        self.body_id, self.link_id, **self._options()
                    )[0]
                )
            return tuple(
                float(value)
                for value in p.getBasePositionAndOrientation(
                    self.body_id, **self._options()
                )[0]
            )
        except Exception:
            return self._position

    def set_fallback_position(self, position):
        self._position = tuple(float(value) for value in position)

    def _axes(self):
        default = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        if p is None or self.body_id is None:
            return default
        try:
            if self.link_id >= 0:
                quaternion = p.getLinkState(
                    self.body_id, self.link_id, **self._options()
                )[1]
            else:
                quaternion = p.getBasePositionAndOrientation(
                    self.body_id, **self._options()
                )[1]
            matrix = p.getMatrixFromQuaternion(quaternion)
            return (
                (matrix[0], matrix[3], matrix[6]),
                (matrix[1], matrix[4], matrix[7]),
                (matrix[2], matrix[5], matrix[8]),
            )
        except Exception:
            return default

    def projected_area(self, direction=(0.0, 0.0, 1.0)):
        norm = math.sqrt(sum(float(value) ** 2 for value in direction))
        if norm == 0:
            return 0.0
        unit_direction = tuple(float(value) / norm for value in direction)
        x, y, z = self.dimensions
        face_areas = (y * z, x * z, x * y)
        axes = self._axes()
        return sum(
            face_areas[index]
            * abs(
                sum(
                    axes[index][component] * unit_direction[component]
                    for component in range(3)
                )
            )
            for index in range(3)
        )

    def sunlight(self, sun_direction, ray_distance=1000.0):
        if p is None or self.body_id is None or sun_direction is None:
            return 1.0
        norm = math.sqrt(sum(float(value) ** 2 for value in sun_direction))
        if norm == 0:
            return 0.0
        direction = tuple(float(value) / norm for value in sun_direction)
        radius = 0.5 * math.sqrt(sum(side * side for side in self.dimensions)) + 0.02
        center = self.position()
        start = [center[index] + direction[index] * radius for index in range(3)]
        end = [start[index] + direction[index] * ray_distance for index in range(3)]
        try:
            hit = p.rayTest(start, end, **self._options())[0]
            return 1.0 if hit[0] < 0 or hit[0] == self.body_id else self.diffuse_shade
        except Exception:
            return 1.0

    def conduction_rate(
        self,
        surface_temp,
        area=None,
        length=None,
        surface_conductivity=None,
    ):
        contact_area = self.contact_area if area is None else float(area)
        path_length = self.contact_length if length is None else float(length)
        effective_conductivity = self.k
        if surface_conductivity is not None and float(surface_conductivity) > 0:
            surface_k = float(surface_conductivity)
            effective_conductivity = 2.0 * self.k * surface_k / (self.k + surface_k)
        conductance = (
            effective_conductivity * max(contact_area, 0.0) / max(path_length, 1e-6)
        )
        return conductance * (float(surface_temp) - self.temperature)

    def convection_coefficient(self, wind_speed):
        wind = max(float(wind_speed), 0.0)
        if wind == 0:
            return self.natural_h
        characteristic_length = max(self.dimensions)
        reynolds = wind * characteristic_length / self.air_nu
        if reynolds < 5e5:
            nusselt = 0.664 * math.sqrt(reynolds) * self.air_pr ** (1.0 / 3.0)
        else:
            nusselt = max(0.037 * reynolds**0.8 - 871.0, 0.0) * self.air_pr ** (
                1.0 / 3.0
            )
        forced_h = self.air_k * nusselt / characteristic_length
        return (self.natural_h**3 + forced_h**3) ** (1.0 / 3.0)

    def convection_rate(self, ambient_temp, wind_speed=0.0):
        exposed_area = max(self.surface_area - self.contact_area, 0.0)
        return (
            self.convection_coefficient(wind_speed)
            * exposed_area
            * (float(ambient_temp) - self.temperature)
        )

    def longwave_rate(self, surroundings_temp):
        return (
            self.emiss
            * SIGMA
            * self.surface_area
            * (float(surroundings_temp) ** 4 - self.temperature**4)
        )

    def solar_rate(
        self,
        irradiance,
        sun_fraction=1.0,
        sun_direction=(0.0, 0.0, 1.0),
    ):
        solar = max(float(irradiance), 0.0)
        if solar <= 1.5:
            solar *= 1000.0
        return (
            self.absorpt
            * solar
            * self.projected_area(sun_direction)
            * max(0.0, min(float(sun_fraction), 1.0))
        )

    def heat_rates(
        self,
        *,
        ambient_temp=None,
        surroundings_temp=None,
        wind_speed=0.0,
        solar_irradiance=0.0,
        sun_fraction=1.0,
        sun_direction=(0.0, 0.0, 1.0),
        contact_temp=None,
        contact_area=None,
        contact_length=None,
        contact_conductivity=None,
        conductive_watts=None,
    ):
        rates = self._zero_rates()
        if ambient_temp is not None:
            rates["convection"] = self.convection_rate(ambient_temp, wind_speed)
        if surroundings_temp is not None:
            rates["longwave"] = self.longwave_rate(surroundings_temp)
        if solar_irradiance > 0:
            rates["solar"] = self.solar_rate(
                solar_irradiance, sun_fraction, sun_direction
            )
        if conductive_watts is not None:
            rates["conduction"] = float(conductive_watts)
        elif contact_temp is not None:
            rates["conduction"] = self.conduction_rate(
                contact_temp,
                contact_area,
                contact_length,
                contact_conductivity,
            )
        rates["radiation"] = rates["longwave"] + rates["solar"]
        rates["internal"] = self.internal_heat
        rates["total"] = (
            rates["conduction"]
            + rates["convection"]
            + rates["radiation"]
            + rates["internal"]
        )
        return rates

    def step(self, dt, **conditions):
        if dt < 0:
            raise ValueError("dt must not be negative")
        rates = self.heat_rates(**conditions)
        self.temperature = max(
            1.0,
            self.temperature + rates["total"] * float(dt) / self.thermal_mass,
        )
        self.last_rates = rates
        self.last_terms = {
            name: watts / self.thermal_mass for name, watts in rates.items()
        }
        return self.temperature

    def get_temp(
        self,
        dt,
        irradiance,
        ambient,
        T_sky,
        sun_dir=None,
        temps=None,
        wind=0.0,
    ):
        del temps
        direction = (0.0, 0.0, 1.0) if sun_dir is None else sun_dir
        shade = self.sunlight(direction) if irradiance > 0 else 0.0
        return self.step(
            dt,
            ambient_temp=ambient,
            surroundings_temp=T_sky,
            wind_speed=wind,
            solar_irradiance=irradiance,
            sun_fraction=shade,
            sun_direction=direction,
        )

    def as_dict(self):
        return {
            "body_id": self.body_id,
            "link_id": self.link_id,
            "T": self.temperature,
            "dimensions": self.dimensions,
            "area": self.surface_area,
            "volume": self.volume,
            "mass": self.mass,
            "cp": self.cp,
            "conductivity": self.k,
            "emiss": self.emiss,
            "absorpt": self.absorpt,
            "contact_area": self.contact_area,
            "contact_length": self.contact_length,
            "heat_watts": self.internal_heat,
        }
