import math

try:
    import pybullet as p
except ImportError:
    p = None

from sim.Environment.ThermalObject import ThermalObject


class ThermalMaterialLibrary:
    DEFAULT = {
        "T": 288.8,
        "cp": 900.0,
        "density": 1000.0,
        "conductivity": 0.7,
        "emiss": 0.95,
        "absorpt": 0.85,
        "natural_h": 5.0,
    }

    MATERIALS = {
        "tree": {"T": 294.3, "cp": 1700.0, "density": 700.0, "conductivity": 0.18, "emiss": 0.98, "absorpt": 0.75},
        "cloud": {"T": 260.0, "cp": 1005.0, "density": 0.8, "conductivity": 0.025, "emiss": 0.90, "absorpt": 0.25},
        "metal": {"T": 291.0, "cp": 900.0, "density": 2700.0, "conductivity": 180.0, "emiss": 0.20, "absorpt": 0.45},
        "robot": {"T": 285.0, "cp": 850.0, "density": 1200.0, "conductivity": 15.0, "emiss": 0.80, "absorpt": 0.65},
        "terrain": {"T": 290.0, "cp": 800.0, "density": 1600.0, "conductivity": 1.5, "emiss": 0.92, "absorpt": 0.80},
        "generic": DEFAULT,
    }

    @staticmethod
    def match_material_from_filename(filename):
        name = str(filename).lower()
        if "tree" in name:
            key = "tree"
        elif "cloud" in name:
            key = "cloud"
        elif "metal" in name or "aluminum" in name:
            key = "metal"
        elif "robot" in name or "agent" in name or "drone" in name:
            key = "robot"
        elif "terrain" in name or "ground" in name or "flat" in name:
            key = "terrain"
        else:
            key = "generic"
        material = dict(ThermalMaterialLibrary.DEFAULT)
        material.update(ThermalMaterialLibrary.MATERIALS[key])
        return material


class ThermalManager:
    def __init__(self, time_of_day, ambient_K=293.0, T_sky=260.0, physics_client=None):
        self.objects = {}
        self.ambient = float(ambient_K)
        self.T_sky = float(T_sky)
        self.time_of_day = time_of_day
        self.physics_client = physics_client
        self.wind_speed = 0.0

    def add_object(
        self,
        body_id,
        link_id=-1,
        init_T=None,
        alpha=None,
        beta=None,
        emiss=None,
        gamma=None,
        material=None,
        dimensions=None,
        area=None,
        volume=None,
        mass=None,
        cp=None,
        absorpt=None,
        conductivity=None,
        contact_area=None,
        contact_length=None,
        heat_watts=None,
    ):
        mat = dict(ThermalMaterialLibrary.DEFAULT if material is None else material)
        if emiss is not None:
            mat["emiss"] = emiss
        if absorpt is not None:
            mat["absorpt"] = absorpt
        if cp is not None:
            mat["cp"] = cp
        if conductivity is not None:
            mat["conductivity"] = conductivity
        if alpha is not None:
            mat["legacy_alpha"] = alpha
        if beta is not None:
            mat["legacy_beta"] = beta
        if gamma is not None:
            mat["legacy_gamma"] = gamma

        obj = ThermalObject(
            body_id,
            link_id,
            client_id=self.physics_client,
            material=mat,
            temperature=mat.get("T", self.ambient) if init_T is None else init_T,
            dimensions=dimensions,
            area=area,
            volume=volume,
            mass=mass,
            specific_heat=cp,
            conductivity=conductivity,
            emissivity=emiss,
            absorptivity=absorpt,
            contact_area=contact_area,
            contact_length=contact_length,
            internal_heat=heat_watts,
        )
        self.objects[(body_id, link_id)] = obj
        return obj

    def _hour(self):
        value = self.time_of_day
        if hasattr(value, "hour"):
            return value.hour + value.minute / 60.0 + value.second / 3600.0
        if isinstance(value, str):
            parts = [float(x) for x in value.split(":")]
            return parts[0] + (parts[1] if len(parts) > 1 else 0.0) / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0
        return float(value)

    def sundirection(self):
        hour = self._hour() % 24.0
        elevation = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))
        if elevation <= 0:
            return None
        azimuth = 2.0 * math.pi * (hour - 6.0) / 24.0
        horizontal = math.sqrt(max(1.0 - elevation * elevation, 0.0))
        return (horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), elevation)

    def tempmap(self):
        return {key: obj.temperature for key, obj in self.objects.items()}

    def _bodykey(self, body_id, link_id):
        key = (body_id, link_id)
        if key in self.objects:
            return key
        key = (body_id, -1)
        return key if key in self.objects else None

    def _contactrates(self, temperatures):
        rates = {key: 0.0 for key in self.objects}
        if p is None:
            return rates
        opts = {} if self.physics_client is None else {"physicsClientId": self.physics_client}
        try:
            points = p.getContactPoints(**opts)
        except Exception:
            return rates
        pairs = set()
        for point in points:
            first = self._bodykey(point[1], point[3])
            second = self._bodykey(point[2], point[4])
            if first is not None and second is not None and first != second:
                pairs.add(tuple(sorted((first, second))))
        for first, second in pairs:
            a, b = self.objects[first], self.objects[second]
            area = max(min(a.contact_area, b.contact_area), 1e-9)
            resistance = a.contact_length / (max(a.k, 1e-9) * area)
            resistance += b.contact_length / (max(b.k, 1e-9) * area)
            watts_to_a = (temperatures[second] - temperatures[first]) / resistance
            rates[first] += watts_to_a
            rates[second] -= watts_to_a
        return rates

    def update(self, dt, irradiance, wind=None, sun_direction=None):
        temperatures = self.tempmap()
        direction = self.sundirection() if sun_direction is None else sun_direction
        wind_speed = self.wind_speed if wind is None else float(wind)
        contact_rates = self._contactrates(temperatures)

        for key, obj in self.objects.items():
            shade = obj.sunlight(direction) if direction is not None and irradiance > 0 else 0.0
            obj.step(
                dt,
                ambient_temp=self.ambient,
                surroundings_temp=self.T_sky,
                wind_speed=wind_speed,
                solar_irradiance=irradiance,
                sun_fraction=shade,
                sun_direction=(0.0, 0.0, 1.0) if direction is None else direction,
                conductive_watts=contact_rates[key],
            )

    def compute_radiative(self, obj):
        if isinstance(obj, ThermalObject):
            return obj.longwaverate(self.T_sky) / obj.thermal_mass
        return 0.0

    def get_temperature(self, body_id, link_id=-1):
        obj = self.objects.get((body_id, link_id))
        return self.ambient if obj is None else obj.temperature

    def register_body(self, body_id, filename, per_link=False):
        material = ThermalMaterialLibrary.match_material_from_filename(filename)
        self.add_object(body_id, -1, material=material, init_T=material["T"])
        if not per_link or p is None:
            return
        opts = {} if self.physics_client is None else {"physicsClientId": self.physics_client}
        try:
            links = p.getNumJoints(body_id, **opts)
        except Exception:
            return
        for link_id in range(links):
            self.add_object(body_id, link_id, material=material, init_T=material["T"])
