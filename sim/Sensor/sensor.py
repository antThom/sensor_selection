# sensor.py
from abc import ABC, abstractmethod
import json

class Sensor(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_output(self):
        """Return sensor output (override in subclasses)."""
        raise NotImplementedError

# Factory kept OUTSIDE the class to avoid circular imports
def load_sensor_from_file(filepath: str) -> Sensor:
    with open(filepath, "r") as f:
        cfg = json.load(f)
    t = cfg.get("type")
    if t == "camera":
        # Lazy import avoids circular dependency
        from sim.Sensor.Cameras.camera import Camera
        return Camera(cfg)
    raise ValueError(f"Unknown sensor type: {t!r}")
