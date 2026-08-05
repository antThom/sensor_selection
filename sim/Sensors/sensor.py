"""Shared sensor base types and legacy file-based sensor construction."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from sim.Environment.Thermal.thermal_manager import ThermalManager
from sim.rendering.object import RenderableObject


class SensorType(Enum):
    """Stable identifiers used by sensor loaders and camera controls."""

    EMPTY = None
    DUMMY = "dummy"
    CAMERA = "camera"
    RBGCAMERA = "rgbcamera"
    EOCAMERA = RBGCAMERA
    IRCAMERA = "ircamera"
    MICROPHONE = "microphone"


class Sensor(ABC, RenderableObject):
    """Base class for rendered sensors that may be attached to an agent."""

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.tf = {}
        self.agent = None
        self.type = SensorType.EMPTY

    def attach_to_agent(self, agent):
        """Record the owning agent and return this sensor for fluent setup."""
        self.agent = agent
        return self


def load_sensor_from_file(
    filepath: str | Path,
    name: str | None = None,
    thermal_mgr: ThermalManager | None = None,
) -> Sensor:
    """Construct a sensor from a legacy JSON configuration file.

    New scene loading should use :class:`sim.loaders.sensor_loader.SensorLoader`.
    This adapter remains for ``old_agent.py`` and applies the same current
    sensor classes without relying on their removed historical constructors.
    """
    path = Path(filepath)
    with path.open(encoding="utf-8") as stream:
        configuration = json.load(stream)

    sensor_type = str(configuration.get("type", "")).lower().replace(" ", "_")
    if sensor_type in {"camera", "eo_camera", "rgb_camera"}:
        from sim.Sensors.Cameras.eo_camera import EOCamera

        sensor = EOCamera()
    elif sensor_type in {"ir_camera", "ircamera"}:
        from sim.Sensors.Cameras.ir_camera import IRCamera

        sensor = IRCamera(thermal_manager=thermal_mgr)
    elif sensor_type == "microphone":
        from sim.Sensors.Microphone.microphone import MicrophoneSensor_Uniform

        return MicrophoneSensor_Uniform(
            configuration,
            name or configuration.get("name", path.stem),
        )
    elif sensor_type == "dummy":
        from sim.Sensors.dummy_sensor import DummySensor

        sensor = DummySensor()
    else:
        raise ValueError(f"unknown sensor type {sensor_type!r} in {path}")

    sensor.set_configs(configuration)
    if name is not None:
        sensor.name = name
    return sensor
