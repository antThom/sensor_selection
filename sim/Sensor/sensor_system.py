from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from .latest_value import LatestValue
from .sensor_worker import SensorWorker

@dataclass
class SensorHandle:
    name: str
    latest: LatestValue
    worker: SensorWorker
    rate_hz: float

class SensorSystem:
    """Owns sensor workers and provides a lightweight observation API.

    Design:
    - Sensors run at their own rates and write to LatestValue buffers.
    - RL/environment can pull latest outputs any time (no capture in env.step).
    - Sensor selection toggles .enabled, without destroying threads (fast).
    """

    def __init__(self):
        self._sensors: Dict[str, SensorHandle] = {}

    def register_sensor(self, name: str, capture_fn, rate_hz: float, enabled: bool = True) -> None:
        if name in self._sensors:
            raise ValueError(f"Sensor already registered: {name}")
        latest = LatestValue()
        worker = SensorWorker(name=name, capture_fn=capture_fn, out=latest, rate_hz=rate_hz, enabled=enabled)
        self._sensors[name] = SensorHandle(name=name, latest=latest, worker=worker, rate_hz=float(rate_hz))

    def start_all(self) -> None:
        for h in self._sensors.values():
            h.worker.start()

    def stop_all(self) -> None:
        for h in self._sensors.values():
            h.worker.stop()

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._sensors[name].worker.set_enabled(enabled)

    def set_rate(self, name: str, rate_hz: float) -> None:
        h = self._sensors[name]
        h.rate_hz = float(rate_hz)
        h.worker.set_rate(rate_hz)

    def read(self, name: str) -> Tuple[Any, float, int]:
        return self._sensors[name].latest.read()

    def snapshot(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        keys = names if names is not None else list(self._sensors.keys())
        for k in keys:
            v, ts, seq = self._sensors[k].latest.read()
            out[k] = v
        return out

    def all_names(self) -> List[str]:
        return list(self._sensors.keys())
