from __future__ import annotations

from typing import Optional, Dict, Any
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from gui.gui import CameraViewer  # your existing GUI window
from .world import World
from sim.Sensor.sensor_system import SensorSystem

class ViewerController:
    """Decoupled GUI: renders at fixed FPS and *pulls* latest data."""

    def __init__(self, world: World, sensors: SensorSystem, refresh_hz: float = 5.0):
        self.world = world
        self.sensors = sensors
        self.refresh_hz = float(refresh_hz)

        self.app = QApplication.instance() or QApplication([])
        self.gui = CameraViewer(physicsClientId=self.world.client_id)

        self._timer = QTimer()
        self._timer.timeout.connect(self.on_refresh)

        self._last_refresh_t = 0.0

    def start(self):
        interval_ms = max(10, int(1000.0 / self.refresh_hz))
        self._timer.start(interval_ms)
        self.gui.show()
        self.app.exec_()

    def on_refresh(self):
        # Pull world state (cheap)
        state = self.world.get_state(done=False)

        # Pull latest sensors (cheap)
        latest = self.sensors.snapshot()

        # Update GUI with selected agent only (your GUI already has selection logic)
        # For now, we simply call the GUI's refresh methods. Ideally you'd add a method
        # like gui.update(state, latest) to avoid extra Bullet queries.
        try:
            self.gui.refresh_current_agent()
        except Exception:
            pass
