# sensor.py
from abc import ABC, abstractmethod
import threading
import time
import json
from sim.Environment.Thermal.thermal_manager import ThermalManager
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal

class FrameSignal(QObject):
    new_frame = pyqtSignal(str, object, float)  # (sensor_name, image, timestamp)


class Sensor(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.agent = None
        self._capture_thread = None
        self._running = False
        self._rate_hz = config.get("frame_rate", 10)
        self.period = 1.0 / self._rate_hz
        self.last_output = None
        self.last_timestamp = None
        self.tf       = {}
        self.signals = FrameSignal()
        self._lock = threading.Lock()


    @abstractmethod
    def get_output(self):
        """Return sensor output (override in subclasses)."""
        raise NotImplementedError

    def attach_to_agent(self, agent):
        self.agent = agent

    # --------- FIXED-RATE CAPTURE LOOP ----------
    def start_capture(self, rate_hz=None):
        """Begin periodic sampling in a background thread."""
        if self._running:
            return  # already running

        if rate_hz is not None:
            self._rate_hz = rate_hz
            self._period = 1.0 / self._rate_hz

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print(f"[Sensor] Started capture loop at {self._rate_hz:.1f} Hz")

    def _capture_loop(self):
        while self._running:
            start = time.time()
            try:
                output = self.get_output()
                with self._lock:
                    self.last_output = output.copy()
                    self.last_timestamp = start
                # self.last_output = self.get_output() 
                # self.last_timestamp = start 
            except Exception as e:
                print(f"[Sensor] Error during capture: {e}")
            # emit frame to GUI
            try:
                with self._lock:
                    self.signals.new_frame.emit(self.name, self.last_output, self.last_timestamp)
            except Exception as e:
                print(f"[Sensor] Frame emit error: {e}")
            elapsed = time.time() - start
            sleep_time = max(0, self._period - elapsed)
            time.sleep(sleep_time)

    def stop_capture(self):
        """Stop the periodic capture thread."""
        if not self._running:
            return
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        print(f"[Sensor] Capture loop stopped")

# Factory kept OUTSIDE the class to avoid circular imports
def load_sensor_from_file(filepath: str, name: str, thermal_mgr: ThermalManager=None) -> Sensor:
    with open(filepath, "r") as f:
        cfg = json.load(f)
    sensor_type = cfg.get("type")
    if sensor_type == "camera":
        # Lazy import avoids circular dependency
        from sim.Sensor.Cameras.camera import Camera
        return Camera(cfg,name)
    elif sensor_type == "ir_camera":
        from sim.Sensor.Cameras.ir_camera import IRCamera
        return IRCamera(cfg,name,thermal_mgr)
    elif sensor_type == "microphone":
        from sim.Sensor.Microphone.microphone import MicrophoneSensor_Uniform
        return MicrophoneSensor_Uniform(cfg,name)
    raise ValueError(f"Unknown sensor type: {sensor_type!r}")
