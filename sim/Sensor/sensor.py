# sensor.py
from abc import ABC, abstractmethod
import threading
import time
import json
from sim.Environment.Thermal.thermal_manager import ThermalManager
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from dataclasses import dataclass
from typing import Any, Optional, Callable

class FrameSignal(QObject):
    new_frame = pyqtSignal(str, object, float)  # (sensor_name, image, timestamp)

@dataclass
class LatestValue:
    """Thread-safe single-slot buffer."""
    _lock: threading.Lock = threading.Lock()
    _value: Any = None
    _timestamp: float = 0.0
    _seq: int = 0

    def write(self, value: Any, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.perf_counter()
        with self._lock:
            self._value = value
            self._timestamp = timestamp
            self._seq += 1

    def read(self):
        with self._lock:
            return self._value, self._timestamp, self._seq
        
class SensorWorker:
    """
    Runs a capture_fn at a fixed rate in its own thread and stores results in a LatestValue buffer.
    """
    def __init__(
        self,
        name: str,
        capture_fn: Callable[[], object],
        out: LatestValue,
        rate_hz: float,
        enabled: bool = True,
    ):
        self.name = name
        self.capture_fn = capture_fn
        self.out = out
        self.rate_hz = float(rate_hz)
        self.enabled = enabled

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}_worker", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)

    def set_rate(self, rate_hz: float) -> None:
        self.rate_hz = float(rate_hz)

    def _run(self) -> None:
        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        next_t = time.perf_counter()

        while not self._stop_evt.is_set():
            if not self.enabled:
                time.sleep(0.05)
                next_t = time.perf_counter()
                continue

            # Capture
            try:
                sample = self.capture_fn()
            except Exception:
                # If you want: log this once per second instead of printing each time
                sample = None

        # Write latest (avoid copying unless needed)
        self.out.write(sample)

        # Sleep to maintain rate
        if period > 0:
            next_t += period
            now = time.perf_counter()
            sleep_dt = next_t - now
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                # We fell behind: reset schedule to avoid spiraling
                next_t = now
        else:
            # rate_hz <= 0: run as fast as possible (usually NOT what you want)
            time.sleep(0.0)

class Sensor(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.agent = None
        self.name = config.name
        self._capture_thread = None
        self._running = False
        self._rate_hz = config.get("frame_rate", 10)
        self.period = 1.0 / self._rate_hz
        self.latest = LatestValue()
        self._worker: Optional[SensorWorker] = None

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
            self._rate_hz = float(rate_hz)
            # self._period = 1.0 / self._rate_hz

        if self._worker is None:
            self._worker = SensorWorker(
                name=self.name,
                capture_fn=self.capture,
                out=self.latest,
                rate_hz=self._rate_hz,
                enabled=True,
            )
        else:
            self._worker.set_rate(self._rate_hz)

        self._worker.start()


        # self._running = True
        # self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        # self._capture_thread.start()
        print(f"[Sensor] Started capture loop at {self._rate_hz:.1f} Hz")

    def capture(self):
        """Override in subclasses; returns a sample (numpy array, scalar, dict, etc.)."""
        raise NotImplementedError
    
    # def _capture_loop(self):
    #     while self._running:
    #         start = time.time()
    #         try:
    #             output = self.get_output()
    #             with self._lock:
    #                 self.last_output = output.copy()
    #                 self.last_timestamp = start
    #             # self.last_output = self.get_output() 
    #             # self.last_timestamp = start 
    #         except Exception as e:
    #             print(f"[Sensor] Error during capture: {e}")
    #         # emit frame to GUI
    #         try:
    #             with self._lock:
    #                 self.signals.new_frame.emit(self.name, self.last_output, self.last_timestamp)
    #         except Exception as e:
    #             print(f"[Sensor] Frame emit error: {e}")
    #         elapsed = time.time() - start
    #         sleep_time = max(0, self._period - elapsed)
    #         time.sleep(sleep_time)

    def stop_capture(self):
        """Stop the periodic capture thread."""
        # if not self._running:
        #     return
        # self._running = False
        # if self._capture_thread:
        #     self._capture_thread.join(timeout=1.0)
        if self._worker:
            self._worker.stop()
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
