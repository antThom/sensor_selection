import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

@dataclass
class LatestValue:
    """Thread-safe single-slot buffer for latest sensor output."""
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

    def read(self) -> Tuple[Any, float, int]:
        with self._lock:
            return self._value, self._timestamp, self._seq
