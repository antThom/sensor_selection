import threading
import time
from typing import Callable, Optional

from .latest_value import LatestValue

class SensorWorker:
    """Runs capture_fn in a background thread at rate_hz and stores output in LatestValue."""

    def __init__(
        self,
        name: str,
        capture_fn: Callable[[], object],
        out: LatestValue,
        rate_hz: float,
        enabled: bool = True,
        catch_exceptions: bool = True,
    ):
        self.name = name
        self.capture_fn = capture_fn
        self.out = out
        self.rate_hz = float(rate_hz)
        self.enabled = bool(enabled)
        self.catch_exceptions = bool(catch_exceptions)

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

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def _run(self) -> None:
        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        next_t = time.perf_counter()

        while not self._stop_evt.is_set():
            if not self.enabled:
                time.sleep(0.05)
                next_t = time.perf_counter()
                continue

            try:
                sample = self.capture_fn()
            except Exception:
                if not self.catch_exceptions:
                    raise
                sample = None

            self.out.write(sample)

            if period > 0:
                next_t += period
                now = time.perf_counter()
                sleep_dt = next_t - now
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                else:
                    next_t = now
            else:
                time.sleep(0.0)
