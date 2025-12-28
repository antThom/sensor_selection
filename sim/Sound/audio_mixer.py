import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
from sim.Sound.point_source import SoundPointSource
from sim.Constants import *
import pybullet as p

class AudioMixer:
    """
    Mixes all SoundPointSources into a single combined audio field.
    The mixed field is stored internally and can be queried by microphones.
    """
    def __init__(self, sample_rate=44100, dt=sim_dt):
        self.sample_rate = sample_rate
        self.dt = dt
        self.sources = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._mixed_buffer = np.zeros((int(self.sample_rate * self.dt), 1), dtype=np.float32)

    def add_source(self, source: SoundPointSource):
        if source.sample_rate != self.sample_rate:
            raise ValueError("All sources must have same sample rate")
        with self._lock:
            self.sources.append(source)

    def remove_inactive(self):
        with self._lock:
            self.sources = [s for s in self.sources if s.active]

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._mix_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        sd.stop()

    def update(self):
        """Compute next mixed frame from all active sources."""
        chunk_size = int(self.sample_rate * self.dt)
        mix = np.zeros((chunk_size, 1), dtype=np.float32)
        active_sources = []
        for src in self.sources:
            if src.active:
                mix += src.get_chunk(chunk_size)
                active_sources.append(src)
        self.sources = active_sources
        self._mixed_buffer = np.clip(mix, -1.0, 1.0)

    def _mix_loop(self):
        chunk_size = int(self.sample_rate * self.dt)
        while not self._stop.is_set():
            with self._lock:
                mix = np.zeros((chunk_size, 1), dtype=np.float32)
                active_sources = []
                for src in self.sources:
                    if src.active:
                        mix += src.get_chunk(chunk_size)
                        active_sources.append(src)
                self.sources = active_sources
                self._mixed_buffer = np.clip(mix, -1.0, 1.0)
            time.sleep(self.dt)

    def get_field_at(self, position):
        with self._lock:
            return self._mixed_buffer.copy()

    # def get_field_at(self, position):
    #     """
    #     Compute the sound pressure at a microphone position.
    #     Includes distance attenuation and propagation delay.
    #     """
    #     if not self.sources:
    #         return np.zeros_like(self._mixed_buffer)
        
    #     position = np.array(position).reshape((3,))
    #     chunk_size = int(self.sample_rate * self.dt)
    #     mic_buffer = np.zeros((chunk_size, 1), dtype=np.float32)

    #     with self._lock:
    #         for src in self.sources:
    #             if not src.active:
    #                 continue

    #             # Get emitter position
    #             src_pos = np.array(src.pos).flatten()
    #             distance = np.linalg.norm(src_pos - position)
    #             if distance < 1e-6:
    #                 distance = 1e-6  # avoid div/0

    #             # --- Optional: Occlusion check ---
    #             if getattr(src, "occlusion_check", False):
    #                 try:
    #                     hit = p.rayTest(src_pos.tolist(), position.tolist())[0]
    #                     if hit[0] != -1:
    #                         continue  # blocked
    #                 except Exception:
    #                     pass

    #             # --- Distance attenuation ---
    #             # attenuation = 1.0 / (distance + 1.0)  # simple spherical model
    #             # attenuation = np.exp(-0.05 * distance) / (distance + 1.0)
    #             attenuation = 1.0

    #             # --- Propagation delay ---
    #             delay_seconds = distance / speed_of_sound
    #             delay_samples = int(delay_seconds * self.sample_rate)

    #             # Get the current waveform chunk
    #             # chunk = src.get_chunk(chunk_size)
    #             chunk = src.get_chunk(chunk_size, advance=False)


    #             # Apply delay (zero-pad front)
    #             # if delay_samples > 0:
    #             #     if delay_samples < chunk_size:
    #             #         chunk = np.vstack(
    #             #             [np.zeros((delay_samples, 1)), chunk[: chunk_size - delay_samples]]
    #             #         )
    #             #     else:
    #             #         chunk = np.zeros((chunk_size, 1))

    #             mic_buffer += attenuation * chunk

    #     # Normalize and clip
    #     mic_buffer = np.clip(mic_buffer, -1.0, 1.0)
    #     return mic_buffer.copy()
