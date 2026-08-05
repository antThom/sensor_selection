import numpy as np
import soundfile as sf

from sim.utils.CONSTANTS import SPEED_OF_SOUND


class SoundPointSource:
    def __init__(
        self,
        sound_file,
        dt,
        loop=True,
        position=None,
        velocity=None,
    ):
        self.pos = np.zeros((3, 1)) if position is None else position
        self.vel = np.zeros((3, 1)) if velocity is None else velocity
        self.volume = 1.0
        self.speed_of_sound = SPEED_OF_SOUND
        self.dt = dt

        # Load waveform
        self.waveform, self.sample_rate = sf.read(str(sound_file))
        if self.waveform.ndim == 1:
            self.waveform = self.waveform[:, np.newaxis]

        self.samples_per_step = int(self.sample_rate * dt)
        self.loop = loop
        self.active = False
        self.num_samples = len(self.waveform)

        # Playback state
        self.current_sample = 0

    def set_active(self, is_active: bool):
        self.active = is_active

    # def get_chunk(self, chunk_size):
    #     """Return next audio chunk attenuated by distance to observer."""
    #     if not self.active:
    #         return np.zeros((chunk_size, self.waveform.shape[1]), dtype='float32')

    #     # Compute distance attenuation

    #     start = self.current_sample
    #     end = start + chunk_size

    #     if end >= self.num_samples:
    #         if self.loop:
    #             part1 = self.waveform[start:]
    #             part2 = self.waveform[:end - self.num_samples]
    #             chunk = np.concatenate([part1, part2], axis=0)
    #             self.current_sample = end - self.num_samples
    #         else:
    #             chunk = np.zeros((chunk_size, self.waveform.shape[1]), dtype='float32')
    #             self.active = False
    #             return chunk
    #     else:
    #         chunk = self.waveform[start:end]
    #         self.current_sample = end

    # return chunk
    # """Continuously feed audio data to the sounddevice stream."""
    # chunk_size = int(self.sample_rate * self.dt)
    # with sd.OutputStream(samplerate=self.sample_rate,
    #                      channels=self.waveform.shape[1],
    #                      dtype='float32') as stream:
    #     while not self._stop.is_set():
    #         start = self.current_sample
    #         end = start + chunk_size

    #         if end >= self.num_samples:
    #             if self.loop:
    #                 part1 = self.waveform[start:]
    #                 part2 = self.waveform[:end - self.num_samples]
    #                 chunk = np.concatenate([part1, part2], axis=0)
    #                 self.current_sample = end - self.num_samples
    #             else:
    #                 chunk = self.waveform[start:]
    #                 self._stop.set()
    #         else:
    #             chunk = self.waveform[start:end]
    #             self.current_sample = end

    #         with self._lock:
    #             stream.write(chunk.astype(np.float32))

    #         # Maintain simulation-like timing
    #         time.sleep(self.dt)

    def get_chunk(self, chunk_size, advance=True):
        start = self.current_sample
        end = start + chunk_size
        if end >= self.num_samples:
            if self.loop:
                part1 = self.waveform[start:]
                part2 = self.waveform[: end - self.num_samples]
                chunk = np.concatenate([part1, part2], axis=0)
                if advance:
                    self.current_sample = end - self.num_samples
            else:
                chunk = np.zeros((chunk_size, self.waveform.shape[1]), dtype="float32")
                if advance:
                    self.active = False
                return chunk
        else:
            chunk = self.waveform[start:end]
            if advance:
                self.current_sample = end
        return chunk
