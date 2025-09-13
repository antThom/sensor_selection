import numpy as np
import pybullet as p
from scipy.signal import resample

class MicrophoneSensor_Uniform:
    def __init__(self, position, velocity, sample_rate=44100, max_distance=50.0, attached_body=None):
        self.pos = np.array(position) if position is not None else None
        self.vel = velocity
        self.attached_body = attached_body
        # self.local_offset = np.array(local_offset)
        self.sample_rate = sample_rate
        self.max_distance = max_distance
        # self.directional = directional  # future support
        self.last_heard = None  # store last buffer (for logging or analysis)
        self.speed_of_sound=343.0

    def get_world_position(self):
        if self.attached_body is not None:
            pos, orn = p.getBasePositionAndOrientation(self.attached_body)
            return np.array(pos)
        return self.pos
    
    def set_pos_vel(self, pos, vel):
        if self.attached_body is not None:
            pos, orn = p.getBasePositionAndOrientation(self.attached_body)
        self.pos = pos
        self.vel = vel

    def sense(self, emitters, samples_per_step):
        mic_pos = self.position
        if mic_pos is None:
            raise ValueError("Microphone position is undefined")

        buffer = np.zeros((samples_per_step, 1), dtype=np.float32)

        for emitter in emitters:
            if emitter.current_sample >= len(emitter.waveform):
                continue  # sound finished

            source_pos, source_vel = emitter.get_pos_vel()
            distance = np.linalg.norm(source_pos - mic_pos)
            if distance > self.max_distance:
                continue  # too far to hear

            # Check LOS
            if emitter.occlusion_check:
                result = p.rayTest(source_pos.tolist(), mic_pos.tolist())[0]
                if result[0] != -1:
                    continue  # occluded

            # Volume attenuation
            attenuation = emitter.volume / (distance**2 + 1e-6)
            attenuation = min(attenuation, 1.0)

            # Get chunk
            start = emitter.current_sample
            end = start + samples_per_step
            if end > len(emitter.waveform):
                if emitter.loop:
                    part1 = emitter.waveform[start:]
                    part2 = emitter.waveform[:end - len(emitter.waveform)]
                    chunk = np.concatenate([part1, part2], axis=0)
                else:
                    chunk = emitter.waveform[start:]
                    if len(chunk) < samples_per_step:
                        pad = np.zeros((samples_per_step - len(chunk), 1))
                        chunk = np.vstack((chunk, pad))
            else:
                chunk = emitter.waveform[start:end]

            # Accumulate into buffer
            buffer += chunk * attenuation

        # Normalize and clip
        buffer = np.clip(buffer, -1.0, 1.0)
        self.last_heard = buffer.copy()
        return buffer
    
    def doppler_shift_factor(self,source_pos, source_vel):
        direction = np.asarray(self.pos).reshape((-1,1)) - np.array(source_pos)
        direction /= np.linalg.norm(direction) + 1e-9  # unit vector

        v_s = np.dot(source_vel, direction)
        v_r = np.dot(self.vel, direction)

        factor = (self.speed_of_sound + v_r) / (self.speed_of_sound + v_s)
        return factor
