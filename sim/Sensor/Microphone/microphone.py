import numpy as np
import pybullet as p
from scipy.signal import resample
from sim.Sensor.sensor import Sensor
from scipy.spatial.transform import Rotation as Rot
import sounddevice as sd
import soundfile as sf
import threading
import time
from sim.Sound.audio_mixer import AudioMixer
from sim.Constants import *

class MicrophoneSensor_Uniform(Sensor):
    def __init__(self, param: dict, name: str):
        super().__init__(param)  # keep config in base
        self.name             = name
        self.speed_of_sound   = speed_of_sound
        self.forward          = param.get("forward", [0,0,1])
        self.bias             = param.get("bias", 0)
        self.sensitivity      = param.get("sensitivity", 0.01) # V/Pa
        self.sample_rate      = param.get("sample_rate", 48e3) 
        self.noise_variance   = param.get("noise_variance", 2e-3) 
        self.temperature_coef = param.get("temperature_coef", 2e-3) 
        self.max_distance     = param.get("max_distance", 50) 
        self.attached_body    = True
        self.tf               = {}
        self.mixer            = None
        # self.mixer            = AudioMixer(sample_rate=self.sample_rate, dt=sim_dt)

    def set_audio_mixer(self, mixer):
        self.mixer = mixer

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

    def get_output(self):
        """Called by Sensor._capture_loop() in its own thread."""
        if self.agent is None:
            raise RuntimeError("Microphone must be attached to an agent before use.")
        output = self.sense()
        self.last_heard = output.copy()
        return output.copy()


    def sense(self):
        """Sample the current audio field from the environment mixer."""
        if self.agent is None or self.mixer is None:
            return np.zeros((int(self.sample_rate * sim_dt), 1), dtype=np.float32)

        # --- Compute mic world pose (consistent with Camera and other sensors) ---
        pos_agent = self.agent.position.flatten()
        quat_agent = (
            self.agent.orientation.flatten().tolist()
            if len(self.agent.orientation.flatten()) > 3
            else p.getQuaternionFromEuler(self.agent.orientation.flatten().tolist())
        )
        R_agent = Rot.from_quat(quat_agent)

        mount = self.agent.tf.get("body2Sensor", None)
        if mount:
            R_body2sensor, t_body2sensor = mount
            t_body2sensor = np.array(t_body2sensor)
        else:
            R_body2sensor = Rot.identity()
            t_body2sensor = np.zeros((3, 1))

        R_world2sensor = R_agent * R_body2sensor
        mic_pos = pos_agent + R_world2sensor.apply(t_body2sensor.flatten())

        # --- Query mixer for the current field at this position ---
        buffer = self.mixer.get_field_at(mic_pos)

        # Store last heard buffer for RL observation or visualization
        self.last_heard = buffer.copy()
        return buffer

    
    def doppler_shift_factor(self,source_pos, source_vel):
        direction = np.asarray(self.pos).reshape((-1,1)) - np.array(source_pos)
        direction /= np.linalg.norm(direction) + 1e-9  # unit vector

        v_s = np.dot(source_vel, direction)
        v_r = np.dot(self.vel, direction)

        factor = (self.speed_of_sound + v_r) / (self.speed_of_sound + v_s)
        return factor
