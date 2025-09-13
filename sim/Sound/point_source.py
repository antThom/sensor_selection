import numpy as np
import pybullet as p
import pybullet_data
import time
import json
import os
from pathlib import Path
import simpleaudio as sa
import soundfile as sf
import sounddevice as sd

class SoundPointSource():
    def __init__(self,sound_file,dt):
        self.pos = np.zeros((3,1))
        self.vel = np.zeros((3,1))
        self.volume = 1.0
        self.speed_of_sound = 343.0 # m/s
        self.dt = dt
        # Loade waveform into memory
        self.waveform, self.sample_rate = sf.read(str(sound_file))
        if self.waveform.ndim == 1:
            self.waveform = np.expand_dims(self.waveform, axis=1)  # Ensure 2D shape: (samples, channels)

        self.samples_per_step = int(self.sample_rate * dt)
        self.current_sample = 0
        self.playing = False
        self.observer_volume = 0
        self.loop = False

    def reset(self):
        self.current_sample = 0
        self.playing = False

    def set_looping(self,loop):
        self.loop = loop

    def set_pos_vel(self,pos,vel):
        self.pos = pos
        self.vel = vel

    def set_volume(self,volume):
        self.volume = volume

    def get_pos_vel(self):
        return self.pos, self.vel

    def adjust_volume_for_distance(self,distance):
        self.observer_volume = min( self.volume / (distance ** 2 + 1e-6), 1.0 )

    def play_current_step(self):
        start = self.current_sample
        end = min(start + self.samples_per_step, len(self.waveform))
        
        # Handle Looping
        if end >= len(self.waveform):
            if self.loop:
                part1 = self.waveform[start:]
                part2 = self.waveform[:end - len(self.waveform)]
                chunk = np.concatenate([part1, part2], axis=0)
                self.current_sample = end - len(self.waveform)

            else:
                chunk = self.waveform[start:]
                self.current_sample = len(self.waveform)
        else:
            chunk = self.waveform[start:end]
            self.current_sample = end

        # Play the sound
        sd.play(chunk, samplerate=self.sample_rate, blocking=False)

        self.current_sample = end