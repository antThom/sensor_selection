from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, LVector3, TextureStage, Loader
from panda3d.core import Filename, DirectionalLight, Vec4, Vec3
from sim.utils.string2time import string_to_time
import numpy as np
import sys

light_temp = np.array([1000, 2700, 5500, 6500, 10000])
light_rgb  = np.array([(255, 56, 0, 1), (255, 169, 87, 1), (255, 184, 114, 1), (255, 190, 126, 1), (255, 193, 132, 1)])


class SUN:
    def __init__(self, config, loader, render, day_length=24, time_of_day=12):
        self.sun = DirectionalLight(config["obj_path"])
        self.set_color(config["color_temp"])
        self.node = render.attachNewNode(self.sun)
        
        # Set the direction of sunlight 
        self.set_direction(day_length, time_of_day)

        render.setLight(self.node)
       # --- Add Ambient Light for soft shadows ---
        # ambient_light = AmbientLight("ambient_light")
        # ambient_light.setColor(Vec4(0.3, 0.3, 0.35, 1))  # Soft blueish ambient
        # ambient_light_node = self.render.attachNewNode(ambient_light)
        # self.render.setLight(ambient_light_node)

    def set_color(self, color=5500):
        RED = np.interp(color, light_temp, light_rgb[:,0])/255
        GREEN = np.interp(color, light_temp, light_rgb[:,1])/255
        BLUE = np.interp(color, light_temp, light_rgb[:,2])/255
        ALPHA = np.interp(color, light_temp, light_rgb[:,3])
        self.sun.setColor(Vec4(RED, GREEN, BLUE, ALPHA))  # Warm sunlight color

    def set_direction(self, day_length, time_of_day, azimuth0=0.0):
        time_of_day = string_to_time(time_of_day)
        t = (time_of_day.hour % day_length) / day_length
        azimuth = azimuth0 + 2 * np.pi * t

        # sunrise -> noon -> sunset -> night
        elevation = np.pi / 2 * max(0.0, np.sin(2 * np.pi * t - np.pi / 2))
        inclination = np.pi / 2 - elevation

        self.node.setHpr(azimuth, inclination, 0)  # Heading, Pitch, Roll
