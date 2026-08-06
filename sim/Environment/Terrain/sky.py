import numpy as np
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Vec4,
)

from sim.utils.functions import string_to_time

light_temp = np.array([1000, 2700, 5500, 6500, 10000])
light_rgb = np.array(
    [
        (255, 56, 0, 1),
        (255, 169, 87, 1),
        (255, 184, 114, 1),
        (255, 190, 126, 1),
        (255, 193, 132, 1),
    ]
)


class SKY:
    def __init__(self, loader, config, render, day_length=24, time_of_day="08:00:00"):
        self.load_sky(config, loader, render, day_length, time_of_day)
        self.load_sun(config, loader, render, day_length, time_of_day)
        self.day_length = day_length
        self.start_time = string_to_time(time_of_day)

    def follow_camera(self, camera):
        self.object.setCompass(camera)

    def load_sun(self, config, loader, render, day_length, time_of_day):
        # Visible sun disk
        self.sun = loader.loadModel(
            config["light_source"].get("obj_path", "models/misc/sphere")
        )
        self.sun.reparentTo(render)
        self.sun.setScale(config["light_source"].get("obj_scale", 120))
        self.set_color(config["light_source"].get("color_temp", 5000))
        self.sun.setLightOff()
        self.sun.setDepthWrite(False)
        self.sun.setBin("background", 0)
        self.sun.setTwoSided(True)
        self.sun_distance = config["light_source"].get("distance", 5000)

        # Directional sunlight
        self.sun_light = DirectionalLight("sun_light")
        self.sun_light_np = render.attachNewNode(self.sun_light)
        # Set the direction of sunlight
        self.set_direction(day_length=day_length, time_of_day=time_of_day)
        render.setLight(self.sun_light_np)

        # Small ambient floor so objects do not go completely dark
        self.ambient_light = AmbientLight("ambient_light")
        self.ambient_light_np = render.attachNewNode(self.ambient_light)
        render.setLight(self.ambient_light_np)

    def load_sky(self, config, loader, render, day_length, time_of_day):
        self.sky = loader.loadModel(config["obj_path"])
        self.sky.reparentTo(render)
        self.sky.setScale(config.get("obj_scale", 5000))

        # View sphere from inside
        self.sky.setTwoSided(True)

        # Disable lighting and depth
        self.sky.setLightOff()
        self.sky.setDepthWrite(False)
        self.sky.setBin("background", 0)

        # Time of Day
        time_of_day = string_to_time(time_of_day)

        # Start sky as blue
        self.day_start = config["color_day"]  # blue
        self.night_end = config["color_night"]  # black
        self.twilight = config["color_twilight"]

        self.set_sky_color(time_of_day, day_length)

    def set_sky_color(self, time_of_day, day_length):
        t = (time_of_day.hour % day_length) / day_length
        fade = (1 - np.cos(np.pi * t)) / 2

        r = self.day_start[0] * (1.0 - fade) + self.night_end[0] * fade
        g = self.day_start[1] * (1.0 - fade) + self.night_end[1] * fade
        b = self.day_start[2] * (1.0 - fade) + self.night_end[2] * fade

        self.sky.setColor(r, g, b, 1.0)

    def set_color(self, color=5500):
        RED = np.interp(color, light_temp, light_rgb[:, 0]) / 255
        GREEN = np.interp(color, light_temp, light_rgb[:, 1]) / 255
        BLUE = np.interp(color, light_temp, light_rgb[:, 2]) / 255
        ALPHA = np.interp(color, light_temp, light_rgb[:, 3])
        self.sun.setColor(Vec4(RED, GREEN, BLUE, ALPHA))  # Warm sunlight color

    def set_direction(self, day_length, time_of_day, azimuth0=0.0, R=5000):
        time_of_day = string_to_time(time_of_day)
        t = (time_of_day.hour % day_length) / day_length
        azimuth = azimuth0 + 2 * np.pi * t

        # sunrise -> noon -> sunset -> night
        elevation = np.pi / 2 * max(0.0, np.sin(2 * np.pi * t - np.pi / 2))
        inclination = np.pi / 2 - elevation

        self.sun_light_np.setHpr(
            np.rad2deg(azimuth), np.rad2deg(inclination), 0
        )  # Heading, Pitch, Roll

        x, y, z = (
            R * np.cos(azimuth) * np.sin(inclination),
            R * np.sin(azimuth) * np.sin(inclination),
            R * np.sin(inclination),
        )
        self.sun.setPos(x, y, z)

    def _update_sun(self, task_time):
        # Normalized time of day: 0..1
        u = (
            (self.start_time.hour * 3600 + self.start_time.minute * 60 + task_time)
            % self.day_length
        ) / self.day_length
        # Sun path: east -> west, below horizon at night
        azimuth = 2.0 * np.pi * u - (np.pi / 2.0)
        elevation = np.pi / 2 * np.sin(2 * np.pi * u - np.pi / 2)

        # Position of the visible sun disk
        x = self.sun_distance * np.sin(elevation) * np.cos(azimuth)
        y = self.sun_distance * np.sin(elevation) * np.sin(azimuth)
        z = self.sun_distance * np.sin(elevation)
        self.sun.setPos(x, y, z)

        # Make the directional light point from the sun toward the scene
        self.sun_light_np.setPos(0, 0, 0)
        self.sun_light_np.lookAt(self.sun)

        # Sun brightness fades out at night
        sun_factor = max(0.0, np.sin(elevation))
        sun_r = 1.0 * sun_factor
        sun_g = 0.98 * sun_factor
        sun_b = 0.85 * sun_factor
        self.sun_light.setColor(Vec4(sun_r, sun_g, sun_b, 1.0))

        # Ambient light keeps the scene readable at night
        ambient = 0.03 + 0.17 * sun_factor
        self.ambient_light.setColor(Vec4(ambient, ambient, ambient, 1.0))

        # Fade sky from blue to black as the sun goes down
        blue_weight = sun_factor
        r = self.night_end[0] * (1.0 - blue_weight) + self.day_start[0] * blue_weight
        g = self.night_end[1] * (1.0 - blue_weight) + self.day_start[1] * blue_weight
        b = self.night_end[2] * (1.0 - blue_weight) + self.day_start[2] * blue_weight
        self.sky.setColor(r, g, b, 1.0)
