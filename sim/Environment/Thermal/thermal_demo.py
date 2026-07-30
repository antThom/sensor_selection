import argparse
import atexit
import math
import sys
from pathlib import Path


args = sys.argv[1:]
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    import pybullet as p
    from ursina import AmbientLight, DirectionalLight, Entity, Sky, Text, Ursina
    from ursina import Vec3, camera, clamp, color, held_keys, mouse, time, window
    from ursina.shaders import lit_with_shadows_shader
except ImportError as error:
    raise SystemExit(
        "Install the demo dependencies with: pip install -r requirements.txt"
    ) from error

from sim.Environment.Thermal.thermal_static_object import ThermalStaticObject


demo = None


class FlyCamera(Entity):
    def __init__(self):
        super().__init__(position=(6, 3.5, -8), rotation_y=-32)
        self.pitch = 12.0
        self.speed = 6.0
        camera.parent = self
        camera.position = (0, 0, 0)
        camera.rotation = (self.pitch, 0, 0)
        mouse.locked = True

    def update(self):
        if mouse.locked:
            self.rotation_y += mouse.velocity[0] * 70.0
            self.pitch = clamp(
                self.pitch - mouse.velocity[1] * 70.0,
                -85.0,
                85.0,
            )
            camera.rotation_x = self.pitch
        move = self.forward * (held_keys["w"] - held_keys["s"])
        move += self.right * (held_keys["d"] - held_keys["a"])
        move += Vec3(
            0,
            held_keys["space"] - held_keys["control"],
            0,
        )
        if move.length() > 0:
            boost = 2.0 if held_keys["shift"] else 1.0
            self.position += (
                move.normalized() * self.speed * boost * time.dt
            )

    def input(self, key):
        if key == "tab":
            mouse.locked = not mouse.locked


class ThermalDemo:
    def __init__(self, options):
        self.options = options
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.ambient = 270.0
        self.surface_temp = 285.0
        self.sun_direction = self._unit((0.45, -0.30, 0.84))
        self.sim_seconds = 0.0

        ground_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[15, 15, 0.05],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            basePosition=[0, 0, -0.05],
            physicsClientId=self.client,
        )

        self.ground = Entity(
            model="plane",
            texture="white_cube",
            texture_scale=(30, 30),
            scale=30,
            color=color.rgb(92, 112, 91),
            shader=lit_with_shadows_shader,
        )
        self.cube_view = Entity(
            model="cube",
            texture="white_cube",
            scale=1,
            color=color.rgb(193, 111, 70),
            shader=lit_with_shadows_shader,
        )
        self.cube = ThermalStaticObject.box(
            size=(1, 1, 1),
            position=(0, 0, 0.5),
            client_id=self.client,
            visual=self.cube_view,
            thermal_mass=35.0,
            temperature=300.0,
            specific_heat=850.0,
            conductivity=15.0,
            emissivity=0.82,
            absorptivity=0.72,
            contact_area=1.0,
            contact_length=0.5,
        )

        Sky(color=color.rgb(154, 191, 218))
        self.sun_view = Entity(
            model="sphere",
            scale=1.2,
            position=(7, 11, -5),
            color=color.rgb(255, 214, 99),
            unlit=True,
        )
        self.sunlight = DirectionalLight(shadows=True)
        self.sunlight.look_at(Vec3(-0.45, -0.84, 0.30))
        self.ambient_light = AmbientLight(
            color=color.rgba(105, 116, 128, 255)
        )
        FlyCamera()

        self.readout = Text(
            position=(-0.86, 0.45),
            origin=(-0.5, 0.5),
            scale=1.05,
        )
        self.update(0.0)

    @staticmethod
    def _unit(values):
        length = math.sqrt(sum(float(value) ** 2 for value in values))
        return tuple(float(value) / length for value in values)

    def update(self, real_dt=None):
        elapsed = time.dt if real_dt is None else real_dt
        simulation_dt = max(
            float(elapsed) * self.options.time_scale,
            0.0,
        )
        steps = max(1, int(math.ceil(simulation_dt / 2.0)))
        step_dt = simulation_dt / steps
        for _ in range(steps):
            shade = self.cube.sunlight(self.sun_direction)
            self.cube.step(
                step_dt,
                ambient_temp=self.ambient,
                surroundings_temp=self.ambient,
                wind_speed=self.options.wind,
                solar_irradiance=900.0,
                sun_fraction=shade,
                sun_direction=self.sun_direction,
                contact_temp=self.surface_temp,
            )
            self.sim_seconds += step_dt
        self.cube.sync_visual()
        self._set_color()
        rate = self.cube.last_rates["total"]
        self.readout.text = (
            f"Cube {self.cube.temperature:6.2f} K    "
            f"Net heat {rate:+7.1f} W    "
            f"Sim time {self.sim_seconds / 60.0:5.1f} min"
        )

    def _set_color(self):
        amount = clamp(
            (self.cube.temperature - 260.0) / 80.0,
            0.0,
            1.0,
        )
        cold = (57, 118, 153)
        hot = (210, 74, 58)
        rgb = [
            int(cold[index] + amount * (hot[index] - cold[index]))
            for index in range(3)
        ]
        self.cube_view.color = color.rgb(*rgb)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)


def main():
    global demo
    parser = argparse.ArgumentParser(
        description="Panda3D/PyBullet thermal cube demo"
    )
    parser.add_argument(
        "--wind",
        type=float,
        default=3.0,
        help="constant wind speed in m/s",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=120.0,
        help="simulated seconds per real second",
    )
    options = parser.parse_args(args)

    app = Ursina(
        title="Thermal Object Demo",
        borderless=False,
        development_mode=False,
    )
    window.color = color.rgb(154, 191, 218)
    demo = ThermalDemo(options)
    atexit.register(demo.close)
    app.run()


def update():
    if demo is not None:
        demo.update()


if __name__ == "__main__":
    main()
