from __future__ import annotations

import pybullet as p

from sim.world import World
from sim.Sensor.sensor_system import SensorSystem
from sim.viewer_app import ViewerController

def main():
    # Visualization: GUI
    world = World(bullet_mode=p.GUI, time_step=0.01)
    sensors = SensorSystem()
    sensors.register_sensor("dummy_sensor", capture_fn=lambda: 0.0, rate_hz=10.0, enabled=True)
    sensors.start_all()

    viewer = ViewerController(world=world, sensors=sensors, refresh_hz=5.0)
    viewer.start()

if __name__ == "__main__":
    main()
