from __future__ import annotations

import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.world import World
from sim.Sensor.sensor_system import SensorSystem
from sim.rl_env import SensorSelectionEnv

def build_env():
    # Fast training: DIRECT
    world = World(bullet_mode=p.DIRECT, time_step=0.01)
    sensors = SensorSystem()

    # Example: register a placeholder sensor that reads nothing (replace with real sensor.capture)
    sensors.register_sensor("dummy_sensor", capture_fn=lambda: 0.0, rate_hz=50.0, enabled=True)
    sensors.start_all()

    env = SensorSelectionEnv(world=world, sensors=sensors, decision_hz=10.0, n_substeps=5)
    return env

def main():
    vec_env = DummyVecEnv([build_env])
    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("ppo_sensorselection")

if __name__ == "__main__":
    main()
