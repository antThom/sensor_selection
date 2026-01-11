from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .world import World
from sim.Sensor.sensor_system import SensorSystem

class SensorSelectionEnv(gym.Env):
    """Thin Gym wrapper around World + SensorSystem.

    - World handles physics stepping and state.
    - SensorSystem runs threads and provides latest sensor data.
    - This env's step() should do almost no heavy work.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world: World,
        sensors: SensorSystem,
        decision_hz: float = 10.0,
        n_substeps: int = 1,
        max_steps: int = 1000,
        n_agents_blue: int = 1,
        n_agents_red: int = 1,
        n_sensors_per_agent: int = 1,
    ):
        super().__init__()
        self.world = world
        self.sensors = sensors
        self.decision_hz = float(decision_hz)
        self.n_substeps = int(n_substeps)
        self.max_steps = int(max_steps)

        # Example action space: sensor mask per agent + movement discrete per agent.
        # You should adapt this to your real control interface.
        self.action_space = spaces.Dict({
            "blue_sensor_mask": spaces.MultiBinary(n_sensors_per_agent * n_agents_blue),
            "red_sensor_mask": spaces.MultiBinary(n_sensors_per_agent * n_agents_red),
            "blue_move": spaces.MultiDiscrete([4] * n_agents_blue),  # e.g. forward/back/left/right
            "red_move": spaces.MultiDiscrete([4] * n_agents_red),
        })

        # Observation: minimal example: world state + selected sensor outputs snapshot keys.
        # In practice you'd build a vector/Dict and keep it fixed-size.
        self.observation_space = spaces.Dict({
            "world": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "sensors": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

        self._episode_steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        state = self.world.soft_reset(seed=seed)
        # Sensors keep running; selection/masks handled per-step.
        obs = self._build_obs(state)
        info = {}
        return obs, info

    def step(self, action: Dict[str, Any]):
        self._episode_steps += 1

        # 1) Apply sensor selection by enabling/disabling workers or setting masks
        self._apply_sensor_masks(action)

        # 2) Apply controls (movement)
        self.world.apply_controls(action)

        # 3) Step physics (possibly multiple substeps per decision step)
        self.world.step_physics(self.n_substeps)

        # 4) Build obs from latest sensor outputs + world state
        done = self._episode_steps >= self.max_steps
        state = self.world.get_state(done=done)
        obs = self._build_obs(state)

        reward = 0.0  # implement your reward here (use state + obs)
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _apply_sensor_masks(self, action: Dict[str, Any]) -> None:
        # This is the place to map an agent mask to actual sensor names.
        # Example: assume sensors are registered with names like "blue0_cam", etc.
        # Here we just demonstrate enabling all sensors if any mask bit is 1.
        # Replace with your real mapping.
        if not self.sensors.all_names():
            return

        # Example: if any sensors are selected, keep them enabled; otherwise disable all.
        any_on = bool(np.any(action.get("blue_sensor_mask", []))) or bool(np.any(action.get("red_sensor_mask", [])))
        for name in self.sensors.all_names():
            self.sensors.set_enabled(name, any_on)

    def _build_obs(self, state) -> Dict[str, Any]:
        # Keep this cheap and fixed-shape in production.
        # Placeholder: return small numeric placeholders.
        world_scalar = np.array([state.step_count], dtype=np.float32)
        sensors_scalar = np.array([len(self.sensors.all_names())], dtype=np.float32)
        return {"world": world_scalar, "sensors": sensors_scalar}

    def close(self):
        try:
            self.sensors.stop_all()
        except Exception:
            pass
        self.world.close()
