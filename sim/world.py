from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pybullet as p
import pybullet_data

# Reuse your existing modules
from .Environment.environment import Environment
from .Agent.team import Team

@dataclass
class WorldState:
    t: float
    step_count: int
    blue: Dict[str, Any]
    red: Dict[str, Any]
    done: bool = False

class World:
    """Simulation core: physics + environment + teams. No Gym and no GUI.

    Key performance decision:
    - Build heavy world assets ONCE at construction.
    - Reset repositions bodies & clears internal state (soft reset).
    """

    def __init__(
        self,
        bullet_mode: int = p.DIRECT,
        time_step: float = 0.01,
        gravity: float = -9.81,
        env_config_path: str = "config/scene/flat_forrest/environment_config.yaml",
        blue_team_path: str = "config/scene/flat_forrest/BlueTeam_config.yaml",
        red_team_path: str = "config/scene/flat_forrest/RedTeam_config.yaml",
    ):
        self.client_id = p.connect(bullet_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        self.time_step = float(time_step)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(0, 0, gravity, physicsClientId=self.client_id)

        # Heavy loads once
        self.env = Environment(physicsClientId=self.client_id, config_path=env_config_path)
        self.blue_team = Team(filepath=blue_team_path, physicsClientId=self.client_id, team_name="Blue", thermal=self.env.thermal)
        self.red_team = Team(filepath=red_team_path, physicsClientId=self.client_id, team_name="Red", thermal=self.env.thermal)

        self.t = 0.0
        self.step_count = 0

    def soft_reset(self, seed: Optional[int] = None) -> WorldState:
        # Reset time counters
        self.t = 0.0
        self.step_count = 0

        # Delegate to existing reset hooks if present.
        # If your current code only resets during __init__, implement Team.reset()/Agent.reset() later.
        if hasattr(self.blue_team, "reset"):
            self.blue_team._reset_states(terrain_bound=self.env.terrain['terrain_bounds'][0,:], physicsClient=self.client_id, team=self.blue_team.team_color, seed=seed)
        if hasattr(self.red_team, "reset"):
            self.red_team._reset_states(terrain_bound=self.env.terrain['terrain_bounds'][0,:], physicsClient=self.client_id, team=self.red_team.team_color, seed=seed)

        return self.get_state(done=False)

    def step_physics(self, n_substeps: int = 1) -> None:
        for _ in range(int(n_substeps)):
            p.stepSimulation(physicsClientId=self.client_id)
            self.t += self.time_step
            self.step_count += 1

    def apply_controls(self, control_action: Any) -> None:
        """Apply movement/controls. Keep this minimal.

        Suggested pattern: control_action is a dict with keys like:
          {'blue': {...}, 'red': {...}}
        where per-team actions map agent_id->(vx, vy, yaw_rate) or discrete moves.

        Implement your control logic here; keep sensor selection out of this method.
        """
        # Placeholder: your existing env step moved agents directly.
        # Move this logic from SensorSelection_Env.step into here for speed & clarity.
        pass

    def get_state(self, done: bool) -> WorldState:
        blue = self.blue_team.get_states(self.red_team.agents) if hasattr(self.blue_team, "get_states") else {}
        red = self.red_team.get_states(self.blue_team.agents) if hasattr(self.red_team, "get_states") else {}
        return WorldState(t=self.t, step_count=self.step_count, blue=blue, red=red, done=done)

    def close(self) -> None:
        try:
            p.disconnect(physicsClientId=self.client_id)
        except Exception:
            pass
