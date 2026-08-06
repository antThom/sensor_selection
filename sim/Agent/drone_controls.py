"""Continuous keyboard controls for a rendered drone agent."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from panda3d.core import ClockObject


class DroneControls:
    """Move the first loaded agent while keeping simulation state synchronized."""

    MOVEMENT_KEYS: ClassVar[dict[str, str]] = {
        "w": "forward",
        "s": "backward",
        "a": "left",
        "d": "right",
        "space": "up",
        "shift": "down",
        "arrow_left": "turn_left",
        "arrow_right": "turn_right",
        "arrow_up": "pitch_up",
        "arrow_down": "pitch_down",
    }

    def __init__(self, world, agent=None, move_speed=20.0, rotation_speed=60.0):
        self.world = world
        self.agent = agent if agent is not None else self._first_agent(world)
        self.move_speed = float(move_speed)
        self.rotation_speed = float(rotation_speed)
        self.clock = ClockObject.getGlobalClock()
        self.key_state = dict.fromkeys(self.MOVEMENT_KEYS.values(), False)

        for key, action in self.MOVEMENT_KEYS.items():
            world.accept(key, self.set_key, [action, True])
            world.accept(f"{key}-up", self.set_key, [action, False])

        world.taskMgr.add(self.update, "update-drone-controls")

    @staticmethod
    def _first_agent(world):
        return world.agent_list[0] if world.agent_list else None

    def set_key(self, action, active):
        """Record a press or release for one continuous movement action."""
        self.key_state[action] = bool(active)

    def update(self, task):
        """Apply held controls once per rendered frame."""
        self.apply_motion(self.clock.getDt())
        return task.cont

    def apply_motion(self, dt):
        """Advance the controlled agent by ``dt`` seconds."""
        if self.agent is None or self.agent.object_node_path is None:
            return

        node = self.agent.object_node_path
        dt = max(0.0, float(dt))
        forward = float(self.key_state["forward"] - self.key_state["backward"])
        right = float(self.key_state["right"] - self.key_state["left"])
        vertical = float(self.key_state["up"] - self.key_state["down"])

        # Horizontal movement follows the drone's heading while altitude stays
        # aligned with the world's Z axis, independent of camera/drone pitch.
        heading = math.radians(float(node.getH(self.world.render)))
        world_x = right * math.cos(heading) + forward * math.sin(heading)
        world_y = forward * math.cos(heading) - right * math.sin(heading)
        direction = np.asarray([world_x, world_y, vertical], dtype=np.float64)
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 0.0:
            displacement = direction / magnitude * self.move_speed * dt
            current = node.getPos(self.world.render)
            node.setPos(
                self.world.render,
                current.x + float(displacement[0]),
                current.y + float(displacement[1]),
                current.z + float(displacement[2]),
            )
            self.agent.velocity = (direction / magnitude * self.move_speed).reshape(
                3, 1
            )
        else:
            self.agent.velocity = np.zeros((3, 1))

        heading_input = float(
            self.key_state["turn_right"] - self.key_state["turn_left"]
        )
        pitch_input = float(self.key_state["pitch_up"] - self.key_state["pitch_down"])
        node.setH(node.getH() + heading_input * self.rotation_speed * dt)
        node.setP(node.getP() + pitch_input * self.rotation_speed * dt)

        position = node.getPos(self.world.render)
        self.agent.position = np.asarray(
            [[position.x], [position.y], [position.z]], dtype=np.float64
        )
        self.agent.orientation = np.asarray(
            [[node.getH()], [node.getP()], [node.getR()]], dtype=np.float64
        )
        self.agent.sync_thermal_position()
