"""_summary_
This loads camera and agent controls (agents haven't been implemented yet)
"""

from math import pi, sin, cos
from direct.task import Task


class AgentLoader:
    """_summary_
    Agent loader and manager for the simulation. Loads all agents and moveable parts into the simulation.
    Currently has no controllable agents :(
    """

    def __init__(self, config, world):
        """Sets up internal variables"""
        self.world = world
        self.config = config

        # Implement logic for actual agents

    def load_agents(self):
        """Loads agents."""
        self.world.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        """From Panda3D's quck start tutorial"""
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.world.camera.setPos(350 * sin(angleRadians), -350 * cos(angleRadians), 150)
        self.world.camera.setHpr(angleDegrees, -15, 0)

        self.world.sky.sky._update_sun(task.time)
        return Task.cont
