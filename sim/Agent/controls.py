"""
Files creating the controls for using and moving the agent around.
"""

import agent
from direct.task import Task
from panda3d import ShowBase


class Controls:
    """
    Implements the controls for the overhead class. Takes the overhead world class as an input to modify and
    handle commands for.
    """

    def __init__(self, world):
        self.world = world  # We want to use the overhead object's ShowBase

    def init_tasks(self):
        """Call on class initalization. Required to register controls into panda3D"""
        pass

    def update_camera_orentation(self, pitch, roll, yaw):
        """Sets the camera orentation. Euler overload version."""
        pass

    def update_camera_orentation(self, i, j, k, l):
        """Sets the camera orentation. Quaternion version"""
        pass

    def update_camera_position(self):
        pass

    def update_agent_orentation(self):
        pass

    def update_agent_position(self):
        pass

    def get_user_input(self):
        pass
