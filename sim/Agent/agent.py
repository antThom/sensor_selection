import numpy as np

from sim.Environment.ThermalObject import ThermalBody
from sim.rendering.object import RenderableObject


class Agent(ThermalBody, RenderableObject):
    """Rendered autonomous platform with sensors and a managed thermal body."""

    def __init__(self, thermal_manager=None):
        super().__init__(thermal_manager=thermal_manager)

        # Position and orientation are inherited from RenderableObject. The
        # remaining dynamics fields are agent-specific and retain array shapes
        # expected by the microphone and control integrations.
        self.agent_id = None
        self.velocity = np.zeros((3, 1))
        self.angular_rates = np.zeros((3, 1))
        self.mass = 1.0
        self.inertia = np.eye(3)
        self.tf = {}
        self.max_vel = 1.0
        self.sensor_list = []
        self.model_list = []

    def get_id(self):
        return getattr(self, "agent_id", None)

    """
    The add_* functions will have magic configuration. They will sort through almost anything and find what they're looking for.
    Otherwise, they'll throw an error.
    """

    def add_model(self, model):
        """Register an auxiliary model with this agent."""
        self.model_list.append(model)
        return model

    def add_sensor(self, sensor, name=None):
        """Attach a configured sensor to this agent.

        The sensor keeps a reference to its owner so camera setup can parent
        the optical node to the agent's Panda3D node. The optional registry
        name is retained for configuration/debug output.
        """
        sensor.attach_to_agent(self)
        if name is not None:
            sensor.registry_name = str(name)
        self.sensor_list.append(sensor)
        return sensor

    def get_sensor(self, name):
        """Return an attached sensor by registry or configured name."""
        for sensor in self.sensor_list:
            if getattr(sensor, "registry_name", None) == name:
                return sensor
            if sensor.name == name:
                return sensor
        raise KeyError(f"agent has no sensor named {name!r}")
