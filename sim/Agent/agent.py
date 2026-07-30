import numpy as np

from sim.Environment.ThermalObject import ThermalBody
from sim.rendering.object import RenderableObject


class Agent(ThermalBody, RenderableObject):
    def __init__(self, thermal_manager=None):
        super().__init__(thermal_manager=thermal_manager)

        """
        Most of these variables contained here are class defaults
        All of these will be configured by the `AgentLoader` class later and through its
        intermediates and helpers
        """

        self.sensor_list = list()
        self.model_list = list()

        # Math
        self.position = np.zeros((3, 1))
        self.velocity = np.zeros((3, 1))
        self.orientation = np.zeros((3, 1))
        self.angular_rates = np.zeros((3, 1))
        self.mass = 1
        self.inertia = np.eye(3)
        self.tf = {}
        self.max_vel = 1

    def get_id(self):
        return getattr(self, "agent_id", None)

    """
    The add_* functions will have magic configuration. They will sort through almost anything and find what they're looking for.
    Otherwise, they'll throw an error.
    """

    def add_model(self, model):
        """_summary_
        Adds a model to the agent. A model must be attached before adding a sensor that uses it
        """

        # Actually adds and CONFIGURES the model to the agent
        pass

    def add_sensor(self, sensor, *args, **kwargs):
        """_summary_
        Adds sensors to the model. You can add models to the agent.
        Args:
            sensor (_type_): _description_
        """

        # create a matcher for a type where the function will search for settings it can set to set it up properly
        match kwargs:
            case "thing1 1":
                pass
            case "thing 2":
                pass
            case _:
                pass

        self.sensor_list.append(sensor)
