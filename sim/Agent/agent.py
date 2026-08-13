import warnings
from sim.environment.ThermalObject import ThermalBody
from sim.rendering.renderable_object import RenderableObject


class Agent(ThermalBody, RenderableObject):
    """Rendered autonomous platform with sensors and a managed thermal body."""

    def __init__(self, thermal_manager=None):
        super().__init__(thermal_manager=thermal_manager)

        # Position and orientation are inherited from RenderableObject.
        # Physics will be implemented by PyBullet.
        self._agent_id = None
        self._name = ""
        self._tf = {}
        self._team = []
        self._sensor_list = []
        self._model_list = []

    @property
    def id(self):
        """Getter for the agent id"""
        return self._id

    @id.setter
    def id(self, value):
        """Setter for an agent's id"""
        try:
            value = int(value)
        except ValueError:
            warnings.warn("ID cannot be converted to integer!")
        self._id = value

    @property
    def name(self):
        """Getter for the agent's name. Usually a string, but doesn't have to be"""
        return self._name

    @name.setter
    def name(self, value) -> None:
        """Getter for agent's name"""
        self._name = str(value)

    @property
    def tf(self):
        """Returns the time set for physics"""
        return self._tf

    @tf.setter
    def tf(self, value):
        """Setter for the tf for physics calculations"""
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                f"Value {value} cannot be converted to float as type {type(value)} and cannot be accepted."
            )
            
        self._if = value

    @property
    def team(self):
        """Getter that returns the team that the object is on"""
        return self._team

    @property
    def sensor_list(self):
        """Getter for the agent's sensor list, which contains the sensors attached to the agent"""
        return self._sensor_list

    @property
    def model_list(self):
        """Getter for the agent's model list"""
        return self._model_list

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
