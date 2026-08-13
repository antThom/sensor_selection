"""Build configured agents, their models, sensors, and thermal bodies."""

from sim.agent.agent import Agent
from sim.utils.functions import extract_yaml_configurations, set_attr_from_configuration
from sim.utils.builder import BuilderTemplate
import functools
import warnings


class AgentLoader:
    """High level handler to controlling agents, such as their creation and loading."""

    def __init__(self, config, world):
        self.world = world
        self.config = config
        self.agent_builder = AgentBuilder(self.world)

    def load_agents(self):
        """Build agents in declaration order, then refresh camera targets."""
        for scene_config in self.config.get("agents", {}).values():
            agent_config_file = extract_yaml_configurations(scene_config["config_path"])

            self.agent_builder.with_configurations(agent_config_file, priority=10)
            self.agent_builder.with_configurations(scene_config, priority=5)
            self.agent_builder.set_thermal_manager(
                thermal_manager=self.world.thermal_model
            )
            self.agent_builder.attach_thermal()
            self.agent_builder.with_sensor_list(agent_config_file.get("sensors", {}))
            self.agent_builder.build().attach_to_world()

        # Cameras may be constructed while the first agent is still loading.
        # Refresh after the complete agent list exists so every IR camera also
        # receives subsequently loaded drones as live thermal renderables.
        for camera in self.world.camera_list[1:]:
            refresh = getattr(camera, "refresh_scene_thermal_nodes", None)
            if refresh is not None:
                refresh(self.world)


from sim.environment.thermal.thermal_manager import ThermalManager


class AgentBuilder(BuilderTemplate):
    """Builder for Agents"""

    def __init__(self, show_base):
        self.world = show_base
        self._agent: Agent = None

        self._name = "Default Name"
        self._id = 0
        self._teams = None
        self._sensor_configs = dict()
        self._model_list = None

        self._configs_dict = dict()
        self._sensors_configs_dict = dict()
        self._using_config_list = False
        self._rendering_agent = True  # Switch to render the agent

        self._thermal_manager = None
        self._thermal_body_args = list()

    def chainable(self, method):
        """
        Decorator to enable a function to be chained on others when calling the builder.
        """

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            method(self, *args, **kwargs)
            return self

        return wrapper

    @property
    def id(self):
        """Getter for an AgentBuilder's agent id"""
        return self._id

    @id.setter
    @chainable
    def id(self, value):
        """Setter for an AgentBuilder's agent id"""
        try:
            value = int(value)
        except ValueError:
            warnings.warn(f"ID cannot be converted to integer!")
        self._id = value

    @property
    def name(self) -> str:
        """Getter for an Agent Builder's Name"""
        return self._name

    @name.setter
    @chainable
    def name(self, value) -> None:
        """Getter for an AgentBuilder's name"""
        self._name = str(value)

    @property
    def render_agents(self):
        """Getter for whether the builder will automatically render the agent."""
        return self._rendering_agent

    @render_agents.getter
    @chainable
    def render_agents(self, value):
        """
        Switch for rendering the agents in the builder.
        True by default. Turn off if you want to manually render the agent.
        """
        try:
            self._rendering_agent = bool(value)
        except:
            raise ValueError(
                "Value could not be converted into a boolean. Switch must be a bool or similar."
            )

    @chainable
    def with_configurations(self, configuration: dict, priority: int = 0):
        """
        Sets the configurations to give to the object.
        Configurations with higher priority will be set first.
        No configurations should have the same priority.
        """
        self._using_config_list = True
        if not isinstance(configuration, dict):
            raise ValueError("Set configurations must be a dictionary")
        try:
            priority = int(priority)
        except ValueError:
            raise ValueError(
                "Given priority could not be converted into an integer. Priority must be an integer or similar."
            )

        if priority in self._configs_dict.keys():
            warnings.warn(
                f"Another agent setting has been set with the same priority. New setting will be applied with priority {priority} to this object"
            )
        self._configs_dict[priority] = configuration

    @chainable
    def with_sensor_configuration(
        self, name: str, configuration: dict, priority: int = 0
    ):
        """
        Sets the configurations to a given sensor
        """
        if not isinstance(configuration, dict):
            raise ValueError("Set configurations must be a dictionary")
        try:
            priority = int(priority)
        except ValueError:
            raise ValueError(
                "Given priority could not be converted into an integer. Priority must be an integer or similar."
            )

        if priority in self._configs_dict.keys():
            warnings.warn(
                f"Another sensor's setting has been set with the same priority. New setting will be applied with priority {priority} to this object"
            )

        self._sensors_configs_dict[priority] = [name, configuration]

    @chainable
    def set_thermal_manager(self, thermal_manager: ThermalManager):
        """Sets the thermal manager of the agent."""
        if not isinstance(thermal_manager, ThermalManager):
            raise ValueError("Given arguement is not a thermal manager")
        self._thermal_manager = thermal_manager

    @chainable
    def add_sensor(self, sensor_name: str, configs):
        """Adds a sensor with the given lists of configurations"""
        # Need to implement...
        pass

    @chainable
    def with_sensor_list(self, config_dict):
        """
        Sets the entire sensor list. Using `add_sensor` will
        append additional sensors to the agent. This will have the lowest priority
        """
        self._sensor_configs = config_dict
        if not isinstance(configuration, dict):
            raise ValueError("Set configurations must be a dictionary")

    @chainable
    def attach_thermal(self, body_id=None, source=None):
        """Attaches a thermal body to the agent."""
        # validation is done later
        self._thermal_body_args = [body_id, source]

    def _reset(self):
        """Resets the builder"""
        self._name = "Default Name"
        self._id = 0
        self._teams = None
        self._sensor_configs = list()
        self._model_list = None

        self._configs_dict = dict()
        self._sensors_configs_dict = dict()
        self._using_config_list = False
        self._rendering_agent = True  # Switch to render the agent

        self._thermal_manager = None
        self._thermal_body_args = list()

    # def _add_to_config_list(self, using_list, priority:int, name=None):
    #     """Reuseable function that adds a setting to the appropriate list"""

    def _apply_configuration_lists(self):
        """
        Applies the configurations in the agent configuration list
        if the configuration list is being used. Otherwise, resort to default values."""

        # Order the configurations in the correct order
        ordered_keys = sorted(self._configs_dict.keys())  # lowest to highest

        for key in ordered_keys:
            set_attr_from_configuration(self._agent, self._configs_dict[key])

    def _set_agent_attributes(self):
        """Applies attributes set on the builder onto the agent. Overrides configurations"""
        self._agent.name = self.name
        self._agent.id = self.id

    def _attach_thermal(self):
        """Build step of setting up thermal attributes"""
        body_id, source = self._thermal_body_args

        body_id = (
            self._agent.id
            if not body_id
            else warnings.warn(
                f"Thermal Body for {self._agent.name } does not have an ID."
            )
        )
        try:
            source = self._agent.model
        except Exception as exc:
            source = self._agent.name

        if not isinstance(body_id, int):
            raise ValueError("body_id should be an integer")
        if not isinstance(source, str):
            raise ValueError("Source should be a string")

        self._agent.attach_thermal(body_id=body_id, source=source)

    def _apply_teams(self):
        """Build step of applying the selected team to the agent"""
        # Not implemented yet...

    def _load_sensors(self, agent, sensor_configs):
        """Build step of creating and loading the sensors"""
        builder = self.world.simulation_manager.renderable_builder
        for registry_name, reference in sensor_configs.items():
            sensor = self.world.sensor_loader.create_sensor(
                reference["type"],
                reference,
                extract_yaml_configurations(reference["config_path"]),
            )
            builder.with_object(sensor).config_from_object(sensor).build()
            agent.add_sensor(sensor, registry_name)
            self.world.sensor_loader.setup_sensor(sensor)

    def _render_agent(self, agent):
        """Calls the renderable builder"""
        builder = self.world.simulation_manager.renderable_builder
        builder.with_object(agent).config_from_object(agent).build()

    @chainable
    def build(self):
        """
        Builds agent with set configurations.
        If the configuration is not listed, they are built with default settings.
        """
        if not self._thermal_manager:
            raise ValueError("A thermal manager must be set.")
        self._agent = Agent(self._thermal_manager)

        if self._using_config_list:
            self._apply_configuration_lists()
        self._set_agent_attributes()

        if self._rendering_agent:
            self._render_agent(self._agent)
        self._load_sensors(self._agent, self._sensors_configs_dict)
        self._attach_thermal()

    def attach_to_world(self):
        """Adds agent to world.agent_list and resets the builder"""
        self.world.agent_list.append(self._agent)

        product = self._agent
        self._reset()
        return product
