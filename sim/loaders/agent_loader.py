"""Build configured agents, their models, sensors, and thermal bodies."""

from sim.agent.agent import Agent
from sim.utils.functions import extract_yaml_configurations, set_attr_from_configuration


class AgentLoader:
    """Instantiate every agent declared by a scene configuration."""

    def __init__(self, config, world):
        self.world = world
        self.config = config

    def load_agents(self):
        """Build agents in declaration order, then refresh camera targets."""
        for scene_config in self.config.get("agents", {}).values():
            agent_config_file = extract_yaml_configurations(scene_config["config_path"])
            new_agent = Agent(thermal_manager=self.world.thermal_model)

            set_attr_from_configuration(new_agent, agent_config_file)
            set_attr_from_configuration(new_agent, scene_config)
            new_agent.attach_thermal(
                body_id=new_agent.agent_id,
                source=getattr(new_agent, "model", None) or new_agent.name,
            )
            self._render_agent(new_agent)
            self._load_sensors(new_agent, agent_config_file.get("sensors", {}))
            self.world.agent_list.append(new_agent)

        # Cameras may be constructed while the first agent is still loading.
        # Refresh after the complete agent list exists so every IR camera also
        # receives subsequently loaded drones as live thermal renderables.
        for camera in self.world.camera_list[1:]:
            refresh = getattr(camera, "refresh_scene_thermal_nodes", None)
            if refresh is not None:
                refresh(self.world)

    def _render_agent(self, agent):
        builder = self.world.simulation_manager.renderable_builder
        builder.with_object(agent).config_from_object(agent).build()

    def _load_sensors(self, agent, sensor_configs):
        manager = self.world.simulation_manager
        for registry_name, reference in sensor_configs.items():
            sensor = self.world.sensor_loader.create_sensor(
                reference["type"],
                reference,
                extract_yaml_configurations(reference["config_path"]),
            )
            manager.renderable_builder.with_object(sensor).config_from_object(sensor).build()
            agent.add_sensor(sensor, registry_name)
            self.world.sensor_loader.setup_sensor(sensor)
