"""_summary_
This loads camera and agent controls (agents haven't been implemented yet)
"""

from math import pi, sin, cos

from sim.agent.agent import Agent
from sim.utils.functions import extract_yaml_configurations, set_attr_from_configuration

from sim.environment.ThermalObject.ThermalObject import ThermalObject


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

        for agent_config in self.config["agents"]:

            agent_config_file = extract_yaml_configurations(
                self.config["agents"][agent_config]["config_path"]
            )
            new_agent = Agent()

            # Set configurations twice: once for the template file and second for the top level config
            set_attr_from_configuration(new_agent, agent_config_file)
            set_attr_from_configuration(
                new_agent, self.config["agents"][agent_config]
            )

            # load math models somehow somewhere???
            # Find a way to attach agents to the model so that it can update them

            # If we want to add animations in the future, switch to actor and implement all the animations
            # Animations aren't worth it right now
            sim_man = self.world.simulation_manager

            sim_man.generate_simulation_node(new_agent, new_agent.model)
            sim_man.configure_sim_model(new_agent)
            sim_man.render_object(new_agent)

            # sim_man.attach_sound(new_agent, agent_config_file) # Later Problem

            # load sensors (sensor loader will deal with it)
            for sensor_config in agent_config_file["sensors"]:
                new_sensor = self.world.sensor_loader.create_sensor(
                    sensor_config, 
                    agent_config_file["sensors"][sensor_config]["type"], 
                    agent_config_file["sensors"][sensor_config],
                    self.world)


                sim_man.generate_simulation_node(new_sensor, agent_config_file["sensors"][sensor_config]["model"])
                sim_man.configure_sim_model(new_sensor)
                sim_man.parent_object_models(new_agent, new_sensor)

                self.world.sensor_loader.setup_sensor(new_sensor)
                sim_man.render_object(new_sensor)
                new_agent.add_sensor(new_sensor, sensor_config)
            # for model in agent_config_file["models"]:
            #     new_agent.add_model(model)

            # attach sensor 3d models to the agent
            # Agent is now fully built & set up
            self.world.agent_list.append(new_agent)



    def generate_agent_models(self):
        """For panda3d"""
        pass

    def load_agent(self):
        pass

    # # Define a procedure to move the camera.
    # def spinCameraTask(self, task):
    #     """From Panda3D's quck start tutorial"""
    #     angleDegrees = task.time * 6.0
    #     angleRadians = angleDegrees * (pi / 180.0)
    #     self.world.camera.setPos(350 * sin(angleRadians), -350 * cos(angleRadians), 150)
    #     self.world.camera.setHpr(angleDegrees, -15, 0)

    #     self.world.sky.sky._update_sun(task.time)
    #     return Task.cont
