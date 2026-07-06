from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletWorld
from panda3d.core import Vec3
import yaml

from sim.loaders.agent_loader import AgentLoader
from sim.loaders.environment_loader import EnvironmentLoader
from sim.loaders.object_loader import ObjectLoader


class WORLD(ShowBase):
    """
    Overall manager for the simulation. Derives from Panda3D's `ShowBase` class.
    Calling point for objects of the simulation.
    """

    def __init__(self, config_file):

        ShowBase.__init__(self)

        with open(config_file, "r") as file:
            yaml_config = yaml.safe_load(file)

        # Load physics simulation with pybullet
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        # DO NOT CHANGE LOADER ORDER, THEY DEPEND ON EACH OTHER
        # environment -> objects -> agents

        # Pass self to attach loaders to world class
        self.environment_loader = EnvironmentLoader(yaml_config, self)
        self.environment_loader.load_environment()

        # World class gets to own all of the objects
        self.terrain = self.environment_loader.terrain
        self.sky = self.environment_loader.sky

        self.object_loader = ObjectLoader(self)
        self.object_loader.load_objects(yaml_config=yaml_config, object_type="static")

        self.agent_loader = AgentLoader(yaml_config, self)
        self.agent_loader.load_agents()
