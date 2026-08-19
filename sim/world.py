from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletWorld
from panda3d.core import Vec3

from sim.agent.camera_controls import CameraControls
from sim.agent.drone_controls import DroneControls
from sim.environment.thermal.thermal_manager import ThermalManager
from sim.loaders.agent_loader import AgentLoader
from sim.loaders.environment_loader import EnvironmentLoader
from sim.loaders.object_loader import ObjectLoader
from sim.loaders.sensor_loader import SensorLoader
from sim.rendering.simulation_manager import SimulationManager
from sim.utils.functions import extract_yaml_configurations


class WORLD(ShowBase):
    """Top-level owner of scene, physics, thermal, agent, and camera systems."""

    def __init__(self, config_file):

        super().__init__()

        yaml_config = extract_yaml_configurations(config_file)

        self.agent_list = list()
        self.camera_list = [base.camera]

        thermal_config = yaml_config.get("thermal", {})
        atmosphere_time = (
            yaml_config.get("atmosphere", {}).get("time", {}).get("time", 12)
        )
        self.thermal_model = ThermalManager(
            time_of_day=thermal_config.get("time_of_day", atmosphere_time),
            ambient_K=thermal_config.get(
                "ambient_temp", yaml_config.get("ambient_temp", 293.0)
            ),
            T_sky=thermal_config.get("sky_temp", yaml_config.get("sky_temp", 260.0)),
        )

        # Load physics simulation with pybullet
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        # DO NOT CHANGE LOADER ORDER, THEY DEPEND ON EACH OTHER
        # environment -> objects -> agents

        self.simulation_manager = SimulationManager(
            self
        )  # Pass self to attach loaders to world class

        self.environment_loader = EnvironmentLoader(yaml_config, self)
        self.environment_loader.load_environment()

        # World class gets to own all of the objects
        self.terrain = self.environment_loader.terrain
        self.sky = self.environment_loader.sky

        self.object_loader = ObjectLoader(self)
        self.object_loader.load_objects(yaml_config=yaml_config, object_type="static")

        self.sensor_loader = SensorLoader(self)

        self.agent_loader = AgentLoader(yaml_config, self)
        self.agent_loader.load_agents()

        # Cameras
        self.camera_controls = CameraControls(self)
        self.drone_controls = DroneControls(self)

        # keybind corner
        self.accept("c", self.camera_controls.camera_list_forward)
        self.accept("x", self.camera_controls.camera_list_back)
        self.accept("z", self.camera_controls.save_current_camera_image)
