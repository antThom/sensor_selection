"""File for holding the SensorLoader class and its utilities"""

from sim.sensors.sensor import SensorType, Sensor
from panda3d.core import PandaNode, NodePath


class SensorLoader:
    """_summary_

    Loader for creating and intializing sensors based on configurations.
    Mostly works as an object generator for AgentLoader
    """

    def __init__(self, world):
        self.world = world

    def create_sensor(self, type, *args, **kwargs):
        """_summary_
        Given a set of configurations, will generate a sensor object. All extra args will be applied first to last.
        Returns:
            _type_: _description_
        """
        sensor = None

        # See design pattern "Strategy"
        # Ensure you set the type so internals are able to react to ernumerators
        match type:
            case "IR Camera":
                pass
            case "RGB Camera":
                pass
            case "RBGS Camera":
                pass
            case "Microphone":
                pass
            case "dummy":
                from sim.sensors.dummy_sensor import DummySensor

                sensor = DummySensor()
                sensor.set_configs(args)
                sensor.type = SensorType.DUMMY
            case "eo_camera":
                from sim.sensors.cameras.eo_camera import EOCamera

                sensor = EOCamera()
                sensor.set_configs(args)
                sensor.type = SensorType.EOCAMERA

            case _:
                pass
                # Nothing matched, raise an error

        return sensor

    def setup_sensor(self, sensor: Sensor):
        match sensor.type:
            case SensorType.EOCAMERA:
                # case "eo_camera":
                self.setupEOCamera(sensor)

    def setupEOCamera(self, sensor: Sensor):
        """_summary_
        Sets up and unboxes all the backend required to create sensors.
        Call on an instianiated camera to initialize it.

        Args:
            sensor (_type_): _description_
            world (_type_): _description_
        """
        from panda3d.core import PerspectiveLens, Camera
        import sim.rendering.simulation_manager
        
        

        sensor.camera_node = Camera(f"{sensor.name}_camera")
        sensor.camera_nodepath = NodePath(sensor.camera_node)
        sensor.camera_nodepath.reparentTo(sensor.object_node_path)
        
        sensor.display_region = self.world.win.makeDisplayRegion()
        sensor.display_region.setCamera(sensor.camera_nodepath)
        sensor.display_region.setActive(False)
        # Always force default display region and camera
        # Disabling all views means that default camera will remain.
        sensor.display_region.setSort(5)

        # Lens Configuration
        lens = PerspectiveLens()

        lens.setFov(sensor.fov)
        lens.setAspectRatio(sensor.aspect)
        lens.setNear(sensor.near)
        lens.setFar(sensor.far)
        lens.setFilmSize(sensor.WIDTH, sensor.HEIGHT)

        if sensor.focal_length is not None:
            lens.setFocalLength(sensor.focal_length)

        sensor.camera_node.setLens(lens)

        self.world.camera_list.append(sensor)

    def register_sensor(self, sensor):
        pass

    def attach_sensor_to_model(self, sensor, model):
        pass

    def update_sensors(sensor_list: list):
        for sensor in sensor_list:
            sensor.update()
