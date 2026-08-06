"""Construction and Panda3D setup for agent-mounted sensors."""

from panda3d.core import Camera as PandaCamera
from panda3d.core import PerspectiveLens

from sim.sensors.sensor import Sensor, SensorType


class SensorLoader:
    """Create configured sensors and connect their render resources."""

    def __init__(self, world):
        self.world = world

    def create_sensor(self, sensor_type, *configurations):
        """Instantiate and configure one supported sensor type."""
        normalized = str(sensor_type).strip().lower().replace(" ", "_")

        if normalized in {"ir_camera", "ircamera"}:
            from sim.sensors.cameras.ir_camera import IRCamera

            sensor = IRCamera(thermal_manager=self.world.thermal_model)
            expected_type = SensorType.IRCAMERA
        elif normalized in {"eo_camera", "rgb_camera", "rgbcamera"}:
            from sim.sensors.cameras.eo_camera import EOCamera

            sensor = EOCamera()
            expected_type = SensorType.EOCAMERA
        elif normalized == "dummy":
            from sim.sensors.dummy_sensor import DummySensor

            sensor = DummySensor()
            expected_type = SensorType.DUMMY
        else:
            raise ValueError(f"unsupported sensor type: {sensor_type!r}")

        sensor.set_configs(configurations)
        # YAML contains a string type field; retain the internal enum after
        # applying all configuration sources.
        sensor.type = expected_type
        if expected_type is SensorType.IRCAMERA:
            sensor.validate_parameters()
        return sensor

    def setup_sensor(self, sensor: Sensor):
        """Allocate Panda3D resources required by a configured sensor."""
        if sensor.type in {SensorType.EOCAMERA, SensorType.IRCAMERA}:
            self.setup_camera(sensor)

    def setup_camera(self, sensor: Sensor):
        """Create a selectable scene camera mounted to the owning agent."""
        sensor.camera_node = PandaCamera(f"{sensor.name}_camera")

        if sensor.agent is not None and sensor.agent.object_node_path is not None:
            camera_parent = sensor.agent.object_node_path
        elif sensor.object_node_path is not None:
            camera_parent = sensor.object_node_path
        else:
            camera_parent = self.world.render

        sensor.camera_nodepath = camera_parent.attachNewNode(sensor.camera_node)
        mount_position = list(
            getattr(
                sensor,
                "mount_position",
                getattr(sensor, "camera_offset", [0.0, 0.0, 0.0]),
            )
        )
        if (
            getattr(sensor, "mount_mode", "absolute") == "model_bounds"
            and sensor.agent is not None
            and sensor.agent.model_node is not None
        ):
            lower, upper = sensor.agent.model_node.getTightBounds()
            mount_position = [
                lower[axis] + float(mount_position[axis]) * (upper[axis] - lower[axis])
                for axis in range(3)
            ]
        sensor.camera_nodepath.setPos(*mount_position)
        mount_hpr = getattr(
            sensor,
            "mount_hpr",
            getattr(sensor, "camera_angle", [0.0, 0.0, 0.0]),
        )
        sensor.camera_nodepath.setHpr(*mount_hpr)

        sensor.display_region = self.world.win.makeDisplayRegion()
        sensor.display_region.setCamera(sensor.camera_nodepath)
        sensor.display_region.setActive(False)
        sensor.display_region.setSort(5)
        sensor.display_region.setClearDepthActive(True)
        sensor.display_region.setClearColorActive(True)
        sensor.display_region.setClearColor((0.02, 0.02, 0.02, 1.0))

        lens = PerspectiveLens()
        horizontal_fov = float(getattr(sensor, "horizontal_fov_deg", sensor.fov))
        vertical_fov = float(getattr(sensor, "vertical_fov_deg", horizontal_fov))
        lens.setFov(horizontal_fov, vertical_fov)
        lens.setNearFar(float(sensor.near), float(sensor.far))

        focal_length = getattr(sensor, "focal_length", None)
        if focal_length is not None:
            lens.setFocalLength(float(focal_length))

        sensor.camera_node.setLens(lens)
        if sensor.type is SensorType.IRCAMERA:
            sensor.setup_live_thermal_view(self.world)
        self.world.camera_list.append(sensor)
        return sensor

    # Compatibility alias for existing callers.
    setupEOCamera = setup_camera

    def register_sensor(self, sensor):
        return sensor

    def attach_sensor_to_model(self, sensor, model):
        sensor.attach_to_agent(model)
        return sensor

    @staticmethod
    def update_sensors(sensor_list):
        for sensor in sensor_list:
            update = getattr(sensor, "update", None)
            if update is not None:
                update()
