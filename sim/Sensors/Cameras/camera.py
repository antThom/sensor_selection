"""Shared optical and Panda3D state for scene cameras."""

from sim.sensors.sensor import Sensor, SensorType

# import cv2


class Camera(Sensor):
    """Base pinhole camera with configurable lens and mount transforms."""

    def __init__(self):
        super().__init__()

        self.fov = 64
        self.WIDTH = 640
        self.HEIGHT = 640  # was WIDTH before
        self.camera_model = "pinhole"
        self.near = 0.1
        self.far = 2000.0
        self.focal_length = None

        # Compatibility mount names introduced by upstream PR #29. IR camera
        # configurations use mount_position/mount_hpr instead.
        self.camera_offset = [0, 0, 0]
        self.camera_angle = [0, 0, 0]
        # Physics
        self.tf = {}

        # Implementation
        self.input = None
        self.output = "image"
        self.encode = None
        self.forward = [0, 0, 1]
        self.up = [0, 1, 0]
        self.aspect = self.WIDTH / self.HEIGHT

        # Rendering
        self.camera_node = None
        self.camera_nodepath = None
        self.display_region = None

        self.type = SensorType.CAMERA
