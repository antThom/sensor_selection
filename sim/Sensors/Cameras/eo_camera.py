"""The file for the EO camera, otherwise known as an RBG camera."""

from sim.sensors.cameras.camera import Camera
from sim.sensors.sensor import SensorType


class EOCamera(Camera):
    """_summary_
    The EO Camera class, otherwise known as an RBG camera.

    Args:
        Camera (_type_): _description_
    """

    def __init__(self):
        super().__init__()

        # Set the default values.
        self.fov = 64
        self.WIDTH = 640
        self.HEIGHT = 640  # was WIDTH before
        self.fx = 3.0e-2
        self.fy = 3.0e-2  # y, not x
        self.c = [320, 320]
        self.forward = [0, 0, 1]
        self.camera_model = "pinhole"
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0
        self.near = 0.1
        self.far = 100.0
        self.input = None
        self.output = "image"
        self.encode = "rgb"
        self.up = [0, 1, 0]
        self.aspect = self.WIDTH / self.HEIGHT
        self.tf = {}

        self.type = SensorType.EOCAMERA
