# camera.py
import numpy as np
import pybullet as p
from sim.sensors.sensor import Sensor, SensorType  # import the CLASS, not the module
from scipy.spatial.transform import Rotation as Rot
import cv2
import time


class Camera(Sensor):
    def __init__(self):
        """_summary_

        Args:
            param (dict): _description_
            name (str): _description_
        """
        super().__init__()
        
        self.fov = 64
        self.WIDTH = 640
        self.HEIGHT = 640  # was WIDTH before
      #  self.fx = 3.0e-2
      #  self.fy = 3.0e-2  # y, not x
      #  self.c = [320, 320]
        self.camera_model = "pinhole"
        # self.k1 = 0.0 # Distortion constants
        # self.k2 = 0.0
        # self.k3 =  0.0
        # self.k4 = 0.0
        self.near = 0.1
        self.far = 100.0
        self.focal_length = None
        
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

