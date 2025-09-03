# camera.py
import numpy as np
from sim.Sensor.sensor import Sensor  # import the CLASS, not the module

class Camera(Sensor):
    def __init__(self, param: dict):
        super().__init__(param)  # keep config in base
        self._fov    = param.get("fov", 60)
        self._WIDTH  = param.get("WIDTH", 640)
        self._HEIGHT = param.get("HEIGHT", 640)          # was WIDTH before
        self._fx     = param.get("x_focal_length", 3.0e-2)
        self._fy     = param.get("y_focal_length", 3.0e-2)  # y, not x
        self._c      = param.get("center", [320, 320])
        self._forward = param.get("forward", [0,0,1])
        self._model   = param.get("model","pinhole")
        self._k1      = param.get("k1", 0.0)
        self._k2      = param.get("k2", 0.0)
        self._k3      = param.get("k3", 0.0)
        self._k4      = param.get("k4", 0.0)
        self.input    = param.get("input", None)
        self.output   = param.get("output","image")

    def get_output(self):
        # implement whatever your pipeline needs
        return 1
