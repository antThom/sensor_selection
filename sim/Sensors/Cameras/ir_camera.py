# camera.py
import numpy as np
import pybullet as p
from sim.Sensors.sensor import Sensor  # import the CLASS, not the module
from scipy.spatial.transform import Rotation as Rot
from sim.Environment.Thermal.thermal_manager import ThermalManager
import cv2
import time


class IRCamera(Sensor):
    def __init__(self, thermal_mgr: ThermalManager):
        super().__init__()  # keep config in base
        self.fov = 64
        self.WIDTH = 640
        self.HEIGHT = 640  # was WIDTH before
        self.fx = 3.0e-2
        self.fy = 3.0e-2  # y, not x
        self.c = [320, 320]
        self.forward = [0, 0, 1]
        self.model = "pinhole"
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0
        self.near = 0.1
        self.far = 100.0
        self.input = None
        self.encode = "ir"
        self.temp_min = 200
        self.temp_max = 350
        self.netd_K = 0.05
        self.k_atm = 0.05
        self.up = [0, 1, 0]
        self.aspect = self._WIDTH / self._HEIGHT
        self.thermal_mgr = thermal_mgr
        self.tf = {}

    def get_output(self):
        """Render a frame given the camera position and target."""
        if self.agent is None:
            raise RuntimeError("Camera must be attached to an agent before use.")

        # --- Agent pose in world ---
        pos_agent = self.agent.position.flatten()
        if len(self.agent.orientation.flatten().tolist()) > 3:
            # This is a quaternion
            quat_agent = self.agent.orientation.flatten().tolist()
        else:
            quat_agent = p.getQuaternionFromEuler(
                self.agent.orientation.flatten().tolist()
            )
        R_agent = Rot.from_quat(
            [quat_agent[0], quat_agent[1], quat_agent[2], quat_agent[3]]
        )

        # --- Mount transform (body -> sensor) ---
        mount = self.agent.tf.get("body2Sensor", None)
        if mount:
            R_body2sensor, t_body2sensor = mount  # R is a Rotation object, t is (3,1)
            t_body2sensor = np.array(t_body2sensor)
        else:
            R_body2sensor = Rot.identity()
            t_body2sensor = np.zeros((3, 1))

        # --- Sensor pose in world ---
        R_world2sensor = R_agent * R_body2sensor
        pos_sensor = pos_agent + R_agent.apply(t_body2sensor.flatten())

        # --- Compute view direction ---
        forward_world = -R_world2sensor.apply(
            [0, 0, 1]
        )  # camera looks along -Z in PyBullet
        target_world = pos_sensor + forward_world

        # print("Camera pos:", pos_sensor)
        # print("Camera target:", target_world)
        # print(f"{self.name}: pos={pos_sensor}, forward={forward_world}, target={target_world}")

        view_matrix = p.computeViewMatrix(
            pos_sensor.tolist(), target_world.tolist(), self.up
        )
        p.addUserDebugText(
            "IR_CAM", pos_sensor.tolist(), textColorRGB=[1, 0, 0], lifeTime=0.1
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._fov, aspect=self.aspect, nearVal=self.near, farVal=self.far
        )

        p.addUserDebugLine(
            pos_sensor.tolist(), target_world.tolist(), [1, 0, 0], 2, 0.1
        )
        _, _, _, _, seg = p.getCameraImage(
            self._WIDTH,
            self._HEIGHT,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        seg = np.asarray(seg).reshape(self._HEIGHT, self._WIDTH)
        body_ids = seg & ((1 << 24) - 1)
        link_index = (seg >> 24) - 1

        # Query the temperature for each visible body
        temp_map = np.zeros_like(seg, dtype=np.float32)
        for bid in np.unique(body_ids):
            mask = body_ids == bid
            T = self.thermal_mgr.get_temperature(bid)
            temp_map[mask] = T

        # Add sensor noise & convert to displayable image
        temp_map += np.random.normal(0, self.netd_K, temp_map.shape)
        img8 = (
            ((temp_map - self.temp_min) / (self.temp_max - self.temp_min) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        img_color = cv2.applyColorMap(img8, cv2.COLORMAP_BONE)
        return img_color

    def get_output():
        """To Implement by other classes"""

        raise NotImplementedError
