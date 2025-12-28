# camera.py
import numpy as np
import pybullet as p
from sim.Sensor.sensor import Sensor  # import the CLASS, not the module
from scipy.spatial.transform import Rotation as Rot
import cv2
import time

class DepthCamera(Sensor):
    def __init__(self, param: dict, name: str):
        super().__init__(param)  # keep config in base
        self.name    = name
        self.extract_specs(param=param)
        self.input    = param.get("input", None)
        self.output   = param.get("output","image")
        self.encode   = param.get("encode","rgb")
        self.tf       = {}

    def extract_specs(self,param):
        if param.get('specs',None) is not None:
            for specs, val in param.get('specs').items():
                spec = specs.lower()
                setattr(self,spec,val)
            
            # Define Aspect
            if hasattr(self,"width") and hasattr(self,"height") and not hasattr(self,"aspect"):
                setattr(self,'aspect',self.width/self.height)
            else:
                setattr(self,'width',640)  # Default value
                setattr(self,'height',640) # Default value
                setattr(self,'aspect',self.width/self.height)

            # Define fov
            if not hasattr(self,"fov_x") and hasattr(self,"x_focal_length"):
                setattr(self,'fov_x',2*np.atan2(self.width,2*self.x_focal_length))
            elif not hasattr(self,"fov_x") and not hasattr(self,"x_focal_length"):
                setattr(self,'fov_x',60) # Default value
            if not hasattr(self,"fov_y") and hasattr(self,"y_focal_length"):
                setattr(self,'fov_y',2*np.atan2(self.height,2*self.y_focal_length))
            elif not hasattr(self,"fov_y") and not hasattr(self,"y_focal_length"):
                setattr(self,'fov_y',60) # Default value

            # Define Near and Far Planes
            if not hasattr(self,'near'):
                setattr(self,'near', 0.1)
            if not hasattr(self,'far'):
                setattr(self,'far', 100.0)

            # Define Up
            self.up = [0,1,0]

    def get_output(self):
        """Render a frame given the camera position and target."""
        if self.agent is None:
            raise RuntimeError("Camera must be attached to an agent before use.")
        
        # --- Agent pose in world ---
        pos_agent = self.agent.position.flatten()
        if len(self.agent.orientation.flatten().tolist())>3:
            # This is a quaternion
            quat_agent = self.agent.orientation.flatten().tolist()
        else:
            quat_agent = p.getQuaternionFromEuler(self.agent.orientation.flatten().tolist())
        R_agent = Rot.from_quat([quat_agent[0], quat_agent[1], quat_agent[2], quat_agent[3]])

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
        forward_world = -R_world2sensor.apply([0, 0, 1])  # camera looks along -Z in PyBullet
        target_world = pos_sensor + forward_world

        # print("Camera pos:", pos_sensor)
        # print("Camera target:", target_world)
        # print(f"{self.name}: pos={pos_sensor}, forward={forward_world}, target={target_world}")

        view_matrix = p.computeViewMatrix(
            pos_sensor.tolist(),
            target_world.tolist(),
            self.up
        )
        p.addUserDebugText("CAM", pos_sensor.tolist(), textColorRGB=[1,0,0], lifeTime=0.1)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov_y,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )

        p.addUserDebugLine(pos_sensor.tolist(), target_world.tolist(), [1, 0, 0], 2, 0.1)
        images = p.getCameraImage(
            self.width, self.height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb = np.reshape(images[2], (self.height, self.width, 4))[:,:,:3]
        depth_buffer_opengl = np.reshape(images[3], [self.width, self.height])
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_opengl)

        
        return rgb, depth
