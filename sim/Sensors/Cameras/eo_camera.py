""" The file for the EO camera, otherwise known as an RBG camera. """

from sim.sensors.cameras.camera import Camera

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
        self.model = "pinhole"
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 =  0.0
        self.k4 = 0.0
        self.near = 0.1
        self.far = 100.0
        self.input = None
        self.output = "image"
        self.encode = "rgb"
        self.up = [0, 1, 0]
        self.aspect = self.WIDTH / self.HEIGHT
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
            "CAM", pos_sensor.tolist(), textColorRGB=[1, 0, 0], lifeTime=0.1
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._fov, aspect=self.aspect, nearVal=self.near, farVal=self.far
        )

        p.addUserDebugLine(
            pos_sensor.tolist(), target_world.tolist(), [1, 0, 0], 2, 0.1
        )
        _, _, rgb, _, _ = p.getCameraImage(
            self._WIDTH,
            self._HEIGHT,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        img = np.reshape(rgb, (self._HEIGHT, self._WIDTH, 4))[:, :, :3]
        # timestamp = time.time()
        return img
