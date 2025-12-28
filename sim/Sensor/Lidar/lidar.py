import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as Rot
from sim.Sensor.sensor import Sensor

class Lidar(Sensor):
    def __init__(self, param: dict, name: str):
        super().__init__(param)  # keep config in base
        self.name    = name
        self.extract_specs(param=param)
        # self.num_channels = int(param.get("resolution_vert", 1))
        self.num_rays  = int(param.get("num_rays", 360))
        self.yaw_deg   = float(param.get("yaw_deg", 0.0))       # mounting yaw offset
        self.input    = param.get("input", None)
        self.output   = param.get("output","point_cloud")
        self.encode   = param.get("encode","point")
        self.bundle_count = int(param.get("bundle_count",5))
        self.max_batch_rays = 8192
        self.tf       = {}

        # Debug
        self.debug_draw = bool(param.get("debug_draw", True))

    def extract_specs(self,param):
        if param.get('specs',None) is not None:
            for specs, val in param.get('specs').items():
                spec = specs.lower()
                setattr(self,spec,val)

        # Define Max/Min Range
        if not hasattr(self,'max_range'):
            setattr(self,'max_range',200.0)
        if not hasattr(self,'min_range'):
            setattr(self,'min_range',0.50)

        # Define Range Noise
        if not hasattr(self,'range_noise_std'):
            setattr(self,'range_noise_std',1E-3)

    def _get_sensor_pose_world(self):
        if self.agent is None:
            raise RuntimeError("Lidar must be attached to an agent before use.")

        pos_agent = self.agent.position.flatten()

        # Agent orientation may be quaternion already in your code path
        if len(self.agent.orientation.flatten().tolist()) > 3:
            quat_agent = self.agent.orientation.flatten().tolist()
        else:
            quat_agent = p.getQuaternionFromEuler(self.agent.orientation.flatten().tolist())

        R_agent = Rot.from_quat([quat_agent[0], quat_agent[1], quat_agent[2], quat_agent[3]])

        # Mount transform (body -> sensor) matches your Camera approach
        mount = self.agent.tf.get("body2Sensor", None)
        if mount:
            R_body2sensor, t_body2sensor = mount
            t_body2sensor = np.array(t_body2sensor).reshape(3,)
        else:
            R_body2sensor = Rot.identity()
            t_body2sensor = np.zeros(3,)

        R_world2sensor = R_agent * R_body2sensor
        pos_sensor = pos_agent + R_agent.apply(t_body2sensor)

        return pos_sensor, R_world2sensor
    
    
    def get_output(self):
        pos_sensor, R_world2sensor = self._get_sensor_pose_world()

        # -------- Parameters (use your class fields if they exist) --------
        fov_x_deg = float(self.fov_x)                 # horizontal FOV (deg)
        fov_y_deg = float(self.fov_y)  # vertical FOV (deg); 0 => planar

        n_az = int(self.num_rays)                     # horizontal samples
        n_el = int(self.resolution_vert)  # vertical channels
        # print(f"n_el: {n_el}")
        yaw0 = np.deg2rad(self.yaw_deg)

        # Ray bundling (tune these)
        # bundle_count = int(getattr(self, "bundle_count", 5))  # 1, 3, 5, 9, ...
        az_jit = np.deg2rad(float(getattr(self, "bundle_az_jitter_deg", 0.10)))  # degrees
        el_jit = np.deg2rad(float(getattr(self, "bundle_el_jitter_deg", 0.10)))  # degrees

        # -------- Build azimuth/elevation grids --------
        half_az = 0.5 * np.deg2rad(fov_x_deg)
        az = np.linspace(-half_az, half_az, n_az, endpoint=False) + yaw0

        if n_el <= 1 or fov_y_deg <= 0.0:
            el = np.array([0.0], dtype=np.float64)
            n_el = 1
        else:
            half_el = 0.5 * np.deg2rad(fov_y_deg)
            el = np.linspace(-half_el, half_el, n_el, endpoint=True)

        # Mesh (n_el, n_az)
        AZ, EL = np.meshgrid(az, el, indexing="xy")

        # -------- Define bundle offsets in (az, el) --------
        # bundle_count=5 => center + 4-neighborhood; 9 => add diagonals
        if self.bundle_count <= 1:
            d_az = np.array([0.0], dtype=np.float64)
            d_el = np.array([0.0], dtype=np.float64)
        elif self.bundle_count <= 5:
            d_az = np.array([0.0, +az_jit, -az_jit, 0.0, 0.0], dtype=np.float64)
            d_el = np.array([0.0, 0.0, 0.0, +el_jit, -el_jit], dtype=np.float64)
        else:
            # 9-ray bundle: center + 8 neighbors
            d_az = np.array([0.0, +az_jit, -az_jit, 0.0, 0.0, +az_jit, +az_jit, -az_jit, -az_jit], dtype=np.float64)
            d_el = np.array([0.0, 0.0, 0.0, +el_jit, -el_jit, +el_jit, -el_jit, +el_jit, -el_jit], dtype=np.float64)

        B = d_az.shape[0]  # actual bundle size used
        # print(f"B: {B}")
        # print(f"AZ: {AZ.shape}")
        # print(f"EL: {EL.shape}")
        # -------- Expand to bundled angles --------
        # AZb/ELb shapes: (B, n_el, n_az)
        AZb = AZ[None, :, :] + d_az[:, None, None]
        ELb = EL[None, :, :] + d_el[:, None, None]
        # print(f"Azb: {AZb}")
        # print(f"ELb: {ELb}")
        # -------- Directions in SENSOR frame (3D) --------
        # x = cos(el)*cos(az), y = cos(el)*sin(az), z = sin(el)
        cosEL = np.cos(ELb)
        dirs_sensor = np.stack(
            [cosEL * np.cos(AZb), cosEL * np.sin(AZb), np.sin(ELb)],
            axis=-1
        )  # (B, n_el, n_az, 3)
        
        # Flatten all rays for rayTestBatch
        dirs_sensor_flat = dirs_sensor.reshape(-1, 3)

        # Convert to WORLD frame
        dirs_world_flat = R_world2sensor.apply(dirs_sensor_flat)
        # print("dirs_world_flat")
        # Rays start at sensor origin (you can also add tiny origin offsets if desired)
        N_total = dirs_world_flat.shape[0]
        ray_from = np.repeat(pos_sensor.reshape(1, 3), N_total, axis=0)
        ray_to = ray_from + dirs_world_flat * self.max_range
        results = []

        for start in range(0, N_total, self.max_batch_rays):
            end = min(start + self.max_batch_rays, N_total)
            results.extend(
                p.rayTestBatch(ray_from[start:end].tolist(), ray_to[start:end].tolist())
            )
        # -------- Parse hits --------
        # We’ll keep the *closest valid* hit across the bundle for each (el, az) beam.
        ranges_best = np.full((n_el, n_az), np.inf, dtype=np.float32)
        points_best = np.full((n_el, n_az, 3), np.nan, dtype=np.float32)

        # results is length N_total; map linear index -> (b, e, a)
        # linear i corresponds to dirs_sensor_flat[i] where i = (((b*n_el)+e)*n_az + a)
        for i, r in enumerate(results):
            hit_id = r[0]
            if hit_id == -1:
                continue

            # --- FILTER: only accept LiDAR proxy bodies ---
            # print("before env")
            env = getattr(self.agent, "environment", None)
            # print(f"env: {env}")
            if env is not None:
                if hit_id not in env.lidar_proxy_ids:
                    continue

            hit_frac = r[2]
            dist = float(hit_frac) * self.max_range
            if dist < self.min_range:
                continue

            # decode indices
            a = i % n_az
            tmp = i // n_az
            e = tmp % n_el
            # b = tmp // n_el   # not needed except for debugging
            # keep closest across bundle
            if dist < ranges_best[e, a]:
                ranges_best[e, a] = dist
                points_best[e, a, :] = np.array(r[3], dtype=np.float32)
                # print("hit!")
        # If you want to keep your old return type when n_el==1:
        if n_el == 1:
            ranges_out = ranges_best.reshape(n_az,)
            points_out = points_best.reshape(n_az, 3)
        else:
            ranges_out = ranges_best
            points_out = points_best

        # print(f"P_out: {points_out}")

        # -------- Debug draw (optional) --------
        if self.debug_draw:
            # draw a subset so GUI doesn't crawl
            # for 3D we sample a few azimuths and a few elevations
            step_az = max(1, n_az // 90)
            step_el = max(1, n_el // 8)

            # For debug lines, use the *center* (non-bundled) directions for readability
            AZc, ELc = AZ, EL
            cosELc = np.cos(ELc)
            dirs_sensor_c = np.stack(
                [cosELc * np.cos(AZc), cosELc * np.sin(AZc), np.sin(ELc)],
                axis=-1
            ).reshape(-1, 3)
            dirs_world_c = R_world2sensor.apply(dirs_sensor_c).reshape(n_el, n_az, 3)

            for e in range(0, n_el, step_el):
                for a in range(0, n_az, step_az):
                    d = dirs_world_c[e, a]
                    rr = ranges_best[e, a]
                    end = (pos_sensor + d * (rr if np.isfinite(rr) else self.max_range)).tolist()
                    p.addUserDebugLine(pos_sensor.tolist(), end, [0, 1, 0], 1, 0.05)

        return ranges_out, points_out
