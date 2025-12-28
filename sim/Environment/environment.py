import numpy as np
import math
import os 
import sim.print_helpers as ph
import pybullet as p
import pybullet_data
from pathlib import Path
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from sim.Environment.Thermal.thermal_manager import ThermalManager
from scipy.spatial import ConvexHull
from sim.Constants import *
import trimesh
import time as time_

class Environment:
    def __init__(self, terrain, thermal: ThermalManager, time_of_day=None ):
        self.terrain_config = terrain
        base_dir = os.getcwd()
        config_file = os.path.join(base_dir,os.path.normpath(self.terrain_config["Layered Surface"]["Mesh"]))
        self.thermal = thermal
        self.lidar_proxy_ids = set()

        print(f"Loading the environment from {config_file}")
        # Load the terrain
        print(f"Loading the terrain: {ph.YELLOW}STARTED{ph.RESET}")
        self.load_terrain(config_file, base_dir)
        print(f"Loading the terrain: {ph.GREEN}COMPLETE{ph.RESET}")

        # Load features
        print(f"Loading the terrain features: {ph.YELLOW}STARTED{ph.RESET}")
        self.load_features(config_file, base_dir)
        print(f"Loading the terrain features: {ph.GREEN}COMPLETE{ph.RESET}")
        


        # Initialize Sound Sources List
        self.sound_sources = []

        # Initialize the sun and ambient conditions                   

    def load_terrain(self,config_file, base_dir):
        self.terrain={}
        if "asc" in config_file:
            print(f"{ph.CYAN}Loading asc file{ph.RESET}")
            self.terrain['header'], height_data = self.read_asc_file(config_file)
            # self.header, height_data = self.read_asc_file(config_file)
            # Normalize / scale height
            height_data = height_data.astype(np.float32)
            height_scale = 1.0  # vertical exaggeration if needed

            # 2) Compute true min/max ignoring NODATA
            hmin = np.nanmin(height_data)
            hmax = np.nanmax(height_data)

            height_data = np.nan_to_num(height_data, nan=hmin)
            self.terrain['height_data'] = height_data
            self.terrain['z_offset'] = -hmin * height_scale
            # z_offset = -height_data_min * height_scale

            # Flatten row-major
            self.terrain['terrain_data'] = height_data.flatten()

            terrain_id = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[self.terrain['header']["cellsize"], self.terrain['header']["cellsize"], height_scale],
                heightfieldTextureScaling=self.terrain['header']["ncols"] / 2,
                heightfieldData=self.terrain['terrain_data'],
                numHeightfieldRows=self.terrain['header']["nrows"],
                numHeightfieldColumns=self.terrain['header']["ncols"]
            )

            self.terrain['terrain'] = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=terrain_id,
                basePosition=[0, 0, self.terrain['z_offset']]
            )

            self.terrain['terrain_bounds'] = np.array(([self.terrain['header']['xllcorner'],self.terrain['header']['xllcorner'] + self.terrain['header']['cellsize']*self.terrain['header']['ncols']],
                                            [self.terrain['header']['yllcorner'],self.terrain['header']['yllcorner'] + self.terrain['header']['cellsize']*self.terrain['header']['nrows']]))
            self.terrain['terrain_area'] = np.product(self.terrain['terrain_bounds'][:,-1])
        else:
            print(f"{ph.RED}Loading a different format{ph.RESET}")

        texture_file_name = os.path.join(base_dir,os.path.normpath(self.terrain_config["Layered Surface"]["Texture"]))
        tex_id = p.loadTexture(texture_file_name)
        p.changeVisualShape(self.terrain['terrain'], -1, textureUniqueId=tex_id)
        p.changeVisualShape(self.terrain['terrain'], -1, rgbaColor=[1,1,1,1], specularColor=[0.1,0.1,0.1])

        self.thermal.register_body(terrain_id, config_file, per_link=False)
        self.thermal.register_body(tex_id, texture_file_name, per_link=False)

    def load_features(self, config_file, base_dir):
        for idx, feat in enumerate(self.terrain_config["Surface Mesh"]):
            if idx>0.0:
                mesh_file_name = str(feat["Mesh"]).strip().strip("'\"")  # remove any stray quotes in the JSON

                mesh_file_path = Path(mesh_file_name)
                if mesh_file_path.is_absolute():
                    mesh_path = mesh_file_path
                else:
                    mesh_path = Path(base_dir) / mesh_file_path

                mesh_path = mesh_path.resolve()

                if feat['type'] == "tree":
                    self.load_trees(feat,mesh_path)
                elif feat['type'] == "cloud":
                    self.load_clouds(feat,mesh_path)
    
    def load_trees(self,feat,mesh_path):
        if feat['position_type'] == "patch":
            poly_coords = np.asarray(feat["patch"]).reshape((-1,2))
            patch_polygon = Polygon(poly_coords)
            tmp_tree_id = p.loadURDF(str(mesh_path), useFixedBase=True)
            self._disable_all_collisions(tmp_tree_id)  # keep it from affecting things
            Num_feat = self.determine_N_features(feat=feat, id=tmp_tree_id, area=patch_polygon.area, N_min=5)
            p.removeBody(tmp_tree_id)
            tree_xy_positions = self.sample_points_in_polygon(patch_polygon, Num_feat, buffer=1.0)
        else:
            tree_xy_positions = np.random.uniform(low=-5, high=5, size=(1, 2))

        # Calibrate from URDF/OBJ once per tree asset (cache it)
        if not hasattr(self, "_tree_proxy_cache"):
            self._tree_proxy_cache = {}

        key = str(mesh_path)
        if key not in self._tree_proxy_cache:
            self._tree_proxy_cache[key] = self._calibrate_tree_proxy_from_urdf(mesh_path)

        cal = dict(self._tree_proxy_cache[key])  # copy

        # Allow JSON config to override calibrated values if desired
        for k in ["trunk_radius", "trunk_height", "trunk_center_z", "canopy_radius", "canopy_center_z"]:
            if k in feat:
                cal[k] = float(feat[k])

        trunk_radius   = cal["trunk_radius"]
        trunk_height   = cal["trunk_height"]
        trunk_center_z = cal["trunk_center_z"]
        canopy_radius  = cal["canopy_radius"]
        canopy_center_z= cal["canopy_center_z"]*2


        for x, y, z in tree_xy_positions:
            z_ter = -self.get_height_from_heightmap(x, y, self.terrain['height_data'], sx=10, sy=10, sz=1.0)
            
            base_pos = [float(x), float(y), float(z_ter + z)]

            # 1) Visual tree (URDF) – collisions OFF
            tree_vis_id = p.loadURDF(
                str(mesh_path),
                basePosition=base_pos,
                useFixedBase=True,
                useMaximalCoordinates=True
            )
            self._disable_all_collisions(tree_vis_id)

            # 2) Collision proxy (primitives) – collisions ON
            trunk_id, canopy_id = self._create_tree_collision_proxy(
                base_pos=base_pos,
                trunk_radius=trunk_radius,
                trunk_height=trunk_height,
                trunk_center_z=trunk_center_z,
                canopy_radius=canopy_radius,
                canopy_center_z=canopy_center_z
            )
            self.lidar_proxy_ids.add(trunk_id)
            self.lidar_proxy_ids.add(canopy_id)

            # 3) Thermal registration (tune per your needs)
            # Visual URDF for appearance (per_link=True makes sense there)
            self.thermal.register_body(tree_vis_id, str(mesh_path), per_link=True)
            
            # tree_id = p.loadURDF(str(mesh_path), basePosition=[x, y, z], useFixedBase=True, useMaximalCoordinates=True)
            # # Apply green color (RGB + alpha)
            # # p.changeVisualShape(tree_id, linkIndex=0, rgbaColor=[0.1, 0.6, 0.1, 0.950])  # canopy
            # self.thermal.register_body(tree_id, str(mesh_path), per_link=True)

    def load_clouds(self,feat,mesh_path):
        cloud_id = p.loadURDF(str(mesh_path), basePosition=[0,0,0], useFixedBase=True)
        
        if feat['position_type'] == "patch" or "patch" in feat:
            # place clouds in the patch area
            poly_coords = np.asarray(feat["patch"]).reshape((-1,2))
            if poly_coords.shape[1]>2:
                # 3D coordinate given
                patch_polygon = Polygon(poly_coords.reshape((-1,3)))
            else:
                # 2D coordinate given
                patch_polygon = Polygon(poly_coords.reshape((-1,2)))
            
            Num_feat = self.determine_N_features(feat=feat,id=cloud_id,area=patch_polygon.area, N_min=1)
            
            pos = self.sample_points_in_polygon(patch_polygon, Num_feat, buffer=1.0)

        elif feat['position_type'] == "position":
            # Place clouds in specified position
            pos = np.asarray(feat['position']).reshape((-1,3))
        elif feat['position_type'] == "random":
            # Place the cloud in a random position
            Num_feat = self.determine_N_features(feat=feat,id=cloud_id,area=self.terrain['terrain_area'], N_min=5)
            # if "density" in feat:
            #     Num_feat = self.compute_N_from_density(id=cloud_id, density=feat['density'], N_min=5, area=self.terrain['terrain_area'])
                
            pos = np.random.uniform(low=-1*np.max(self.terrain['terrain_bounds']), high=1*np.max(self.terrain['terrain_bounds']), size=(Num_feat, 3))
        if 'altitude' in feat:
            pos[:,2] = feat['altitude']
        else:
            pos[:,2] = np.random.uniform(low=20, high=100, size=(Num_feat, 1)).reshape((-1,))

        for x, y, z in pos:
            z_terrain = self.get_height_from_heightmap(x, y, self.terrain['height_data'], sx=10, sy=10, sz=1.0)
            cloud_id = p.loadURDF(str(mesh_path), basePosition=[x, y, z], useFixedBase=True, useMaximalCoordinates=True)
            # Apply cloud color (RGB + alpha)
            # p.changeVisualShape(cloud_id, linkIndex=0, rgbaColor=[0.7, 0.7, 0.7, 1.0])  # canopy
            self.thermal.register_body(cloud_id, str(mesh_path), per_link=False)
    
    def compute_N_from_density(self,id,density,area,N_min=5):
        # Load Mesh
        vis_data = p.getVisualShapeData(id,-1)
        obj_path = Path(str(vis_data[0][4]).replace("b'","").replace("'",""))
        p.removeBody(id)
        mesh = trimesh.load(str(obj_path))
        # Project points onto the XY plane (z = 0)
        projected_points = mesh.vertices[:,:2]  # Drop the Z-coordinate
        
        # Compute the convex hull of the projected points
        hull = ConvexHull(projected_points)

        # Calculate the area of the convex hull
        projected_area = hull.area

        return max(round((density * area) / projected_area), N_min)

    def determine_N_features(self,feat,id,area=0,N_min=1):
        if "N" in feat:
            Num_feat = int(feat["N"])
        elif "density" in feat:
            # Num_feat = max(int(feat["density"] * patch_polygon.area), 5)
            Num_feat = self.compute_N_from_density(id=id, density=feat['density'], N_min=N_min, area=area)
        else:
            Num_feat = 1
        return Num_feat

    def read_asc_file(self,filepath):
        with open(filepath, 'r') as f:
            header = {}
            for _ in range(6):
                key, value = f.readline().strip().split()
                header[key] = float(value) if '.' in value else int(value)

            data = np.loadtxt(f)
        return header, data
    
    def _asc_origin_from_header(self):
        """Return lower-left *corner* (x0,y0) and cellsize from ASC header."""
        cs = float(self.header["cellsize"])
        if "xllcorner" in self.header and "yllcorner" in self.header:
            x0 = float(self.header["xllcorner"])
            y0 = float(self.header["yllcorner"])
        elif "xllcenter" in self.header and "yllcenter" in self.header:
            # convert center of LL cell to corner of raster
            x0 = float(self.header["xllcenter"]) - 0.5*cs
            y0 = float(self.header["yllcenter"]) - 0.5*cs
        else:
            raise KeyError("Header must contain xllcorner/yllcorner or xllcenter/yllcenter")
        return x0, y0, cs
      
    def get_height_from_heightmap(self, x_world, y_world, heightmap, sx=0.1, sy=0.1, sz=1.0):
        rows, cols = heightmap.shape

        # Convert world (x, y) to heightmap (i, j)
        j = int((x_world / (cols * sx)) * cols)
        i = int((y_world / (rows * sy)) * rows)

        # Clamp to bounds
        i = np.clip(i, 0, rows - 1)
        j = np.clip(j, 0, cols - 1)

        raw_height = heightmap[i, j]  # height value in [0, 1] or elevation
        return raw_height * sz
        
    def sample_points_in_polygon(self,polygon, n_points, buffer=1.0):
        minx, miny, maxx, maxy = polygon.bounds
        points = []
        z = 0
        while len(points) < n_points:
            x = np.random.uniform(minx - buffer, maxx + buffer)
            y = np.random.uniform(miny - buffer, maxy + buffer)
            
            if polygon.contains(Point(x, y)):
                points.append((x, y, z))
        return np.asarray(points).reshape((-1,3))
    
    def _disable_all_collisions(self, body_id: int):
        """Disable collisions for every link (including base=-1)."""
        num_joints = p.getNumJoints(body_id)
        # base link
        p.setCollisionFilterGroupMask(body_id, -1, 0, 0)
        # all child links
        for j in range(num_joints):
            p.setCollisionFilterGroupMask(body_id, j, 0, 0)

    def _create_tree_collision_proxy(
        self,
        base_pos,
        base_orn=(0, 0, 0, 1),
        trunk_radius=0.25,
        trunk_height=2.0,
        trunk_center_z=1.0,     # trunk collision center relative to base
        canopy_radius=2.0,
        canopy_center_z=5.0     # canopy collision center relative to base (matches your URDF)
    ):
        base_pos = np.asarray(base_pos, dtype=np.float32).reshape(3,)

        # --- trunk: cylinder along local Z ---
        trunk_col = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=float(trunk_radius),
            height=float(trunk_height)
        )
        trunk_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=trunk_col,
            basePosition=(base_pos + np.array([0, 0, trunk_center_z], dtype=np.float32)).tolist(),
            baseOrientation=base_orn
        )

        # --- canopy: sphere ---
        canopy_col = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=float(canopy_radius)
        )
        canopy_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=canopy_col,
            basePosition=(base_pos + np.array([0, 0, canopy_center_z], dtype=np.float32)).tolist(),
            baseOrientation=base_orn
        )

        return trunk_id, canopy_id
    
    def _obj_bounds(self, obj_path: str):
        """Return (mins, maxs) of OBJ vertices in local coordinates."""
        mins = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float64)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    mins = np.minimum(mins, v)
                    maxs = np.maximum(maxs, v)

        if not np.isfinite(mins).all():
            raise ValueError(f"No vertices found in OBJ: {obj_path}")
        return mins, maxs

    def _apply_scale_and_origin(self, mins, maxs, scale=(1, 1, 1), origin_xyz=(0, 0, 0)):
        s = np.array(scale, dtype=np.float64)
        o = np.array(origin_xyz, dtype=np.float64)
        mins_s = mins * s + o
        maxs_s = maxs * s + o
        return mins_s, maxs_s

    def _bounds_to_cylinder(self, mins, maxs):
        """Approximate bounds with Z-axis cylinder."""
        ext = maxs - mins
        height = float(ext[2])
        radius = float(0.5 * max(ext[0], ext[1]))
        center_z = float(0.5 * (mins[2] + maxs[2]))
        return radius, height, center_z

    def _bounds_to_sphere(self, mins, maxs):
        """Approximate bounds with sphere (enclosing)."""
        ext = maxs - mins
        radius = float(0.5 * max(ext[0], ext[1], ext[2]))
        center_z = float(0.5 * (mins[2] + maxs[2]))
        return radius, center_z

    def _calibrate_tree_proxy_from_urdf(self, urdf_path: Path):
        """
        Reads the URDF, finds trunk/canopy collision mesh filenames + scale + origin,
        parses OBJs to get bounds, and returns tuned proxy parameters.

        Returns dict with:
        trunk_radius, trunk_height, trunk_center_z, canopy_radius, canopy_center_z
        """
        urdf_path = Path(urdf_path)
        urdf_dir = urdf_path.parent

        tree = ET.parse(str(urdf_path))
        root = tree.getroot()

        def parse_xyz(s, default=(0, 0, 0)):
            if s is None:
                return default
            vals = [float(x) for x in s.strip().split()]
            return tuple(vals[:3])

        def parse_scale(s, default=(1, 1, 1)):
            if s is None:
                return default
            vals = [float(x) for x in s.strip().split()]
            return tuple(vals[:3])

        # Collect link collision mesh info
        link_info = {}
        for link in root.findall("link"):
            lname = link.get("name", "")
            col = link.find("collision")
            if col is None:
                continue
            origin = col.find("origin")
            origin_xyz = parse_xyz(origin.get("xyz") if origin is not None else None, (0, 0, 0))

            geom = col.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue

            filename = mesh.get("filename")
            scale = parse_scale(mesh.get("scale"), (1, 1, 1))

            # Resolve OBJ path relative to URDF directory
            obj_path = (urdf_dir / filename).resolve() if filename else None
            if obj_path is None or not obj_path.exists():
                # If your URDF uses package:// or other schemes, you'd resolve them here
                raise FileNotFoundError(f"Could not resolve mesh '{filename}' from URDF: {urdf_path}")

            link_info[lname] = {
                "obj_path": str(obj_path),
                "scale": scale,
                "origin_xyz": origin_xyz
            }

        # Heuristic: match by link names (your URDF uses 'trunk' and 'canopy')
        if "trunk" not in link_info or "canopy" not in link_info:
            raise KeyError(f"URDF does not contain expected links 'trunk' and 'canopy': {list(link_info.keys())}")

        # --- trunk ---
        t = link_info["trunk"]
        tmins, tmaxs = self._obj_bounds(t["obj_path"])
        tmins, tmaxs = self._apply_scale_and_origin(tmins, tmaxs, t["scale"], t["origin_xyz"])
        trunk_radius, trunk_height, trunk_center_z = self._bounds_to_cylinder(tmins, tmaxs)

        # --- canopy ---
        c = link_info["canopy"]
        cmins, cmaxs = self._obj_bounds(c["obj_path"])
        cmins, cmaxs = self._apply_scale_and_origin(cmins, cmaxs, c["scale"], c["origin_xyz"])
        canopy_radius, canopy_center_z = self._bounds_to_sphere(cmins, cmaxs)

        return {
            "trunk_radius": trunk_radius,
            "trunk_height": trunk_height,
            "trunk_center_z": trunk_center_z,
            "canopy_radius": canopy_radius,
            "canopy_center_z": canopy_center_z,
        }

