import numpy as np
import math
import os 
import sim.print_helpers as ph
import pybullet as p
import pybullet_data
from pathlib import Path
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
import trimesh

class Environment:
    def __init__(self, terrain):
        self.terrain_config = terrain
        base_dir = os.getcwd()
        config_file = os.path.join(base_dir,os.path.normpath(self.terrain_config["Layered Surface"]["Mesh"]))
        
        print(f"Loading the environment from {config_file}")
        # Load the terrain
        print(f"Loading the terrain: {ph.YELLOW}STARTED{ph.RESET}")
        self.load_terrain(config_file, base_dir)
        print(f"Loading the terrain: {ph.GREEN}COMPLETE{ph.RESET}")

        # Load features
        print(f"Loading the terrain features: {ph.YELLOW}STARTED{ph.RESET}")
        self.load_features(config_file, base_dir)
        print(f"Loading the terrain features: {ph.GREEN}COMPLETE{ph.RESET}")
                    

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

            self.terrain['terrain_shape'] = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[self.terrain['header']["cellsize"], self.terrain['header']["cellsize"], height_scale],
                heightfieldTextureScaling=self.terrain['header']["ncols"] / 2,
                heightfieldData=self.terrain['terrain_data'],
                numHeightfieldRows=self.terrain['header']["nrows"],
                numHeightfieldColumns=self.terrain['header']["ncols"]
            )

            self.terrain['terrain'] = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self.terrain['terrain_shape'],
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

    def load_features(self, config_file, base_dir):
        self.id = []
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
            tree_id = p.loadURDF(str(mesh_path))
            Num_feat = self.determine_N_features(feat=feat,id=tree_id,area=patch_polygon.area, N_min=5)
            # if "N" in feat:
            #     Num_feat = int(feat["N"])
            # elif "density" in feat:
                
            #     Num_feat = self.compute_N_from_density(id=tree_id, density=feat['density'], N_min=5, area=patch_polygon.area)
            # else:
            #     Num_feat = 1
            tree_xy_positions = self.sample_points_in_polygon(patch_polygon, Num_feat, buffer=1.0)
        else:
            tree_xy_positions = np.random.uniform(low=-5, high=5, size=(1, 2))

        for x, y, z in tree_xy_positions:
            z_ter = -self.get_height_from_heightmap(x, y, self.terrain['height_data'], sx=10, sy=10, sz=1.0)
            tree_id = p.loadURDF(str(mesh_path), basePosition=[x, y, z], useFixedBase=True, useMaximalCoordinates=True)
            # Apply green color (RGB + alpha)
            # p.changeVisualShape(tree_id, linkIndex=0, rgbaColor=[0.1, 0.6, 0.1, 0.950])  # canopy
            self.id.append(tree_id)

    def load_clouds(self,feat,mesh_path):
        self.cloud_ids = []
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
            self.id.append(cloud_id)

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
