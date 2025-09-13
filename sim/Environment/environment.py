import numpy as np
import math
import os 
import sim.print_helpers as ph
import pybullet as p
import pybullet_data
from pathlib import Path
from shapely.geometry import Polygon, Point

class Environment:
    def __init__(self, terrain):
        self.terrain_config = terrain
        base_dir = os.getcwd()
        config_file = os.path.join(base_dir,os.path.normpath(self.terrain_config["Layered Surface"]["Mesh"]))
        
        print(f"Loading the environment from {config_file}")

        if "asc" in config_file:
            print(f"{ph.GREEN}Loading asc file{ph.RESET}")
            self.header, height_data = self.read_asc_file(config_file)
            # Normalize / scale height
            height_data = height_data.astype(np.float32)
            height_scale = 1.0  # vertical exaggeration if needed

            # 2) Compute true min/max ignoring NODATA
            hmin = np.nanmin(height_data)
            hmax = np.nanmax(height_data)

            height_data = np.nan_to_num(height_data, nan=hmin)
            z_offset = -hmin * height_scale
            # z_offset = -height_data_min * height_scale

            # Flatten row-major
            self.terrain_data = height_data.flatten()

            self.terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[self.header["cellsize"], self.header["cellsize"], height_scale],
                heightfieldTextureScaling=self.header["ncols"] / 2,
                heightfieldData=self.terrain_data,
                numHeightfieldRows=self.header["nrows"],
                numHeightfieldColumns=self.header["ncols"]
            )

            self.terrain = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self.terrain_shape,
                basePosition=[0, 0, z_offset]
            )

            self.terrain_bounds = np.array(([self.header['xllcorner'],self.header['xllcorner'] + self.header['cellsize']*self.header['ncols']],
                                           [self.header['yllcorner'],self.header['yllcorner'] + self.header['cellsize']*self.header['nrows']]))
        else:
            print(f"{ph.RED}Loading a different format{ph.RESET}")

        # Load Textures
        texture_file_name = os.path.join(base_dir,os.path.normpath(self.terrain_config["Layered Surface"]["Texture"]))
        tex_id = p.loadTexture(texture_file_name)
        p.changeVisualShape(self.terrain, -1, textureUniqueId=tex_id)
        p.changeVisualShape(self.terrain, -1, rgbaColor=[1,1,1,1], specularColor=[0.1,0.1,0.1])

        self.vis, self.col, self.id = [], [], []
        for idx, feat in enumerate(self.terrain_config["Surface Mesh"]):
            if idx>0.0:
                mesh_file_name = str(feat["Mesh"]).strip().strip("'\"")  # remove any stray quotes in the JSON

                mesh_file_path = Path(mesh_file_name)
                if mesh_file_path.is_absolute():
                    mesh_path = mesh_file_path
                else:
                    mesh_path = Path(base_dir) / mesh_file_path

                mesh_path = mesh_path.resolve()
                
                # mesh_file_name = os.path.join(base_dir,os.path.normpath(feat["Mesh"]))
                # mesh_file_name = Path(repr(str(mesh_file_name)))
                if "patch" in feat:
                    poly_coords = np.asarray(feat["patch"]).reshape((-1,2))
                    patch_polygon = Polygon(poly_coords)
                    if "N" in feat:
                        Num_feat = int(feat["N"])
                    elif "density" in feat:
                        Num_feat = max(int(feat["density"] * patch_polygon.area), 5)
                    else:
                        Num_feat = 1
                    tree_xy_positions = self.sample_points_in_polygon(patch_polygon, Num_feat, buffer=1.0)
                else:
                    tree_xy_positions = np.random.uniform(low=-5, high=5, size=(20, 2))
                tree_ids = []

                for x, y in tree_xy_positions:
                    z = -self.get_height_from_heightmap(x, y, height_data, sx=10, sy=10, sz=1.0)
                    tree_id = p.loadURDF(str(mesh_path), basePosition=[x, y, z], useFixedBase=True)
                    # Apply green color (RGB + alpha)
                    p.changeVisualShape(tree_id, linkIndex=0, rgbaColor=[0.1, 0.6, 0.1, 0.950])  # canopy
                    tree_ids.append(tree_id)
                

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
        while len(points) < n_points:
            x = np.random.uniform(minx - buffer, maxx + buffer)
            y = np.random.uniform(miny - buffer, maxy + buffer)
            if polygon.contains(Point(x, y)):
                points.append((x, y))
        return np.asarray(points).reshape((-1,2))
