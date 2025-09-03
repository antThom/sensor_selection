import numpy as np
import os 
import sim.print_helpers as ph
import pybullet as p
import pybullet_data

class Environment:
    def __init__(self, config_file):
        print(f"Loading the environment from {config_file}")

        if "asc" in config_file:
            print(f"{ph.GREEN}Loading asc file{ph.RESET}")
            self.header, height_data = self.read_asc_file(config_file)
            # Normalize / scale height
            height_data = height_data.astype(np.float32)
            height_scale = 1.0  # vertical exaggeration if needed

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
                basePosition=[0, 0, 0]
            )

            self.terrain_bounds = (self.header['xllcorner'] + self.header['cellsize']*self.header['ncols'],self.header['yllcorner'] + self.header['cellsize']*self.header['nrows'])

        else:
            print(f"{ph.RED}Loading a different format{ph.RESET}")

    def read_asc_file(self,filepath):
        with open(filepath, 'r') as f:
            header = {}
            for _ in range(6):
                key, value = f.readline().strip().split()
                header[key] = float(value) if '.' in value else int(value)

            data = np.loadtxt(f)
        return header, data
    

