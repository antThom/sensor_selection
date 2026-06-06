from panda3d.core import Point3
from panda3d.core import Texture
from panda3d.core import TextureStage
from panda3d.core import TexGenAttrib
from pathlib import Path
import numpy as np
from sim.Environment.Static_Object.static_object import STATIC_OBJECT

class TERRAIN(STATIC_OBJECT):
    def __init__(self,loader,config=None):
        STATIC_OBJECT.__init__(self)
        self._set_loader(loader)
        self._load(str(Path(config['obj_path'])))
        self._set_texture(texture_path=str(Path(config['texture_path'])), scale=config['texture_scale'])
        self._transform_terrain(pos=config["obj_pos"],scale=config["obj_scale"])
        # terrain_path = Path("generation","Terrain_Generation","terrain_type","Mountain","Mountain_5Peaks_200Height_0Seed_debug.obj")
        # self.terrain = loader.loadModel(str(terrain_path))

        # texture_path = Path("sim","assets","textures","rock.jpg")
        # self.__set_texture(loader=loader,texture_path=texture_path,u_scale=0.005,v_scale=0.005)

    def terrain_height_at(self,x, y, ray, cTrav, rayQueue, render=None):
        ray.setOrigin(x, y, 1000)
        ray.setDirection(0, 0, -1)

        cTrav.traverse(render)
        if rayQueue.getNumEntries() > 0:
            rayQueue.sortEntries()
            entry = rayQueue.getEntry(0)
            return entry.getSurfacePoint(render).z
        return None