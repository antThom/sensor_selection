from panda3d.core import Point3
from panda3d.core import Texture
from panda3d.core import TextureStage
from panda3d.core import TexGenAttrib
from panda3d.core import (
    CollisionRay,
    CollisionNode,
    CollisionTraverser,
    CollisionHandlerQueue,
    BitMask32,
)
from sim.utils.object import OBJECT
from pathlib import Path
import numpy as np


class STATIC_OBJECT(OBJECT):
    def __init__(self, loader=None, gen_type=None):
        OBJECT.__init__(self, loader=loader, gen_type=gen_type)

    def _set_texture(self, texture_path: str = None, scale: list = None):
        self.tex = self.loader.loadTexture(str(texture_path))
        self.tex.setWrapU(Texture.WM_repeat)
        self.tex.setWrapV(Texture.WM_repeat)
        self.object.setTexture(self.tex, 1)

        self.object.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldPosition)

        if scale is not None:
            self.object.setTexScale(TextureStage.getDefault(), scale[0], scale[1])
        else:
            self.object.setTexScale(TextureStage.getDefault(), 1, 1)

    def _set_position(
        self,
        pos: str = None,
        N: int = 1,
        pos_val: np.ndarray = None,
        terrain=None,
        render=None,
    ):
        if pos == "center":
            # Get bounding box
            if getattr(self, "object", None) is None:
                min_point, max_point = terrain.getTightBounds()
                center = (min_point + max_point) * 0.5
                bottom_z = min_point.z
                terrain.setPos(-center[0], -center[1], bottom_z)
            else:
                min_point, max_point = self.object.getTightBounds()
                center = (min_point + max_point) * 0.5
                bottom_z = min_point.z
                self.object.setPos(-center[0], -center[1], bottom_z)

        elif pos == "random":
            if terrain.object is not None:
                min_point, max_point = terrain.object.getTightBounds()
            else:
                min_point, max_point = -50, 50
            position = np.random.uniform(low=min_point.x, high=max_point.x, size=(2,))
            self.cTrav = CollisionTraverser()
            self.rayQueue = CollisionHandlerQueue()

            ray = CollisionRay()
            rayNode = CollisionNode("treeRay")
            rayNode.addSolid(ray)
            rayNode.setFromCollideMask(BitMask32.bit(1))
            rayNode.setIntoCollideMask(BitMask32.allOff())

            self.rayNP = render.attachNewNode(rayNode)
            self.cTrav.addCollider(self.rayNP, self.rayQueue)
            z = terrain.terrain_height_at(
                x=position[0],
                y=position[1],
                ray=ray,
                render=render,
                cTrav=self.cTrav,
                rayQueue=self.rayQueue,
            )
            if z is None:
                pass

            self.object.instanceTo(render)
            self.object.setPos(position[0], position[1], z)
            self.object.show()
        else:
            pass

    def _transform_terrain(self, pos: str = None, scale: np.ndarray = None):
        self._set_scale(scale)

        if pos == "center":
            # Get bounding box
            min_point, max_point = self.object.getTightBounds()
            center = (min_point + max_point) * 0.5
            bottom_z = min_point.z
            self.object.setPos(-center[0], -center[1], bottom_z)
