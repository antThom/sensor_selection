from panda3d.core import (
    CollisionRay,
    CollisionNode,
    CollisionTraverser,
    CollisionHandlerQueue,
    BitMask32,
)
from pathlib import Path
import numpy as np


class OBJECT:
    def __init__(
        self,
        pos=None,
        ori=None,
        vel=None,
        ang_vel=None,
        loader=None,
        gen_type: str = None,
    ):
        self.gen_type = gen_type

        if loader is not None:
            self._set_loader(loader)

        if pos is not None:
            self._set_init_position(pos)

    def _set_loader(self, loader):
        self.loader = loader

    def _load(self, filename: str, hide: bool = False):
        self.object = self.loader.loadModel(filename)
        self.object.setCollideMask(BitMask32.bit(1))
        if hide:
            self.object.hide()
        return self.object

    def _set_scale(self, scale: np.ndarray = None):
        if scale is not None:
            self.object.setScale(*scale)
        else:
            self.object.setScale(0.25, 0.25, 0.25)

    def _set_init_position(
        self,
        pos=None,
        min_point: list = None,
        max_point: list = None,
        on_ground: bool = False,
    ):
        if self.gen_type is None:
            self.object.setPos(*pos)
        elif self.gen_type == "random":
            if min_point is None:
                # Set min_point to 0
                min_point = 0
            if max_point is None:
                # Set the max_point to inf
                max_point = np.inf
            if not on_ground:
                pos = np.random.uniform(low=min_point, high=max_point, size=(3,))
            else:
                pos = np.random.uniform(low=min_point, high=max_point, size=(2,))
                # Determine Terrain Height pos_z
            self.object.setPos(*pos)
