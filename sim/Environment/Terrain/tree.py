from pathlib import Path

from sim.Environment.static_object import STATIC_OBJECT


class TREE(STATIC_OBJECT):
    def __init__(
        self,
        loader,
        config=None,
        pos_type=None,
        pos_val=None,
        N=1,
        terrain=None,
        render=None,
    ):
        STATIC_OBJECT.__init__(self, loader=loader, gen_type=pos_type)
        # self._set_loader(loader)
        if type(config) is str:
            self._load(str(Path(config)))
        elif type(config) is dict:
            self._load(str(Path(config["obj_path"])))

        # self._set_position(pos=pos_type,N=N,terrain=terrain,render=render)

    def _set_init_position(
        self, pos=None, min_point=None, max_point=None, on_ground=False
    ):
        return super()._set_init_position(pos, min_point, max_point, on_ground)
