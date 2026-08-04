"""
Object loader class. This manages all the object loading.
"""

import numpy as np
from sim.Environment.Terrain.tree import TREE


class ObjectLoader:
    """_summary_
    Object loader and manager for the simulation. Loads all objects into the simulation.
    This does not implement any physics so far into the simulation.
    """

    # TODO: Add multi-object loader based on YAML files
    # TODO: Add enumerated classes and switch case instead of if/else statements

    def __init__(self, world):
        """Sets internal variables. Does not load objects"""
        self.world = world

    def load_objects(self, yaml_config: dict, object_type: str):
        """
        Loads objects into the simulation.
        Current owns all objects and does not attach them to the world class
        """

        object_dict = yaml_config[object_type]
        self.static_object = {}
        # stat_obj = STATIC_OBJECT(self.loader)

        for key in object_dict.keys():
            if key == "trees":
                tree_types = len(object_dict[key].get("obj_path", []))
                min_point, max_point = self.world.terrain.object.getTightBounds()
                for ii in np.arange(tree_types):
                    self.trees = TREE(
                        loader=self.world.loader,
                        config=object_dict[key].get("obj_path")[ii],
                        pos_type=object_dict[key].get("obj_pos"),
                    )
                    num_trees = object_dict[key].get("obj_number")[ii]
                    for jj in np.arange(
                        num_trees
                    ):  # This should absolutely not work but it does
                        # tree = self.trees.object.instanceTo(self.render)
                        tree = self.trees.object.copyTo(self.world.render)
                        # Find terrain ground
                        x = np.random.uniform(min_point.x, max_point.x)
                        y = np.random.uniform(min_point.y, max_point.y)
                        position = [x, y]

                        z = self.world.environment_loader.get_terrain_height(position)
                        if z is None:
                            continue
                        else:
                            position = list(position)
                            position.append(z)
                        tree.setPos(*position)
                        tree.setScale(*object_dict[key].get("obj_scale"))
                        self.static_object.setdefault("trees", []).append(tree)
