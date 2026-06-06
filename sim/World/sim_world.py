from direct.showbase.ShowBase import ShowBase
from sim.Environment.Terrain.terrain import TERRAIN
from sim.Environment.Tree.tree import TREE
from panda3d.core import CollisionRay, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32
from sim.Environment.Static_Object.static_object import STATIC_OBJECT
from direct.task import Task
import numpy as np
import yaml
from math import pi, sin, cos

class WORLD(ShowBase):

    def __init__(self,config_file):
        ShowBase.__init__(self)

        with open(config_file, 'r') as file:
            yaml_config = yaml.safe_load(file)

        # Import Ground Terrain
        self.terrain = TERRAIN(self.loader,yaml_config['terrain'])
        self.terrain.object.reparentTo(self.render)
        
        # Import Static Environmental Objects
        self.load_objects(yaml_config=yaml_config, object_type="static")

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
    
    def load_objects(self, yaml_config:dict, object_type:str):
        object_dict = yaml_config[object_type]
        self.static_object = {}
        # stat_obj = STATIC_OBJECT(self.loader)
        
        for key in object_dict.keys():
            if key == "trees":
                tree_types = len(object_dict[key].get('obj_path',[]))
                for ii in np.arange(tree_types):
                    self.trees = TREE(loader=self.loader,config=object_dict[key].get('obj_path')[ii],pos_type=object_dict[key].get('obj_pos'))
                    num_trees = object_dict[key].get('obj_number')[ii]
                    for jj in np.arange(num_trees):
                        tree = self.trees.object.instanceTo(self.render)
                        # Find terrain ground
                        min_point, max_point = self.terrain.object.getTightBounds()
                        position = np.random.uniform(low=min_point.x,high=max_point.x,size=(2,))

                        z = self.get_terrain_height(position)
                        if z is None:
                            continue
                        else:
                            position = list(position)
                            position.append(z)
                            # position = np.ndarray(list_pos)
                        tree.setPos(*position)
                        tree.setScale(*object_dict[key].get('obj_scale'))

                        # self.trees.object._set_init_position(min_point=min_point.x,max_point=max_point.x, on_ground=True)
                        # self.trees.object._set_init_position(pos=object_dict[key].get('obj_pos'),N=object_dict[key].get('obj_number')[ii],terrain=self.terrain,render=self.render,object=tree)
                        # tree._set_init_position(pos=object_dict[key].get('obj_pos'),N=object_dict[key].get('obj_number')[ii],terrain=self.terrain,render=self.render)
                        # tree.object._set_scale(object_dict[key].get('obj_scale'))

    def get_terrain_height(self,position):
        self.cTrav = CollisionTraverser()
        self.rayQueue = CollisionHandlerQueue()

        ray = CollisionRay()
        rayNode = CollisionNode("treeRay")
        rayNode.addSolid(ray)
        rayNode.setFromCollideMask(BitMask32.bit(1))
        rayNode.setIntoCollideMask(BitMask32.allOff())

        self.rayNP = self.render.attachNewNode(rayNode)
        self.cTrav.addCollider(self.rayNP, self.rayQueue)
        z = self.terrain.terrain_height_at(x=position[0], y=position[1], ray=ray, render=self.render, cTrav=self.cTrav, rayQueue=self.rayQueue)
        return z
    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(300 * sin(angleRadians), -300 * cos(angleRadians), 100)
        self.camera.setHpr(angleDegrees, -15, 0)
        return Task.cont