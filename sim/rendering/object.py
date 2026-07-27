""" Script that handles all the object rendering. """

from sim.utils.functions import set_attr_from_configuration
from panda3d.core import PandaNode, NodePath

class RenderableObject():
    """_summary_
    Class that managees the rendering of objects into panda3D. Inherited by all rendered objects (except terrain) in the simulation
    """
    
    def __init__(self):
        """_summary_
        Sets up variables and configurations of the rendered object for panda3D. To configure the object and make it actually render, call `configure_model()`.
        To configure textures, call `set_texture()`.
        Args:
            config (dict): _description_
        """
        self.name = ""
        
        self.position = [0, 0, 0]
        self.orentation = [0, 0, 0]
        self.color = [255, 255, 255] # In the format of RGB
        self.scale = 1
        
        self.object_node = None # Dummy node to hold all nodes related to an object.
        self.object_node_path = None
        self.parent_node_path = None
        
        self.model_node = None
        self.model_node_path = None
        
        self.model: str = ""
        self.animations: list = list()
        self.textures: list = list()
        
    def set_configs(self, config:dict, *args, **kwargs):
        """_summary_
        Hook for manually adding configurations 
        Args:
            config (dict): _description_
        """
        
        set_attr_from_configuration(self, config, args, kwargs)
        
    def configure_model(self) -> None:
        """_summary_
        Plugs the object's settings into panda3d. 
        Make sure to pair the node to it's parent node so it will render 

        Args:
            world (ShowBase): _description_
        """

        
        if self.parent_node_path is not None:
            self.object_node_path.setPos(self.parent_node_path, self.position[0], self.position[1], self.position[2])
        else:
            self.object_node_path.setPos(self.position[0], self.position[1], self.position[2])
        
        self.object_node_path.setHpr(self.orentation[0], self.orentation[1], self.orentation[2])
        self.object_node_path.setScale(self.scale)
        
        # ignoring fancy transparency from panda3d 
        self.object_node_path.setColor(self.color[0], self.color[1], self.color[2], 1)
        
        
    def set_textures(self):
        """_summary_
        Renders textures with the provided settings
        """
        
        # Textures in panda3d are really complicated
        pass
    