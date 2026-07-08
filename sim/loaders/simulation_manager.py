"""
File for the class `SimulationManager`,
the handler for interfacing and adding objects to panad3d
"""
from typing import Any
from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase

class SimulationManager:
    """_summary_
    Class that handles and interfaces with third party SDKs and APIs such as Panda3D and PyBullet, allowing for
    easier interfacing with the simulation and underlying code. Also allows decoupling for under the hood optimization

    TLDR: *I make stuff look and sound pretty :)*
    """

    def __init__(self, show_base:ShowBase):
        self.world = show_base

    def generate_simulation_actor(self, object_to_change:Any, config:Any):
        """_summary_

        Attaches a Panda3D Actor to the object. Allows object to have animations
        Yet to be implemented

        Args:
            object (Any): Object to add an animateable model to
        """
        pass

    def generate_simulation_node(self, object_to_change, model):
        """_summary_

        Attaches a Panda3D node to the object. Does not allow object to have animations
        Configures it to make it appearl
        Args:
            object (Any): Object to add a non-animatable model to
            kwargs (Any): Additional special settings
        """
        
        
        object_to_change.actor = self.world.loader.loadModel(model)
        object_to_change.actor.reparentTo(self.world.render)
        
        # Go through relevant attributes and set them to where the object is
        object_to_change
        

        
    def configure_sim_model(self, object_to_change, defaults=None, **kwargs):
        """_summary_
        Configures the model to alternative configurations
        
        Args:
            object (_type_): _description_
        """

                # Future me can do magic variable changes with match case and kwargs in
        # configuration searching and other matching
        
        if defaults != None:
            object_to_change.actor.setPos(object_to_change.position[1,1], object_to_change.position[2,1], object_to_change.position[3,1])
            object_to_change.actor.setScale(object_to_change.scale)
            # object_to_change.actor.setHpr()
            
        
        
    def _configuration_keyword_matcher(self, config: dict) -> dict:
        """_summary_
        Filters through a configuration set and outputs a dictionary of configurations needed by the simulation manager

        Args:
            config (dict): Configuration from a JSON or YAML file

        Returns:
            dict: Dictionary of needed files
        """

    def attach_sound(self, object, config) -> None:
        """_summary_
        Attaches the sound system to an object and configures it. Edits their attributes in order to add it to the sound system

        Args:
            object (_type_): _description_
            config (_type_): _description_
        """

    def parent_object_models(self, parent, child) -> None:
        """_summary_
        Wrapper of Panda3d' parent fuction. Takes any two objects and parents one to another.

        Args:
            parent (Any): _description_
            child (Any): _description_
        """

        """
        This function searches the attributes of the objects, finds where the model is located 
        (should be named the same thing anyways if done by this manager), and changes the order of the 
        node graph with minor overhead by the caller.
        """

        pass
