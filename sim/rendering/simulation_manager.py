"""
File for the class `SimulationManager`,
the handler for interfacing and adding objects to panad3d
"""
from typing import Any
from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath, PandaNode


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

    def generate_simulation_node(self, object_to_change, model, parent= None):
        """_summary_

        Attaches a Panda3D node to the object. Does not allow object to have animations
        Configures it to make it appearl
        Args:
            object (Any): Object to add a non-animatable model to
            kwargs (Any): Additional special settings
        """
        
        object_to_change.object_node = PandaNode(object_to_change.name)
        object_to_change.object_node_path = NodePath(object_to_change.object_node)
        
        try:
            object_to_change.model_node = self.world.loader.loadModel(model)
            object_to_change.model_node_path = NodePath(object_to_change.model_node)
            object_to_change.model_node_path.reparentTo(object_to_change.object_node_path)
        except (TypeError) as error:
            print("No path for the model was listed. Check your configurations again!")
            raise error
        
    
    def render_object(self, object):
        if object.parent_node_path is not None:
            object.object_node_path.reparentTo(object.parent_node_path)
        else:
            object.object_node_path.reparentTo(self.world.render)


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

        child.parent_node_path = parent.object_node_path
        child.object_node_path.reparentTo(parent.object_node_path)

    def configure_sim_model(self, object: Any):
        object.configure_model()