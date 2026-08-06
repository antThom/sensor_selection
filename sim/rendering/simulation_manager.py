"""
File for the class `SimulationManager`,
the handler for interfacing and adding objects to panad3d
"""

from typing import Any

from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath, PandaNode


class SimulationManager:
    """Translate configured simulation objects into Panda3D scene nodes."""

    def __init__(self, show_base: ShowBase):
        self.world = show_base

    def generate_simulation_node(self, object_to_change, model, parent=None):
        """Create a transform root and optionally load its visible model."""

        object_to_change.object_node = PandaNode(object_to_change.name)
        object_to_change.object_node_path = NodePath(object_to_change.object_node)

        # Some sensors only need a transform anchor, not visible geometry.
        if not model:
            object_to_change.model_node = None
            object_to_change.model_node_path = None
            return object_to_change.object_node_path

        try:
            object_to_change.model_node = self.world.loader.loadModel(model)
            object_to_change.model_node_path = NodePath(object_to_change.model_node)
            object_to_change.model_node_path.reparentTo(
                object_to_change.object_node_path
            )
        except TypeError as error:
            print("No path for the model was listed. Check your configurations again!")
            raise error

    def render_object(self, object):
        """Attach an object's transform root to its parent or the world root."""
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
        """Parent a child simulation object to another object's root."""

        child.parent_node_path = parent.object_node_path
        child.object_node_path.reparentTo(parent.object_node_path)

    def configure_sim_model(self, object: Any):
        """Apply the object's configured transform and appearance."""
        object.configure_model()
