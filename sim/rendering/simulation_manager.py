"""Panda3D node construction and parenting helpers."""

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

        object_to_change.model_node = self.world.loader.loadModel(model)
        object_to_change.model_node_path = NodePath(object_to_change.model_node)
        object_to_change.model_node_path.reparentTo(object_to_change.object_node_path)
        return object_to_change.object_node_path

    def render_object(self, object):
        """Attach an object's transform root to its parent or the world root."""
        if object.parent_node_path is not None:
            object.object_node_path.reparentTo(object.parent_node_path)
        else:
            object.object_node_path.reparentTo(self.world.render)

    def parent_object_models(self, parent, child) -> None:
        """Parent a child simulation object to another object's root."""

        child.parent_node_path = parent.object_node_path
        child.object_node_path.reparentTo(parent.object_node_path)

    def configure_sim_model(self, object: Any):
        """Apply the object's configured transform and appearance."""
        object.configure_model()
