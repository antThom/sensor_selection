"""Script that handles all the object rendering."""

from sim.utils.functions import set_attr_from_configuration
from panda3d.core import PandaNode, NodePath
from direct.showbase.ShowBase import ShowBase
from sim.utils.functions import set_attr_from_configuration

class RenderableObject:
    """Configuration and Panda3D node references shared by rendered objects."""

    def __init__(self):
        self.name = ""

        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0]
        self.color = [255, 255, 255]  # In the format of RGB
        self.scale = 1

        self.object_node = None
        self.object_node_path = None
        self.parent_node_path = None

        self.model_node = None
        self.model_node_path = None

        self.model: str = ""
        self.animations: list = []
        self.textures: list = []
        
        self.hidden = False

        # This class does all physics and math done by PyBullet

    def parent_object_models(self, parent, child) -> None:
        """Parent a child simulation object to another object's root."""

        child.parent_node_path = parent.object_node_path
        child.object_node_path.reparentTo(parent.object_node_path)

    def hide(self):
        """Hides the object from view"""
        self.object_node_path.hide()

    def unhide(self):
        """Shows the object if hidden"""
        self.object_node_path.show()


class RenderableObjectBuilder:
    """Builder to construct RenderableObjects"""

    def __init__(self, show_base: ShowBase):
        """
        Creates a builder to create renderable objects with default configs.
        You must configure the model for the object to render.
        """

        self.show_base = show_base
        self._renderable_object: RenderableObject = (
            RenderableObject()
        )  # Empty object to modify

        # Below are default configurations

        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0]
        self.color = [255, 255, 255]  # In the format of RGB
        self.scale = 1

        self._is_actor = False
        self._has_model = True

        self._parent_node_path = None

        self._model: str = ""
        self._animations: list = []
        self._textures: list = []

    def is_actor(self, value: bool):
        """Sets whether the object is an actor or not"""
        self._is_actor = value
        return self  # Allows for chaining in constructor

    def has_model(self, value: bool):
        """Sets whether the object has a renderable model or not"""
        self._has_model = value
        return self  # Allows for chaining in constructor

    def with_object(self, object_to_modify):
        """
        Allows the builder to modify the object instead  of building an object
        from scratch. Call this first if using this option.
        """

        self._renderable_object = object_to_modify
        return self  # Allows to chain function calls

    def with_configurations(self, config: dict, *args, **kwargs):
        """
        Apply one or more nested configuration mappings onto the builder at
        once. This will match configurations to configs of the builder
        """
        set_attr_from_configuration(self, config, args, kwargs)
        return self
    
    def config_from_object(self, object):
        """
        Use configurations from the object specified.
        If applicable, the attributes will be applied to the builder.
        """
        set_attr_from_configuration(self, object.__dict__)

    def with_parent(self, parent_object_node):
        """Attaches the parent object node to the object"""
        self._renderable_object.parent_node_path = parent_object_node

    def with_model(self, model_path: str):
        """Creates the node with the model given"""
        self._renderable_object.model = model_path
        return self

    def with_animations(self, *args, **kwargs):
        """ "Sets animation for the created model"""
        self._is_actor = True

        return self

    def with_textures(self, *args, **kwargs):
        """Sets textures of the created object"""

        return self

    def _reset(self):
        """Resets the builder to build a new object"""
        self.__init__(self.show_base)

    def _generate_simulation_node(self):
        """Create a transform root and optionally load its visible model."""

        self._renderable_object.object_node = PandaNode(self._renderable_object.name)
        self._renderable_object.object_node_path = NodePath(
            self._renderable_object.object_node
        )

        # Some cases only need a transform anchor, not visible geometry.
        if not self._has_model:
            self._renderable_object.model_node = None
            self._renderable_object.model_node_path = None

        if self._model == "":
            print(
                "Rendering an object must have a model. If you do not want a model, ensure that it is disabled."
            )
            raise AttributeError

        try:
            self._renderable_object.model_node = self.show_base.loader.loadModel(model)
            self._renderable_object.model_node_path = NodePath(
                self._renderable_object.model_node
            )
            self._renderable_object.model_node_path.reparentTo(
                self._renderable_object.object_node_path
            )
        except TypeError as error:
            print("No path for the model was listed. Check your configurations again!")
            raise error

    def _generate_simulation_actor(self, *args, **kwargs):
        """Creates a panda3d actor, which can have it's model move"""

        if not self._is_actor:
            print(
                "This object is not an actor! Ensure that this node is configured to be an actor"
            )
            raise AttributeError

        if not self._has_model:
            print("Actors must have models!")
            raise AttributeError

    def _configure_model(self) -> None:
        """Apply configured transform and display color to the scene node."""
        if self._renderable_object.object_node_path is None:
            raise RuntimeError("generate a simulation node before configuring it")

        if self._parent_node_path is not None:
            self._renderable_object.object_node_path.setPos(
                self.parent_node_path,
                self.position[0],
                self.position[1],
                self.position[2],
            )
        else:
            self._renderable_object.object_node_path.setPos(
                self.position[0], self.position[1], self.position[2]
            )

        self._renderable_object.object_node_path.setHpr(
            self.orientation[0], self.orientation[1], self.orientation[2]
        )
        self._renderable_object.object_node_path.setScale(self.scale)

        self._renderable_object.object_node_path.setColor(
            self.color[0], self.color[1], self.color[2], 1
        )

    def _render_object(self):
        """Attach an object's transform root to its parent or the world root."""
        if self._parent_node_path is not None:
            self._renderable_object.object_node_path.reparentTo(self.parent_node_path)
        else:
            self._renderable_object.object_node_path.reparentTo(self.show_base.render)

    def _load_animations(self, *args, **kwargs):
        """_summary_ Loads animations into the model"""
        self._using_animations = True

    def _attach_textures(self, *args, **kwargs):
        """Attaches textures onto the model of the RenderableObject"""

    def build(self):
        """Builds with created configurations"""
        # attach parents

        if not self._is_actor:
            self._generate_simulation_actor()
        else:
            self._generate_simulation_node()

        self._configure_model()
        self._attach_textures()

        if self._using_animations:
            self._load_animations()

        self._render_object()

        product = self._renderable_object
        self._reset()
        return product
