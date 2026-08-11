"""Script that handles all the object rendering."""

from panda3d.core import PandaNode, NodePath
from direct.showbase.ShowBase import ShowBase
from sim.utils.functions import set_attr_from_configuration
from sim.utils.builder import BuilderTemplate
import functools


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

    @property
    def position(self):
        return self.position
    
    @position.setter
    def position(self, value):
        if value is not isinstance(list):
            raise TypeError("Position must be a list in [x, y, z] format.")
        if len(value) != 3:
            raise ValueError("Position must be a list in [x, y, z] format.")
        
        self.position = value
        self.object_node_path.setPos(self.position[0], self.position[1], self.position[2])
    
    @property
    def orientation(self):
        """Getter for a RenderableObjectBuilder's orientation"""
        return self.orientation
    
    @orientation.setter
    def orientation(self, value):
        """Setter for a RenderableObjectBuilder's orientation"""
        if value is not isinstance(list):
            raise TypeError("Orientation must be a list in [x, y, z] format.")
        if len(value) != 3:
            raise ValueError("Orientation must be a list in [x, y, z] format.")
        self.orientation = value
        self.object_node_path.setHpr(self.orientation[0], self.orientation[1], self.orientation[2])
    
    @property
    def color(self):
        """Getter for a RenderableObjectBuilder's color array"""
        return self.color
    
    @color.setter
    def color(self, value):
        """Setter for a RenderableObjectBuilder's color array"""
        if value is not isinstance(list):
            raise TypeError("Color must a list in the format of [R, G, B]")
        if len(value) != 3:
            raise ValueError("Color must a list in the format of [R, G, B]")
        for i in list:
            if i is not (isinstance(int) or isinstance(float)):
                raise TypeError("Color values must be either an integer or float.")
            if i > 255:
                raise ValueError("Color values cannot go over 255")
            if i < 0:
                raise ValueError("Color values cannot be negative")
        self.color = value
        self.object_node_path.setColor(self.color[0], self.color[1], self.color[2], 1)
        
    @property
    def scale(self):
        """Getter for a RenderableObjectBuilder's scale"""
        return self.scale
    
    @scale.setter
    def scale(self, value):
        """Setter for a RenderableObjectBuilder's scale"""
        if value == 0:
            raise ValueError("Scale cannot be zero!")
        if value < 0:
            raise ValueError("Scale cannot be negative.")
        self.scale = value
        self.object_node_path.setScale(self.scale)

    @staticmethod
    def parent_object_models(parent: RenderableObject, child: RenderableObject) -> None:
        """Parent a child simulation object to another object's root."""

        child.parent_node_path = parent.object_node_path
        child.object_node_path.reparentTo(parent.object_node_path)
        
    def parent_node_to(self, parent) -> None:
        self.parent_node_path = parent
        self.parent_object_models(parent, self)

    def hide(self):
        """Hides the object from view"""
        self.object_node_path.hide()
        self.hidden = True

    def unhide(self):
        """Shows the object if hidden"""
        self.object_node_path.show()
        self.hidden = False


class RenderableObjectBuilder(BuilderTemplate):
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
        self._using_animations = False

        self._parent_node_path = None

        self.model: str = ""
        self._animations: list = []
        self._textures: list = []

    def chainable(method):
        """
        Decorator to enable a function to be chained on others when calling the builder.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            method(self, *args, **kwargs)
            return self
        return wrapper
    
    @property
    def position(self):
        return self.position
    
    @position.setter
    @chainable
    def position(self, value):
        if value is not isinstance(list):
            raise TypeError("Position must be a list in [x, y, z] format.")
        if len(value) != 3:
            raise ValueError("Position must be a list in [x, y, z] format.")
        
        self.position = value
    
    @property
    def orientation(self):
        """Getter for a RenderableObjectBuilder's orientation"""
        return self.orientation
    
    @orientation.setter
    @chainable
    def orientation(self, value):
        """Setter for a RenderableObjectBuilder's orientation"""
        if value is not isinstance(list):
            raise TypeError("Orientation must be a list in [x, y, z] format.")
        if len(value) != 3:
            raise ValueError("Orientation must be a list in [x, y, z] format.")
        self.orientation = value
    
    @property
    def color(self):
        """Getter for a RenderableObjectBuilder's color array"""
        return self.color
    
    @color.setter
    @chainable
    def color(self, value):
        """Setter for a RenderableObjectBuilder's color array"""
        if value is not isinstance(list):
            raise TypeError("Color must a list in the format of [R, G, B]")
        if len(value) != 3:
            raise ValueError("Color must a list in the format of [R, G, B]")
        for i in list:
            if i is not (isinstance(int) or isinstance(float)):
                raise TypeError("Color values must be either an integer or float.")
            if i > 255:
                raise ValueError("Color values cannot go over 255")
            if i < 0:
                raise ValueError("Color values cannot be negative")
        self.color = value
        
    @property
    def scale(self):
        """Getter for a RenderableObjectBuilder's scale"""
        return self.scale
    
    @scale.setter
    @chainable
    def scale(self, value):
        """Setter for a RenderableObjectBuilder's scale"""
        if value == 0:
            raise ValueError("Scale cannot be zero!")
        if value < 0:
            raise ValueError("Scale cannot be negative.")
        self.scale = value
    

    @chainable
    def is_actor(self, value: bool):
        """Sets whether the object is an actor or not"""
        self._is_actor = value

    @chainable
    def has_model(self, value: bool):
        """Sets whether the object has a renderable model or not"""
        self._has_model = value
        return self  # Allows for chaining in constructor

    @chainable
    def with_object(self, object_to_modify):
        """
        Allows the builder to modify the object instead  of building an object
        from scratch. The object must inherit from RenderableObject.
        """

        self._renderable_object = object_to_modify
        return self  # Allows to chain function calls

    @chainable
    def with_configurations(self, config: dict, *args, **kwargs):
        """
        Apply one or more nested configuration mappings onto the builder at
        once. This will match configurations to configs of the builder
        """
        set_attr_from_configuration(self, config, args, kwargs)
        return self
    
    @chainable
    def config_from_object(self, object):
        """
        Use configurations from the object specified.
        If applicable, the attributes will be applied to the builder.
        The object must inherit from RenderableObject
        """
        
        # TODO: Safety check on whether object has it
        # The object should have these attributes because it's a
        # renderable object
        self.position = object.position
        self.orientation = object.orientation
        self.color = object.color
        self.scale = object.scale
        self.model = object.model

    @chainable
    def with_parent(self, parent_object_node):
        """Attaches the parent object node to the object"""
        self._renderable_object.parent_node_path = parent_object_node

    @chainable
    def with_model(self, model_path: str):
        """Creates the node with the model given"""
        self.model = model_path
        return self

    @chainable
    def with_animations(self, *args, **kwargs):
        """ "Sets animation for the created model"""
        self._is_actor = True

        return self

    @chainable
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

        if self.model == "":
            raise AttributeError("Rendering an object must have a model. If you do not want a model, ensure that it is disabled.")

        try:
            self._renderable_object.model_node = self.show_base.loader.loadModel(self.model)
            self._renderable_object.model_node_path = NodePath(
                self._renderable_object.model_node
            )
            self._renderable_object.model_node_path.reparentTo(
                self._renderable_object.object_node_path
            )
        except TypeError:
            raise TypeError("No path for the model was listed. Check your configurations again!")

    def _generate_simulation_actor(self, *args, **kwargs):
        """Creates a panda3d actor, which can have it's model move"""

        if not self._is_actor:
            raise AttributeError("This object is not an actor. Ensure that this node is configured to be an actor")

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

        if self._is_actor:
            self._generate_simulation_actor()
        else:
            self._generate_simulation_node()

        self._configure_model()
        self._attach_textures()

        if self._using_animations:
            self._load_animations()

        self._render_object()

        # For creating new objects
        # The return section is dropped when modifying an object
        product = self._renderable_object
        self._reset()
        return product

