"""Script that handles all the object rendering."""

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

    def set_configs(self, config: dict, *args, **kwargs):
        """Apply one or more nested configuration mappings."""
        set_attr_from_configuration(self, config, args, kwargs)

    def configure_model(self) -> None:
        """Apply configured transform and display color to the scene node."""
        if self.object_node_path is None:
            raise RuntimeError("generate a simulation node before configuring it")

        if self.parent_node_path is not None:
            self.object_node_path.setPos(
                self.parent_node_path,
                self.position[0],
                self.position[1],
                self.position[2],
            )
        else:
            self.object_node_path.setPos(
                self.position[0], self.position[1], self.position[2]
            )

        self.object_node_path.setHpr(
            self.orientation[0], self.orientation[1], self.orientation[2]
        )
        self.object_node_path.setScale(self.scale)

        self.object_node_path.setColor(self.color[0], self.color[1], self.color[2], 1)
