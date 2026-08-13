# Rendering

The simulation window and rendering are created by Panda3D, a game engine written in C++ that contains a python wrapper. The physics engine is provided by PyBullet. All renderable objects in `sensor_selection` inherit from the `RenderableObject` class, which can be found in `sim/rendering/renderable_object.py`. This class provides a standardized API for controlling an object in panda3d. In order to use and create these objects, use it in conjunction with `simulation_manager`.As part of an effort to modulate the systems of the simulation, all of the rendering instantiation, handling, and utilities are managed by the `SimulatorManager` class.

## `SimulationManager`

There is a single instance of SimulationManager in the entire codebase. The way you call this object depends on the context. It should probably be a global, but those always spell bad ideas.

```python
# Sometimes it appears like this
world.simulation_manger

# In Loaders, it will appear like this
self.world.simulation_manager

# Often it is abbreviated for shorter calls
sim_man = self.world.simulation_manager
sim_man.builder.build()

```

## Rendering an Object

In order to render an object, use `RenderableObjectBuilder`. You can use to create new renderable objects or render classes that inherit the `RenderableObject` class. To see all options that the builder has, go to the API documentation.

```python 
from sim.rendering.renderable_object import RenderableObjectBuilder
builder = RenderableObjectBuilder(world)

# Required Only
mini_renderable = builder.with_model().build()

# Model customization
pretty_renderable = builder.with_model().with_texture().is_actor(True).with_animations().build()

# Renderable without a model
# Ignores the model option
ghost_renderable = builder.has_model(False).build()

from sim.agents.agent import Agent
from sim.environment.thermal.thermal_manager import ThermalManager

# Modifying another object
# As long as it inherits from RenderableObject, you can configure it
# Ignores the `with_configurations` option
wild_agent = Agent(ThermalManager())
builder.with_object(wild_agent).config_from_object(wild_agent).build() 

pretend_agent = Agent(ThermalManager())

# With all settings
extensive_configuration = builder.position([21, 67, 69]) \
    .orientation([180, 180, 180])  \
    .color([255, 20, 20]) \
    .scale(999999) \
    .is_actor(False) \
    .has_model(True)  \
    .with_object(pretend_agent) \
    .with_configurations("configs\\go\\here") \
    .config_from_object(pretend_agent).with_parent(ghost_renderable) \
    .with_model("amazing\\model\\path\\to\\asset") \
    .with_animations("fantastic\\animations\\path") \
    .with_texture("gorgeous\\textures\\path").build() 
```

## Configuring Models

If you want to apply any configurations to a `RenderableObject`, you can programmically set them after creation. If you wish, you can also set these configurations at creation. If you want to set textures or animations, those can only be configured by the builder at creation.


Internally implemented to all of the renderable classes are the attributes to make it renderable. If you ever wish to access the simulation and visual parts of the object, call it with `<object>.model`.

Here are the available attributes that are configurable to render.

```python
from sim.rendering.renderable_object import RenderableObject, RenderableObjectBuilder

renderable = RenderableObject() # Creates with default and no rendering

renderable.position = [123, 456, 789]
renderable.orientation = [45, 60, 90] # In degrees
renderable.color = [0, 230, 255] # RBG
renderable.scale = 0.2

# Callable on two renderable objects
renderable2 = RenderableObject()

parent_object_models(renderable2, renderable)
renderable2.parent_node_to(renderable)

renderable.hide()
renderable.unhide()

```

If you wish to implement more of these functions, search out the [Panda3D documentation](https://docs.panda3d.org/1.10/python/index)

## Configuration Files

All renderable objects will have the same configurations shown below.

- `position`
- `orientation`
- `color`
- `scale`
- `model`
- `animations`
- `textures`

Note that textures and animations features are not available yet.

In the object's configuration file, the object will have these configurations shown below with their defaults.

```yaml
model_configs:
  model: ".\\assets\\PUT\\PATH\\HERE.obj"
  position: [0, 0, 0]
  orientation: [0, 0, 0]
  color: [255, 255, 255]
  scale: 1
  textures: {}
  animations: []
```

Here is an example that creates a blue quadcopter.

```yaml
model_configs:
  model: ".\\assets\\Agents\\Generic Quadcopter Drone.obj"
  position: [-0, 10, 50]
  orientation:
  color: [0, 0, 255]
  scale: 0.1
```

If the value for a key is missing, or the attribute line is missing entirely, the attribute will revert to default settings. Parent objects will override or modify the behaviors of their children.

## Using An Object's Model

You can always manipulate the internal Panda3D node. Getters and setters have not been built yet, so that's a future problem.

The graph tree for renderable object groups the entire object's node under a single node, the object's `object_node`. Parent all related nodes for that object. For example, the model of the object has it's own node, `model_node`, and parented to the object's `object_node`. Node paths are provided for both the `object_node` in `object_node_path` and `model_node` in `model_node_path`. The parent node of the object is also an attribute of the object, found in `parent_node_path`. When possible, please create and use getters/setters for these objects instead of using the internals.
