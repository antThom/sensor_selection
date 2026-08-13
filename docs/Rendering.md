# Rendering

The simulation window and rendering are created by Panda3D, a game engine written in C++ that contains a python wrapper. The physics engine is provided by PyBullet. All renderable objects in `sensor_selection` inherit from the `RenderableObject` class, which can be found in `sim/rendering/object.py`. This class provides a standardized API for controlling an object in panda3d. In order to use and create these objects, use it in conjunction with `simulation_manager`.As part of an effort to modulate the systems of the simulation, all of the rendering instantiation, handling, and utilities are managed by the `SimulatorManager` class.

## `SimulationManager`

There is a single instance of SimulationManager in the entire codebase. The way you call this object depends on the context. It should probably be a global, but those always spell bad ideas.

```python
# Sometimes it appears like this
world.simulation_manger

# In Loaders, it will appear like this
self.world.simulation_manager

# Often it is abbreviated for shorter calls
sim_man = self.world.simulation_manager
sim_man.generate_simulation_node(foo, bar)
sim_man.configure_sim_model(fizz, buzz=fizzbuzz)
```

## Rendering an Object

In order to make an object render, the object needs to do a few things.

1. It's class needs to inherit from `RenderableObject`. All sensors and agents already inherit from this class.
2. Set the configurations
3. Generate the model.
4. If the object has a parent, parent the child to the parent
5. Configure the model
6. Make it render

These steps need to occur in some fashion in any order.

```python

```

To add models to an object, call either `generate_simulation_node()` or `generate_simulation_agent()`.
If you want to parent a model to another, use `parent_object_models()`.

```python

robo_dog = Agent()
stick = Agent() # pretend the stick has sensors on it
sim_man = world.simulation_manager

# Nodes cannot have animations 
# Actors can have animations
sim_man.generate_simulation_actor(robo_dog, model="assets/robot_dog.egg")
sim_man.generate_simulation_node(stick, model="assets/stick.egg")

sim_man.configure_sim_model(stick, color="brown")
sim_man.configure_sim_model(robo_dog, scale=5, animation={"Walk": "walk_dog.egg"})

# Parenting the stick to the dog moves the stick with the dog
sim_man.parent_object_models(stick, robo_dog)

```

## Configuring Models

If you want to apply any configurations such as position and color, use `configure_sim_model()`. It will take any group of key value pairs in a dictionary, as well as any nested pairs in a configuration or similar. `configure_sim_model` takes any number of key word arguments, so you can be quite messy with configurations.

```python
# Change the color of the model
configure_sim_model(tree, color="green")

# Change Position
configure_sim_model(agent,position=[1,2,3])

# Set Scale of model
configure_sim_model(drone, scale=0.3)

```

Internally implemented to all of the renderable classes are the attributes to make it renderable. If you ever wish to access the simulation and visual parts of the object, call it with `<object>.model`.

Here are the available attributes that are configurable to render.

| Name | attribute |
| ---- | --- |
| Position | object.position |
| Scale | object.scale |
| Orientation | object.orientation |
| Color | object.color |

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
