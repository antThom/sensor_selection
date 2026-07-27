# Rendering

The simulation window and rendering are created by Panda3D, a game engine written in C++ that contains a python wrapper. The physics engine is provided by PyBullet. As part of an effort to modulate the systems of the simulation, all of the rendering instantiation, handling, and utilities are managed by the `SimulatorManager` class

## `SimulationManager`

There is a single instance of SimulationManager in the entire codebase. The way you call this object depends on the context. It should probably be a global, but those always spell bad idea.

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

## Adding to the Simulation Rendering

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

If you want to apply any configurations such as position and color, use `configure_sim_model()`. It will take any group of key value pairs in a dictionary, as well as any nested pairs in a configuration or similar. `configure_sim_model` takes any number of key word arguements, so you can be quite messy with configurations.

```python
# Change the color of the model
configure_sim_model(tree, color="green")

# Change Positon
configure_sim_model(agent,position=[1,2,3])

# Set Scale of model
configure_sim_model(drone, scale=0.3)

```


Internally implemented to all of the renderable classes are the attributes to make it renderable. If you ever wish to access the simulation and visual parts of the object, call it with `<object>.model`.

Here are the available attributes that are configureable to render 

| Name | attribute | 
| ----  | --- |
| Position | object.position |
| Scale | object.scale |

If you wish to implement more of these functions, search out the [Panda3D documentation](https://docs.panda3d.org/1.10/python/index)