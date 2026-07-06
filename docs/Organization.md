# Organization

The repository is organized into overall general buckets.

The program hook is sensor_selection_simulator.py, which handles all the command line arguments, configuration files, and user inputs. sim/world is the top-most file that runs Panda3D, pybullet, and generates the simulation world.

## Loaders

To initialize, the `World` class calls its loaders, each of which instantiate world's attributes and manage their own domain. These can be found in the loader class.

- `EnvironmentLoader` - Loads terran and attaches related classes to world
- `ObjectLoader` - Loads all objects. Manage the objects from here
- `AgentLoader` - Work in progress

More things such as paralellization and optimization frameworks coming soon.
