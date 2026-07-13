# Introduction

`sensor_selection` is a simulator created by Dr. Anthony Thompson, a scientist at the Johns Hopkins Applied Physics Lab, for the research of sensor selection algorithms, particularly in the subject of tracking moving agents and objects.

`sensor_selection` is built around around simulating and modeling the environment in different states and effects and the different awareness of sensors in these conditions. As agents move in the environment and the types of data availble to the sensors change, different algorithms become more useful than others. This simulation is designed as a sandbox for different environments with alterable conditions to place sensors in.

See the [contributing page](/CONTRIBUTING.md) for how to develop and contribute for this project.

## Running `sensor_selection`

The program hook is sensor_selection_simulator.py, which handles all the command line arguments, configuration files, and user inputs.

```bash
python sensor_selection_simuator.py --config <path/to/scene/config/>
```

sensor_selection_simulator.py calls the `World` class, the top-most file that runs Panda3D, pybullet, and generates the simulation world. How the `World` does so is through the simulator's many loaders.

## Loaders

To initialize, the `World` class calls its loaders, each of which instantiate world's attributes and manage their own domain. These can be found in the loader module.

- `EnvironmentLoader` - Loads terran and attaches related classes to world
- `ObjectLoader` - Loads all objects. Manage the objects from here
- `AgentLoader` - Builds agents
- `SensorLoader` - Called by `AgentLoader` to create sensors
- `SimulationManager` - Not really a loader, abstracts away graphics implementation

Each loader and their configurations are described in their own articles as they are outside of the scope of this introduction:

- [Graphics](/docs/Rendering.md)
- [Agents](/docs/Agents.md)
- [Sensors](/docs/Sensors.md)
- [Environment](/docs/Environment.md)
- [Static Objects](/docs/StaticObjects.md)

These loaders create and and set the objects created as described in the configurations into attributes of the world class.

- `world.agent_list` - List containing all agent objects
- `world.object_list` - List containing all static object lists.
- `world.terrain` - The terrain of the simulation

The loaders take the `world` class as an argument so that they can share and pass along information at run time such as putting sensors on agents during runtime.

More things such as parallelization and optimization frameworks coming soon.

## Configuration

`sensor_selection` uses the format of YAML files to configure the simulation. All configurations can be found under the `/config` directory. The simulator takes in scene configuration files as arguments as a command line application.

Configuration files are needed for every type of component in the chain: sensors, agents, environment, and terrain. Template files are left in each folder for you to base on. If setting configurations are left blank in the configuration file (or even omitted), the simulation will set the attribute or object to their default value.

There are four types of configuration files: `agent`, `scene`, `sensors`, and `terrain`. These files have their own independent schema for configuring the simulation. Scene configurations configure the entire simulation, requiring the paths for agent and terrain configurations. Agent configuration calls the types of sensors it requires, taking the sensor's configuration files.

### Defining Multiple Times

You can change the setting of a value in a higher configuration file. This allows for the useage of the same configuration file as a template while changing aspects later in the configuration. Configurations lower down the hierarchy are applied first, and ones higher up the heiarchy are applied last. 


The hierarchy of the configuration files works as follows:

1. Scene
2. Terrain
3. Agents
4. Sensors


You can change the setting of a value in a higher configuration file. This allows for the useage of the same configuration file as a template while changing aspects later in the configuration.


For example, let's say you desire to specify an agent to have a certain id basing off of a template file. The template file, `config/agent/basic_agent.yaml`, could define the agent id as follows:

```yaml

settings:
  position: [0, 0, 0]
  velocity:
  orientation:
  angular_rates:
  mass:
  agent_id: 1234
  thermal:
  tf:
  file_path:
  max_vel:
  scale: 0.001
```

In the topmost file, you could redefine the attribute again, rewriting the original attribute in the template file in, for example, `config/scene/mountain_range/scene.yaml`.

```yaml
agents:
  basicAgent:
    name: "BasicAgent"
    config_path: "config\\agent\\basic_agent.yaml"
    id: 5678
    position: [900,900,900]
```

By default, an agent in its agent definition file could have a setting of 'agent_id: 1`. In the scene configuration, the agent id could be overwritten, allowing for changing configurations down the line.

### Paths

Paths are locations for the computer to find a file. Inside of configuration files, do not use the absolute path to the location of the file, make them a relative path from the top-most file of the repository.

If there are errors, they are most like to the issues displayed below:

| DO | DON'T |
| -------- | --------- |
| `./sim/loader/agent_loader.py` | `~/Programs/sim/loader/agent_loader.py` |
| `./assets/Agent/Generic Drone` | `"./assets/Agent/Generic Drone"` |
| `.\\assets\\Sounds\\quadcopter.mp3`| `.\\assets\\Sounds\\quadcopter.mp3` |

## Assets

Generally, all useable assets used in the repository are given their own folder for their own category, machine files and all. Even generation files and source code goes in there because they are attached to their generated models and files.

More things such as parallelization and optimization frameworks coming soon.
