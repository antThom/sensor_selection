# Agents

Agents are the moving parts of the simulation, transporting around sensors that inform the agent's decision on where and how to move. From the logic that they gather from sensors, they make decisions about their actions. There are still in a work in progress.

## Usage

These have zero functionality right now. They sit around and do nothing.

You can find all agents in `world.agent_list`. `AgentLoader` loads up all the agents from their configuration file. Sensors belonging to the agent can be found in the `sensor_list` attribute. The model of the agent can be found as the model attribute.

## Configurations

Agents are created in the simulation through configuration files. To register agents into the scene, they need to be put into the scene's agent list. The configuration path outlines all the data required to create an agent, but you can redefine attributes in the scene file.

```yaml
agents:
    name_of_agent:
        config_path: "config\\path\\to\\agent\\template.yaml"
        id: 1234
        # <other tags are listed here to override template
    example_agent:
        config_path: "config\\another\\path\\yaml"
        id: 3456
```


The agent configuration file is more through, containing the model configuration and objects attached and related to the sensor. The general layout for the agent configuration is found below.

```yaml
name:
id:

settings:
  velocity:
  angular_rates:
  mass:
  agent_id:
  thermal:
  tf:
  file_path:
  max_vel:
  
sensors: 
    SensorName:
        type: "an enumerated class"
        config_path: "config\\sensor\\path"
        # Overriding settings here
    example_sensor:
        type: "IRSensor"
        config_path: "config\\sensors\\IrCamera.yaml"

model_configs:
  model: ".\\assets\\Agents\\Generic Quadcopter Drone.obj"
  position: [-0, 10, 50]
  orientation:
  color: [0, 0, 255]
  scale: 0.1
```

Attributes left blank are left to their defaults. They are displayed here in this page.
Generally, each attribute should have their own unique name. Internally, the simulation matches attributes that matches the names of the attributes inside and drops others that do not apply, parsing from top to bottom. The order technically does not matter, but attempt to format the agent configurations in this format.

Inside the agent configuration are where sensors, animations, and agent model are defined. In the setting subcategory goes attributes modeling the properties and state of the agent. To add sensors, list the type and path to the configuration file of the sensor, similar to how agents are defined.

```yaml
sensors: 
    SensorName:
        type: "an enumerated class"
        config_path: "config\\sensor\\path"
        # Overriding settings here
    example_sensor:
        type: "IRSensor"
        config_path: "config\\sensors\\IrCamera.yaml"

```

Like all renderable objects, agents have a model configuration setting.

```yaml
model_configs:
  model: ".\\assets\\Agents\\Generic Quadcopter Drone.obj"
  position: [-0, 10, 50]
  orientation:
  color: [0, 0, 255]
  scale: 0.1
```

Remember that you can always redefine attributes in higher, more general configuration files that reapply.


## Attributes

Only a few attributes are required in the creation of an agent. If none are stated, the configuration will fall onto their default value.

### Required

| Name | Location | Type |
| ---- | ----- | ----- |
| Name | Scene Config | String |
| Config | Scene Config | Path STring |
| Id | Scene Config | String |

Note that having sensors is optional. If you do, ensure that you have the type and configuration file when defining them.

### Optional

See the Agent class definition for all available attributes in `sim/agent/agent.py`.

## Contributing

When adding attributes to agent classes, ensure that they are added to this page. Additionally, ensure that all attributes added have their own unique attribute name. If a superclass defines it, use that attribute by that name or create a new unique attribute name so the superclass attributes are not impacted.