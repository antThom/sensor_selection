# Agents

Agents are the moving parts of the simulation, transporting around sensors that inform the agent's decesion on where and how to move. From the logic that they gather from sensors, they make decesions about their actions. There are still in a work in progress.

# Useage

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

Remember that you can always redefine attributes in higher, more general configuration files that reapply.

## Attributes

Only a few attributes are required in the creation of an agent. If none are stated, the configuration will fall onto their default value.

### Required

| Name | Location | Type |
| ---- | ----- | ----- |
| Name | Scene Config | String |
| Config | Scene Config | Path STring |
| Id | Scene Config | String |
| Model | Agent Config | Path String |

Note that having sensors is optional. If you do, ensure that you have the type and configuration file when defining them.

### Optional

See the Agent class definition for all available attributes in `sim/agent/agent.py`.