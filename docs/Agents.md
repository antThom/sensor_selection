# Agents

Agents are the moving parts of the simulation, transporting around sensors that inform the agent's decision on where and how to move. From the logic that they gather from sensors, they make decisions about their actions. There are still in a work in progress.

## Usage

You can find all agents in `world.agent_list`. `AgentLoader` loads up all the agents from their configuration file. Sensors belonging to the agent can be found in the `sensor_list` attribute. The model of the agent can be found as the model attribute.

Below is an example demonstrating the functions available:

```python
from sim.agent.agent import Agent
# Create with construture with default configs
drone = Agent(thermal_manager=world.thermal_manager)

drone.id = 5
drone.name = "Quadcopter"
drone.tf = 0.001
drone.team = "BLUE"

# Read-only attributes
print(drone.sensor_list)
print(drone.model_list)

drone.add_model("assets/agent/Drone.obj")

from sim.sensors.dummy_sensor import DummySensor

sensor = DummySensor()
drone.add_sensor(sensor, "dummy")
drone.get_sensor("dummy")
```

Only a few attributes are required in the creation of an agent. If none are stated, the configuration will fall onto their default value. It is not ideal to go line by line settings all the attributes of an agent. Instead, use the `AgentBuilder` to create agents


Note that having sensors is optional. If you do, ensure that you have the type and configuration file when defining them.

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

## Creating an Agent

Even though the `AgentLoader` manages all of the agents, a separate class handles the creation of agents: `AgentBuilder`. To create an agent, make an instance of an agent builder. You can make multiple calls, or you can chain them in a single line.

```python
from sim.loaders.agent_loader import AgentBuilder
from sim.environment.thermal.thermal_manager import ThermalManager
# You can't call ThermalManager(), but we pretend we can for this example
builder = AgentBuilder(world)

# Required options only
# Will automatically make it render
basic_agent = builder.set_thermal_manager(ThermalModel()).build()
basic_agent.attach_to_world() # This makes it findable by World and resets builder.

# Using External Configurations
configuration_dictionary, sensor_dictionary = dict()
builder.set_thermal_manager(ThermalModel())
builder.with_configurations(configuration_dictionary)
builder.with_sensor_configurations(sensor_dictionary)
external_agent = builder.build().attach_to_world()

# Using all options

crazy_agent = builder.id().name().team() \
    .render_agent(True) \
    .with_configurations("config\\number\\one", priority=10) \
    .with_configurations("config\\number\\two") \
    .with_sensor_list("sensor\\configurations") \
    .add_sensor("sensor_name", "configuration", priority=5) \
    .set_thermal_manager(ThermalManager()) \
    .attach_thermal(body_id=31, source=model)

# Turn off automatic rendering
# Enables you to manually set settings.
special_agent = builder.render_agent(False).set_thermal_manager(ThermalModel()).build()
builder.attach_to_world()

from sim.rendering.renderable_object import RenderableObjectBuilder
render_builder = RenderableObjectBuilder(world)

# Will render the object
render_builder.with_object(special_agent).config_from_object(special_agent).build() 

```

If you wish, you can manually set the rendering settings by setting the attribute `rendering_agent` to `False`.

## Contributing

When adding attributes to agent classes, ensure that they are added to this page. Additionally, ensure that all attributes added have their own unique attribute name. If a superclass defines it, use that attribute by that name or create a new unique attribute name so the superclass attributes are not impacted.