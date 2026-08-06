# Simulator Architecture

## Startup and ownership

`sensor_selection_simulator.py` constructs `sim.world.WORLD`, the top-level
Panda3D `ShowBase`. `WORLD` owns every system with application lifetime:

1. `ThermalManager` stores material properties and registered thermal objects.
2. `BulletWorld` provides collision/physics geometry used by the scene.
3. `EnvironmentLoader` creates terrain and atmosphere.
4. `ObjectLoader` creates vegetation and retains thermal-renderable node paths.
5. `SensorLoader` constructs sensor types and allocates camera display regions.
6. `AgentLoader` creates agents, attaches thermal bodies, models, and sensors.
7. `CameraControls` and `DroneControls` register interactive inputs.

The loader order is intentional. Cameras need the environment and owning agent;
the IR camera then refreshes its renderable registry after all agents exist.

## Configuration flow

Scene YAML references agent templates, and agent templates reference sensor YAML.
`set_attr_from_configuration` recursively flattens configuration sections and
applies only known, non-null attributes. Scene-level values override template
values because they are applied second.

Relevant configuration roots:

- `config/scene/`: terrain, atmosphere, objects, and agent instances
- `config/agent/`: agent model, transform, and sensor payload
- `config/sensors/`: detector, optics, radiometry, output, and mount parameters
- `config/terrain/`: reusable terrain definitions

## Render object lifecycle

`RenderableObject` owns configuration values and Panda3D node references.
`SimulationManager` performs three explicit stages:

1. Create a transform root and optionally load visible model geometry.
2. Apply position, orientation, scale, and color.
3. Parent the root to another object or the world render tree.

Sensors without a visible housing still receive a transform anchor. Mounted
cameras are parented to their agent, so motion automatically propagates.

## Agent lifecycle

`Agent` inherits from `ThermalBody` and `RenderableObject`. `AgentLoader` applies
configuration, attaches one managed thermal body, renders the model, creates its
sensors, and finally adds the complete agent to `world.agent_list`.

The controlled drone updates both Panda3D and simulation state each frame:

- Scene-node position and heading/pitch
- Agent position, orientation, and velocity arrays
- Thermal fallback position
- Child camera transforms through scene-graph parenting

## Camera lifecycle

`SensorLoader` supports EO/RGB, IR, and dummy sensors. Camera setup creates a
Panda3D camera, resolves either `mount_position`/`mount_hpr` or PR #29's
`camera_offset`/`camera_angle` compatibility names, configures a perspective
lens, and registers an inactive display region.

Only the selected sensor display region is active. Camera-specific overlays,
including the IR temperature legend, follow the same activation lifecycle.
