# Sensors

Sensors are the center point of the system. They interface with models of objects, interact with meshes, and feed data to algorithms for their usage. Sensors are still in a work in progress, but they have some logic already programmed. This page will further expand as sensors are more developed by the 

## Usage

Sensors can be found in their parent Agent owner in their `sensor_list` attribute. Unboxed by the `sensor_loader` class, they are configured as directed by configurations. Default values are always given if not specified. They are rendered as a child to their parent agent, given their own model, and collect data independent of their agent parent.

## Sensor Superclass

The `Sensor` abstract class is the blueprint for all sensors, providing the utilities for super


## Types

Different types of sensor are programmed to interact with the environment and objects in different ways. Each has their own logic and systems that they interact with.

### Cameras

Sensors that collect light or radiation to form an image are a part of the `Camera` superclass. They're... a camera! They interact with Panda3D to create images and are the most computationally expensive to calculate.

### Sound

Sensors that relate to sound are the `microphone` and `RGBSCamera`. They need to be reimplemented, as the sound system is not built yet. Sound sensors derive from the `SoundSensor` Superclass and access PyBullet's sound system utilities.

### IR

Infrared is used to find heat signatures of objects. `IRCamera` is currently the only sensor that uses this extensive system to do it's calculations.

`IRCamera` is an agent-mounted Panda3D camera and a radiometric image model.
The live scene camera follows the owning agent and can be selected with the
simulation's camera controls:

- Press `C` to move forward through the available camera views.
- Press `X` to move backward through the available camera views.
- Press `Z` to save the currently selected view.

An IR camera is attached to an agent through the agent YAML:

```yaml
sensors:
  Boson640:
    type: "ir_camera"
    config_path: "config\\sensors\\flir_boson_640.yaml"
```

The configuration files include detector resolution, frame rate, pixel pitch,
spectral band, NETD, radiometric range, lens field of view, clipping planes,
emissivity, atmospheric transmission, reflected temperature, output palette,
and the camera-to-agent mount pose.

Available examples:

- `config/sensors/basic_ir_camera.yaml`
- `config/sensors/flir_boson_640.yaml`
- `config/sensors/flir_tau2_640.yaml`
- `config/sensors/flir_hadron_640r.yaml`

The live Panda3D view represents the IR sensor viewpoint. For radiometric
processing, `temperature_to_image()` accepts a two-dimensional kelvin array and
returns an 8-bit RGB image using white-hot, black-hot, or ironbow palettes.

## Configurations

General configuration for all sensors are required as follows

(*there's no configurations yet here*)

See the API documentation or the object file for their full attribute list.
