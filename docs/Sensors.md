# Sensors

Sensors are the center point of the system. They interface with models of objects, interact with meshes, and feed data to algorithms for their usage. Sensors are still in a work in progress, but they have some logic already programmed. This page will further expand as sensors are more developed.

## Usage

Sensors can be found in their parent Agent owner in their `sensor_list` attribute. Unboxed by the `sensor_loader` class, they are configured as directed by configurations. Default values are always given if not specified. They are rendered as a child to their parent agent, given their own model, and collect data independent of their agent parent.

## Configuration

Stuff to write about

- put sensor in sensor lid
- create a sensor config file
- state the type, model configurations, and other configurations related to type
- For each type, make sure to add related configs

Configurations need to be in a certain order, otherwise, what you intent to be applied to something else will be applied to your sensor. Depending on what type of sensor, there will be additional sections of settings to be added to the configuration file.

This is an example for an EO camera:

```yaml
name: 
type: "EOCAMERA"
id:

input:
output:
encode: "rgb"

tf:

camera_settings:
  fov:
  width:
  height:
  fx:
  fy:
  c:
  forward_angle:
  model: "pinhole"
  k1:
  k2:
  k3:
  k4:
  near:
  far:
  forward_angle:

model_configs:
  model: ".\\assets\\Sensors\\SLR_Camera\\10124_SLR_Camera_SG_V1_Iteration2.obj"
  position: [-0, 0, 0]
  orientation: [90, 45, 90]
  color: [0, 0, 100]
  scale: 1
```

As this is an object that is being rendered, it inherits the attributes for a renderable object. Read the [rendering page](/docs/Rendering.md) on how to do model_configurations.
Remember that these configurations can be overwritten in a higher configuration file in the hierarchy.

## Sensor Superclass

The `Sensor` abstract class is the blueprint for all sensors, providing the utilities for using and operating sensors. All sensors have the following attributes:

- Attributes from Renderable Object
  - Model
  - Parent node
  - Object Node (where other nodes attach to)

All types are identified in their enumerated type `Sensor Type`

## Types

Different types of sensor are programmed to interact with the environment and objects in different ways. Each has their own logic and systems that they interact with.

### Cameras

Sensors that collect light or radiation to form an image are a part of the `Camera` superclass. They're... a camera! They interact with Panda3D to create images and are the most computationally expensive to calculate.

All cameras are availabe to be viewed and switched with the default camera view. Objects that inherit from this class are provided setup for POV, view, and distortion of lens and light.

They are provided the following attributes.

- camera_node
- camera_node_path
- camera_offset
- camera_angle

Internally, a camera node is created and attached to the sensor's object node. This camera node inherits all attributes that the sensor has.

They have the following configuration:

```yaml
settings:
  fov:
  width:
  height:
  fx:
  fy:
  c:
  forward_angle:
  model: "pinhole"
  k1:
  k2:
  k3:
  k4:
  near:
  far:
  forward_angle:
```

These configurations are used to create a Panda3D lens for the camera for the camera to use.

#### EO Cameras

EO cameras, or Electro-Optical cameras, capture optical light to form an image. They are a basic RGB camera.

Use sensor type `"eo_camera"` to use these. The sensor type for an EO camera is `SensorType.EO_CAMERA`. This is the same as the default camera and uses the default Panda3D camera utilities and no special effects.

EO cameras does not have any additional configurations besides the default camera configurations.

#### IR

Infrared is used to find heat signatures of objects. `IRCamera` is currently the only sensor that uses this extensive system to do it's calculations.

### Sound

Sensors that relate to sound are the `microphone` and `RGBSCamera`. They need to be reimplemented, as the sound system is not built yet. Sound sensors derive from the `SoundSensor` Superclass and access PyBullet's sound system utilities.

## Sensor Loader

- Sensor loader is called by agent loader
- internally for each sensor, you need to create a new strategy to install the sensor

## Contributing

To create a sensor, you need to ensure that you modify the following files:

- sensor_loader
- Create your sensor

The sensor loader will be the factory class that actually creates the sensor. Lazy load the import statement inside the function to prevent circular or conflicting imports.
