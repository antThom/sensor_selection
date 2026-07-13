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

## Configurations

General configuration for all sensors are required as follows

(*there's no configurations yet here*)

See the API documentation or the object file for their full attribute list.