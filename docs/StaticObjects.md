# Static Objects

Objects physically modify the environment: they may obscure vision, change heat signatures, and create sound as an agent moves through them. Objects are the least implemented point of the repository. They are not yet configurable either. Development and documentation of changes and development will be much appreciated.

## Object Loader

Currently, the object loader is hard coded to load trees in random places. Still a work in progress.

## Models

Attached to objects are models that correlate to a specific type of physics that a sensor may encounter. See the model pages for further explanation.

For a all objects, they all carry the following models for different types of sensors

| Model | Type |
| ----- | ---- |
| Panda3D | Simulates RGB |
| ThermalObject | Heat and IR |
| Sound | Stereo and Microphone |

## Configurations

Objects are currently unconfigurable. Please implement and document later
