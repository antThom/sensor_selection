# Controls

## Mouse Controls

The default view when the simulation is started is the default camera. This camera is controllable through using a mouse. If you are not using one, you will not have access to all of the controls that are available.

| Button | Action |
| ---- | ---- |
| Left Click | Pan |
| Right Click | Rotate |
| Scroll | Zoon In/Out |
| Press Down on Scroll Wheel | Roll |

## Camera Controls

If camera sensors are created and attached to an agent, you can switch to that camera view and take screenshots of what that view is seeing. Mouse controls do not move what the camera sensor is seeing.

| Key | Action |
| ------ | ------ |
| C | Go to Next Camera |
| X | Go to Previous Camera |
| Z | Capture Current Screen |

## Screen Captures

You can press `z` to capture a screenshot of the current camera that is displayed on the simulation.

You can find screenshots in the `logs/` directory saved to png files. When `z` is pressed, Panda3D saves the last rendered frame as a PMI file to a buffer located in `/logs/buffer/buffer.ppm`. A third party library (Pillow, currently) exports that image into a png in the `/logs` directory.
