# Environment

The environment has many configuration and settings to simulate an environment. The overall settings to change are the classes `Atmosphere`, `Sun`, `Sky`, and `Terrain`. All environmental objects inherit from the `STATIC_OBJECT` for rendering and utility functions.

## Terrain

`Terrain` creates the mesh that acts as the floor to the simulation.

```yaml
terrain:
    obj_path: "assets\\path\\to\\object"
    obj_scale: ""
    obj_pos: "center" # Leave it as 'center'
    texture_path: ""
    texture_scale: [1, 1]
```

If changing the scale is desired, use a list of [x, y, z] for your scales for the model and [x, y] for your texture.

## Atmosphere

The atmosphere section implements the features of the sky, clouds, and time. Colors are in RGB format from the scale of 0 to 1.

```yaml
atmosphere:
  time:
    day_length: 24 # In Seconds
    time: "08:00:00" # Start the sun at this time
  sky:
    obj_path: "models/misc/sphere"
    obj_scale: 10000
    obj_pos: "center"
    color_day: [0.53, 0.81, 0.92, 1.0]
    color_twilight: [0.95, 0.45, 0.20, 1.0]
    color_night: [0.1, 0.1, 0.1, 1.0]
    light_source:
      obj_path: "models/misc/sphere"
      obj_scale: 120
      distance: 5000
      color_temp: 5000
  clouds:
    obj_path: ""
    obj_scale: [1,1,1]
    obj_pos: "random"
    texture_path: ""
    texture_scale: []
```

## Dynamic Features

Environmental features that move are listed under the `dynamic` category list, which include the sun and clouds objects. They have the same attributes and settings.

```yaml
dynamic:
  sun:
    obj_path: ""
    obj_scale: [1,1,1]
    obj_number: 1
    obj_orientation: [0,0,0]
    obj_pos: [0,90]
    obj_vel: []
    obj_vec_space: "cylinder"
    texture_path: ""
    color: ""
  clouds:
    obj_path: ["",""]
    obj_scale: [1,1,1]
    obj_number: 40
    obj_orientation: [0,0,0]
    obj_pos: "random"
    obj_vel: [10,0,0]
    obj_vec_space: "euclidean"
    texture_path: ["",""]
    color: ["",""]
```