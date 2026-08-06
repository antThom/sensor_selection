# Sensor Selection Simulator

A Panda3D simulation for evaluating autonomous agents and sensor payloads under
controlled environmental and thermal conditions. The current scene supports
agent-mounted RGB and long-wave infrared cameras, a geometry-aware thermal
render pass, interactive drone movement, and repeatable thermal validation.

## Features

- Configurable Panda3D terrain, vegetation, atmosphere, sun, and time of day
- Multiple rendered agents with shared thermal-body behavior
- Agent-mounted RGB and radiometric IR cameras
- GPU thermal visualization with emissivity, reflected radiance, atmospheric
  attenuation, solar exposure, material variation, NETD noise, and palette AGC
- Ironbow, white-hot, and black-hot IR palettes
- Live kelvin/Celsius temperature legend in IR mode
- Deterministic thermal sanity checks with CSV and graph output
- Keyboard movement and camera switching

## Windows setup

Python 3.12 is recommended for the pinned scientific and rendering packages.

```powershell
py -3.12 -m venv sim_env
.\sim_env\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Run the default mountain scene:

```powershell
python sensor_selection_simulator.py --config config\scene\mountain_range\scene.yaml
```

Run validation:

```powershell
python -m pytest -q
python sim\Environment\Thermal\thermal_sanity_check.py
```

## Controls

| Key | Action |
| --- | --- |
| `W` / `S` | Move the controlled drone forward/backward |
| `A` / `D` | Move left/right |
| `Space` / `Shift` | Ascend/descend |
| Arrow keys | Change heading and pitch |
| `C` / `X` | Cycle forward/backward through cameras |
| `Z` | Save the composited camera view to `logs/` |

The default mountain scene includes a sensor-equipped drone and a smaller
sensorless target drone positioned in its field of view.

## Documentation

- [Architecture](docs/Architecture.md)
- [Thermal simulation and IR rendering](docs/ThermalSimulation.md)
- [Integration history and conflict resolution](docs/IntegrationNotes.md)
- [Agents](docs/Agents.md)
- [Environment](docs/Environment.md)
- [Rendering](docs/Rendering.md)
- [Sensors](docs/Sensors.md)

## Project status

The thermal solver currently models one bulk temperature per registered object.
The IR shader adds performant surface-scale visual variation; it is not a
finite-element heat-transfer or wavelength-resolved detector model. See the
thermal documentation for equations, assumptions, and interpretation limits.
