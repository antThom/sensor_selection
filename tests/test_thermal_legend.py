from pathlib import Path

import yaml

from sim.Sensors.Cameras.thermal_legend import ThermalLegend

ROOT = Path(__file__).resolve().parents[1]


def test_temperature_legend_formats_kelvin_and_celsius():
    assert ThermalLegend.format_temperature(304.65) == "304.6 K\n31.5 C"
    assert ThermalLegend.format_temperature(273.15) == "273.1 K\n0.0 C"


def test_mountain_scene_contains_sensorless_target_drone():
    with (ROOT / "config" / "scene" / "mountain_range" / "scene.yaml").open(
        encoding="utf-8"
    ) as stream:
        scene = yaml.safe_load(stream)
    target_reference = scene["agents"]["target_drone"]
    with (ROOT / target_reference["config_path"]).open(encoding="utf-8") as stream:
        target = yaml.safe_load(stream)

    assert target_reference["agent_id"] == 4
    assert target["sensors"] == {}
    assert target["model_configs"]["position"] == [0.0, 45.0, 40.0]
