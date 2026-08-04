import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from panda3d.core import Camera as PandaCamera
from panda3d.core import NodePath, PandaNode, ShaderAttrib

from sim.Agent.agent import Agent
from sim.Environment.Thermal.thermal_manager import ThermalManager
from sim.Sensors.Cameras.ir_camera import IRCamera
from sim.Sensors.sensor import SensorType
from sim.loaders.sensor_loader import SensorLoader

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config" / "sensors"


class IRCameraTests(unittest.TestCase):
    def setUp(self):
        self.manager = ThermalManager(time_of_day=12)

    def load_camera(self, filename):
        with (CONFIG_DIR / filename).open(encoding="utf-8") as stream:
            configuration = yaml.safe_load(stream)
        world = SimpleNamespace(thermal_model=self.manager)
        return SensorLoader(world).create_sensor("ir_camera", configuration)

    def test_general_configuration_contains_relevant_parameters(self):
        camera = self.load_camera("basic_ir_camera.yaml")

        self.assertEqual(camera.type, SensorType.IRCAMERA)
        self.assertEqual((camera.width, camera.height), (640, 512))
        self.assertEqual(camera.spectral_band_um, [8.0, 14.0])
        self.assertGreater(camera.frame_rate_hz, 0)
        self.assertGreater(camera.pixel_pitch_um, 0)
        self.assertGreaterEqual(camera.netd_K, 0)
        self.assertEqual(camera.display_temperature_range_K, [260.0, 330.0])
        self.assertGreaterEqual(camera.atmospheric_extinction_per_m, 0.0)
        self.assertTrue(camera.radiometric)
        camera.validate_parameters()

    def test_manufacturer_configurations_load(self):
        expected = {
            "flir_boson_640.yaml": ("20640AS50-6IARX", 12.0, 50.0),
            "flir_tau2_640.yaml": (
                "46640001X",
                17.0,
                45.0,
            ),
            "flir_hadron_640r.yaml": (
                "70640AS32-6PMRXX",
                12.0,
                32.0,
            ),
        }
        for filename, values in expected.items():
            with self.subTest(filename=filename):
                camera = self.load_camera(filename)
                self.assertEqual(camera.model_number, values[0])
                self.assertEqual(camera.pixel_pitch_um, values[1])
                self.assertEqual(camera.horizontal_fov_deg, values[2])
                self.assertEqual((camera.width, camera.height), (640, 512))

    def test_radiometric_conversion_and_palette(self):
        camera = IRCamera(self.manager)
        camera.temperature_range_K = [270.0, 330.0]
        camera.netd_K = 0.0
        temperatures = np.array([[270.0, 285.0], [300.0, 330.0]], dtype=np.float64)

        apparent = camera.apparent_temperature(
            temperatures,
            emissivity=1.0,
            atmospheric_transmission=1.0,
        )
        np.testing.assert_allclose(apparent, temperatures)

        image = camera.temperature_to_image(
            temperatures, palette="ironbow", add_noise=False
        )
        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertFalse(np.array_equal(image[0, 0], image[1, 1]))

    def test_noise_is_repeatable_for_a_configured_seed(self):
        camera = IRCamera(self.manager)
        temperatures = np.full((8, 8), 300.0)
        first = camera.temperature_to_image(temperatures)
        second = camera.temperature_to_image(temperatures)
        np.testing.assert_array_equal(first, second)

    def test_automatic_gain_control_uses_scene_temperature_window(self):
        camera = IRCamera(self.manager)

        display_range = camera._automatic_display_range([285.0, 294.0, 290.0])

        self.assertEqual(display_range, [274.5, 304.5])
        camera.automatic_gain_control = False
        self.assertEqual(
            camera._automatic_display_range([285.0, 294.0]),
            camera.display_temperature_range_K,
        )

    def test_agent_owns_attached_ir_camera(self):
        agent = Agent(thermal_manager=self.manager)
        camera = IRCamera(self.manager)

        agent.add_sensor(camera, "ir_camera")

        self.assertIs(camera.agent, agent)
        self.assertIs(agent.get_sensor("ir_camera"), camera)
        self.assertIn(camera, agent.sensor_list)

    def test_live_view_updates_gpu_input_from_current_temperature(self):
        camera = IRCamera(self.manager)
        scene_node = NodePath(PandaNode("thermal-test-object"))
        thermal_source = SimpleNamespace(temperature=270.0)
        camera.register_thermal_node(scene_node, thermal_source, emissivity=1.0)

        camera.refresh_live_thermal_colors()
        first = scene_node.getShaderInput("thermal_object").getVector()
        self.assertEqual(first.x, 270.0)
        self.assertEqual(first.y, 1.0)

        thermal_source.temperature = 330.0
        camera.refresh_live_thermal_colors()
        second = scene_node.getShaderInput("thermal_object").getVector()
        self.assertEqual(second.x, 330.0)

    def test_thermal_render_state_is_owned_by_ir_camera_only(self):
        camera = IRCamera(self.manager)
        camera.camera_node = PandaCamera("test-ir-camera")
        rgb_camera = PandaCamera("test-rgb-camera")

        camera._build_thermal_render_state()

        shader_slot = ShaderAttrib.getClassSlot()
        self.assertTrue(camera.camera_node.getInitialState().hasAttrib(shader_slot))
        self.assertFalse(rgb_camera.getInitialState().hasAttrib(shader_slot))


if __name__ == "__main__":
    unittest.main()
