import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

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

    def test_agent_owns_attached_ir_camera(self):
        agent = Agent(thermal_manager=self.manager)
        camera = IRCamera(self.manager)

        agent.add_sensor(camera, "ir_camera")

        self.assertIs(camera.agent, agent)
        self.assertIs(agent.get_sensor("ir_camera"), camera)
        self.assertIn(camera, agent.sensor_list)


if __name__ == "__main__":
    unittest.main()
