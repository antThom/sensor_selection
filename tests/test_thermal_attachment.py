import unittest

from sim.Environment.Thermal.thermal_manager import ThermalManager
from sim.Environment.ThermalObject import ThermalBody, ThermalObject


class ExampleThermalBody(ThermalBody):
    def __init__(self, manager):
        self.position = [1.0, 2.0, 3.0]
        super().__init__(thermal_manager=manager)


class ThermalAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.manager = ThermalManager(
            time_of_day=12,
            ambient_K=293.0,
            T_sky=260.0,
        )

    def test_body_attaches_a_thermal_object(self):
        body = ExampleThermalBody(self.manager)
        thermal_object = body.attach_thermal(101, "agent.urdf")

        self.assertIsInstance(thermal_object, ThermalObject)
        self.assertIs(body.thermal_object, self.manager.objects[(101, -1)])
        self.assertEqual(body.temperature, thermal_object.temperature)
        self.assertEqual(thermal_object.position(), (1.0, 2.0, 3.0))

    def test_reattaching_replaces_the_old_registration(self):
        body = ExampleThermalBody(self.manager)
        body.attach_thermal(101, "agent.urdf")
        body.attach_thermal(202, "agent.urdf")

        self.assertNotIn((101, -1), self.manager.objects)
        self.assertIn((202, -1), self.manager.objects)

    def test_detaching_removes_all_body_links(self):
        body = ExampleThermalBody(self.manager)
        body.attach_thermal(101, "agent.urdf")
        self.manager.add_object(101, 0)

        body.detach_thermal()

        self.assertEqual(self.manager.get_body_objects(101), {})
        self.assertIsNone(body.thermal_object)


if __name__ == "__main__":
    unittest.main()
