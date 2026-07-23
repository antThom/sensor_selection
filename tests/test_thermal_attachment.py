import unittest

from sim.Environment.Thermal.thermal_manager import ThermalManager
from sim.Environment.ThermalObject import SceneItem, ThermalBody, ThermalObject


class ThermalAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.manager = ThermalManager(
            time_of_day=12,
            ambient_K=293.0,
            T_sky=260.0,
        )

    def test_scene_item_attaches_a_thermal_object(self):
        item = SceneItem(101, "tree.urdf", self.manager)

        self.assertIsInstance(item, ThermalBody)
        self.assertIsInstance(item.thermal_object, ThermalObject)
        self.assertIs(item.thermal_object, self.manager.objects[(101, -1)])
        self.assertEqual(item.temperature, item.thermal_object.temperature)

    def test_reattaching_replaces_the_old_body_registration(self):
        body = ThermalBody(self.manager)
        body.attach_thermal(101, "tree.urdf")
        body.attach_thermal(202, "cloud.urdf")

        self.assertNotIn((101, -1), self.manager.objects)
        self.assertIn((202, -1), self.manager.objects)
        self.assertEqual(body.thermal_object.body_id, 202)

    def test_detaching_removes_all_body_links(self):
        body = ThermalBody(self.manager)
        body.attach_thermal(101, "tree.urdf")
        self.manager.add_object(101, 0)

        body.detach_thermal()

        self.assertEqual(self.manager.get_body_objects(101), {})
        self.assertIsNone(body.thermal_object)


if __name__ == "__main__":
    unittest.main()
