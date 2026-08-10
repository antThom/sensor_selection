from __future__ import annotations

try:
    import pybullet as p
except ImportError:
    p = None

from sim.environment.ThermalObject import ThermalObject


class ThermalStaticObject(ThermalObject):
    """PyBullet thermal body with an optional Ursina visual."""

    def __init__(self, body_id, *, visual=None, **thermal):
        self.visual = visual
        super().__init__(body_id, **thermal)
        self.sync_visual()

    @classmethod
    def box(
        cls,
        size=(1.0, 1.0, 1.0),
        position=(0.0, 0.0, 0.5),
        *,
        client_id=None,
        visual=None,
        thermal_mass=None,
        **thermal,
    ):
        if p is None:
            raise RuntimeError("pybullet is required to create a thermal static body")
        options = {} if client_id is None else {"physicsClientId": client_id}
        half_extents = [float(side) / 2.0 for side in size]
        shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            **options,
        )
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            basePosition=[float(value) for value in position],
            **options,
        )
        return cls(
            body,
            client_id=client_id,
            visual=visual,
            dimensions=size,
            position=position,
            mass=thermal_mass,
            **thermal,
        )

    def sync_visual(self):
        if self.visual is None:
            return
        x, y, z = self.position()
        # PyBullet is Z-up while Ursina/Panda3D is Y-up.
        self.visual.position = (x, z, y)
