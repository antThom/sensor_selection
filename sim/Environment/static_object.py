from __future__ import annotations

try:
    import pybullet as p
except ImportError:
    p = None

from sim.Environment.ThermalObject import ThermalObject


class StaticObject(ThermalObject):
    def __init__(self, body_id, *, visual=None, **thermal):
        self.visual = visual
        super().__init__(body_id, **thermal)
        self.syncvisual()

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
            raise RuntimeError("pybullet is required to create a StaticObject body")
        opts = {} if client_id is None else {"physicsClientId": client_id}
        half = [float(side) / 2.0 for side in size]
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, **opts)
        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            basePosition=[float(x) for x in position],
            **opts,
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

    def syncvisual(self):
        if self.visual is None:
            return
        x, y, z = self.position()
        # pybullet is z-up and ursina/panda3d is y-up
        self.visual.position = (x, z, y)


STATIC_OBJECT = StaticObject
