from __future__ import annotations

from typing import Optional


class ThermalBody:
    """Mixin that gives a rendered object a managed thermal component."""

    def __init__(self, thermal_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.thermal_manager = thermal_manager
        self.thermal_object = None
        self.thermal_objects = {}
        self._thermal_body_id: Optional[int] = None

    def _thermal_position(self):
        position = getattr(self, "position", (0.0, 0.0, 0.0))
        if hasattr(position, "flatten"):
            position = position.flatten().tolist()
        values = list(position)
        if len(values) < 3:
            values.extend([0.0] * (3 - len(values)))
        return tuple(float(value) for value in values[:3])

    def attach_thermal(
        self,
        body_id=None,
        source="generic",
        *,
        thermal_manager=None,
        per_link=False,
    ):
        """Attach this object to the shared thermal system."""
        if thermal_manager is not None:
            self.thermal_manager = thermal_manager
        if self.thermal_manager is None:
            raise RuntimeError("A ThermalManager is required before attaching a body")

        if self._thermal_body_id is not None:
            self.thermal_manager.unregister_body(self._thermal_body_id)

        resolved_body_id = (
            id(self) if body_id is None else body_id
        )  # BUG: Replace id(self) with a different thing. id() too large to be used with pybullet
        self._thermal_body_id = resolved_body_id
        self.thermal_object = self.thermal_manager.register_body(
            resolved_body_id,
            source,
            per_link=per_link,
            position=self._thermal_position(),
        )
        self.thermal_objects = self.thermal_manager.get_body_objects(resolved_body_id)
        return self.thermal_object

    def sync_thermal_position(self):
        """Update fallback poses for objects not represented in PyBullet."""
        position = self._thermal_position()
        for thermal_object in self.thermal_objects.values():
            thermal_object.set_fallback_position(position)

    def detach_thermal(self):
        if self.thermal_manager is not None and self._thermal_body_id is not None:
            self.thermal_manager.unregister_body(self._thermal_body_id)
        self._thermal_body_id = None
        self.thermal_object = None
        self.thermal_objects = {}

    @property
    def temperature(self):
        return None if self.thermal_object is None else self.thermal_object.temperature
