from __future__ import annotations

from typing import Optional


class ThermalBody:
    """Base class for scene items backed by a managed thermal model."""

    def __init__(self, thermal_manager=None):
        self.thermal_manager = thermal_manager
        self.thermal_object = None
        self.thermal_objects = {}
        self._thermal_body_id: Optional[int] = None

    def attach_thermal(
        self,
        body_id: int,
        source: str,
        *,
        per_link: bool = False,
    ):
        """Attach the PyBullet body and expose its primary ThermalObject."""
        if self.thermal_manager is None:
            raise RuntimeError("A ThermalManager is required before attaching a body")

        if self._thermal_body_id is not None:
            self.thermal_manager.unregister_body(self._thermal_body_id)

        self._thermal_body_id = body_id
        self.thermal_object = self.thermal_manager.register_body(
            body_id,
            source,
            per_link=per_link,
        )
        self.thermal_objects = self.thermal_manager.get_body_objects(body_id)
        return self.thermal_object

    def detach_thermal(self) -> None:
        if self.thermal_manager is not None and self._thermal_body_id is not None:
            self.thermal_manager.unregister_body(self._thermal_body_id)
        self._thermal_body_id = None
        self.thermal_object = None
        self.thermal_objects = {}

    @property
    def temperature(self) -> Optional[float]:
        if self.thermal_object is None:
            return None
        return self.thermal_object.temperature


class SceneItem(ThermalBody):
    """A body placed in the scene and automatically registered for heat flow."""

    def __init__(
        self,
        body_id: int,
        source: str,
        thermal_manager,
        *,
        per_link: bool = False,
    ):
        super().__init__(thermal_manager)
        self.body_id = body_id
        self.source = str(source)
        self.attach_thermal(body_id, self.source, per_link=per_link)
