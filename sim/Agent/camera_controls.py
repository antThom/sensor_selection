"""Camera selection and screenshot controls for the Panda3D simulation."""

from datetime import datetime
from pathlib import Path

from direct.showbase.DirectObject import DirectObject
from panda3d.core import Filename, PNMImage


class CameraControls(DirectObject):
    """Cycle render cameras and capture the composited application window.

    Index zero is always Panda3D's default camera. Later entries are sensor
    cameras with independent display regions. Sensors may optionally implement
    ``set_overlay_visible`` for camera-specific UI such as the IR legend.
    """

    def __init__(self, world):
        super().__init__()
        self.world = world
        self.camera_index = 0

    def camera_list_forward(self) -> None:
        """Activate the next registered camera."""
        self.change_camera(self.camera_index + 1)

    def camera_list_back(self) -> None:
        """Activate the previous registered camera."""
        self.change_camera(self.camera_index - 1)

    def change_camera(self, index: int) -> None:
        """Activate ``index`` after wrapping it into the camera list."""
        if not self.world.camera_list:
            raise RuntimeError("the world has no registered cameras")

        old_index = self.camera_index
        new_index = index % len(self.world.camera_list)
        if old_index == new_index:
            return

        self._set_camera_active(old_index, False)
        self.camera_index = new_index
        self._set_camera_active(new_index, True)

    def _set_camera_active(self, index: int, active: bool) -> None:
        if index == 0:
            if active:
                self.world.camera.reparentTo(self.world.render)
            else:
                self.world.camera.detachNode()
            return

        camera = self.world.camera_list[index]
        camera.display_region.setActive(active)
        overlay = getattr(camera, "set_overlay_visible", None)
        if overlay is not None:
            overlay(active)

    def save_current_camera_image(self) -> Path:
        """Save the composited view, including camera-specific overlays."""
        output_directory = Path("logs")
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H.%M.%S")
        output_path = output_directory / f"camera_{timestamp}.png"

        image = PNMImage()
        if not self.world.win.getScreenshot(image):
            raise RuntimeError("Panda3D could not capture the current camera view")
        filename = Filename.fromOsSpecific(str(output_path))
        if not image.write(filename):
            raise RuntimeError(f"Panda3D could not save camera image to {output_path}")
        print(f"Saved camera image to {output_path}")
        return output_path
