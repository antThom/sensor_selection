"""Configurable long-wave infrared camera sensor.

The camera owns two related representations:

* a Panda3D camera node used to view the scene from the agent mount point; and
* a radiometric conversion path that turns a temperature array into a
  displayable thermal image.

Temperatures are kelvin, wavelength limits are micrometres, focal length and
pixel pitch are millimetres and micrometres respectively, and NETD is kelvin.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from panda3d.core import NodePath, PandaNode, PNMImage

from sim.Environment.Thermal.thermal_manager import (
    ThermalManager,
    ThermalMaterialLibrary,
)
from sim.Sensors.Cameras.camera import Camera
from sim.Sensors.sensor import SensorType


class IRCamera(Camera):
    """Agent-mounted radiometric IR camera with configurable optics."""

    def __init__(self, thermal_manager: Optional[ThermalManager] = None):
        super().__init__()

        # Identification and detector construction.
        self.name = "General IR Camera"
        self.manufacturer = "Generic"
        self.model_name = "Generic LWIR"
        self.model_number = "generic-lwir"
        self.detector_type = "Uncooled microbolometer"
        self.radiometric = True

        # Detector and readout parameters.
        self.width = 640
        self.height = 512
        self.frame_rate_hz = 30.0
        self.pixel_pitch_um = 12.0
        self.ifov_mrad = 1.304
        self.spectral_band_um = [8.0, 14.0]
        self.netd_K = 0.050
        self.temperature_accuracy_K = 5.0
        self.temperature_range_K = [233.15, 423.15]
        self.gain_mode = "high"
        self.nuc_mode = "factory"

        # Lens and scene-view parameters. Panda3D cameras look along local +Y.
        self.horizontal_fov_deg = 50.0
        self.vertical_fov_deg = 40.0
        self.focal_length_mm = 9.2
        self.f_number = 1.0
        self.digital_zoom = 1.0
        self.near = 0.1
        self.far = 1000.0
        self.camera_model = "pinhole"
        self.mount_mode = "model_bounds"
        self.mount_position = [0.5, 1.02, 0.5]
        self.mount_hpr = [0.0, -20.0, 0.0]
        self.mass_g = 10.0
        self.power_consumption_W = 1.0

        # Radiometric assumptions used when converting true surface
        # temperature to apparent temperature at the detector.
        self.emissivity = 0.95
        self.atmospheric_transmission = 1.0
        self.atmospheric_temperature_K = 293.15
        self.reflected_temperature_K = 293.15

        # Output settings.
        self.input = "scene_temperature"
        self.output = "thermal_image"
        self.encode = "ir"
        self.palette = "ironbow"
        self.noise_seed = 0

        self.thermal_manager = thermal_manager
        # Keep the older attribute name available to callers in this branch.
        self.thermal_mgr = thermal_manager
        self.type = SensorType.IRCAMERA

        # Live scene rendering uses Panda3D's per-camera tag states.  Scene
        # geometry remains normally textured for RGB cameras; only this camera
        # substitutes temperature-derived palette colors during its traversal.
        self.thermal_palette_bins = 64
        self._thermal_nodes = []

    @property
    def WIDTH(self):
        """Compatibility alias for older camera configuration files."""
        return self.width

    @WIDTH.setter
    def WIDTH(self, value):
        self.width = int(value)

    @property
    def HEIGHT(self):
        """Compatibility alias for older camera configuration files."""
        return self.height

    @HEIGHT.setter
    def HEIGHT(self, value):
        self.height = int(value)

    @property
    def fov(self):
        """Compatibility alias for the horizontal field of view."""
        return self.horizontal_fov_deg

    @fov.setter
    def fov(self, value):
        self.horizontal_fov_deg = float(value)

    @property
    def aspect_ratio(self):
        return float(self.width) / float(self.height)

    def validate_parameters(self):
        """Reject configurations that cannot represent a physical camera."""
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("IR camera width and height must be positive")
        if not 0.0 < float(self.horizontal_fov_deg) < 180.0:
            raise ValueError("horizontal_fov_deg must be between 0 and 180")
        if not 0.0 < float(self.vertical_fov_deg) < 180.0:
            raise ValueError("vertical_fov_deg must be between 0 and 180")
        if float(self.near) <= 0 or float(self.far) <= float(self.near):
            raise ValueError("camera clipping planes must satisfy 0 < near < far")
        if float(self.frame_rate_hz) <= 0:
            raise ValueError("frame_rate_hz must be positive")
        if (
            float(self.pixel_pitch_um) <= 0
            or float(self.ifov_mrad) <= 0
            or float(self.netd_K) < 0
        ):
            raise ValueError(
                "pixel pitch and IFOV must be positive and NETD non-negative"
            )
        if len(self.spectral_band_um) != 2:
            raise ValueError("spectral_band_um must contain [minimum, maximum]")
        if len(self.temperature_range_K) != 2:
            raise ValueError("temperature_range_K must contain [minimum, maximum]")
        if float(self.temperature_range_K[0]) >= float(self.temperature_range_K[1]):
            raise ValueError("temperature range minimum must be below maximum")
        if not 0.0 < float(self.emissivity) <= 1.0:
            raise ValueError("emissivity must be in the interval (0, 1]")
        if not 0.0 <= float(self.atmospheric_transmission) <= 1.0:
            raise ValueError("atmospheric_transmission must be between 0 and 1")
        return self

    def apparent_temperature(
        self,
        surface_temperature_K,
        *,
        emissivity: Optional[float] = None,
        atmospheric_transmission: Optional[float] = None,
    ):
        """Apply a compact radiance-space atmosphere/emissivity model.

        The Stefan-Boltzmann ``T**4`` relationship is used as a broadband
        approximation. It accounts for emitted surface radiance, reflected
        background radiance, and radiation emitted by the intervening
        atmosphere. This is intentionally simpler than a wavelength-resolved
        detector calibration but preserves the important nonlinear behavior.
        """
        temperature = np.asarray(surface_temperature_K, dtype=np.float64)
        if np.any(temperature <= 0):
            raise ValueError("surface temperatures must be greater than 0 K")

        eps = self.emissivity if emissivity is None else float(emissivity)
        tau = (
            self.atmospheric_transmission
            if atmospheric_transmission is None
            else float(atmospheric_transmission)
        )
        if not 0.0 < eps <= 1.0:
            raise ValueError("emissivity must be in the interval (0, 1]")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("atmospheric_transmission must be between 0 and 1")

        reflected = float(self.reflected_temperature_K) ** 4
        atmosphere = float(self.atmospheric_temperature_K) ** 4
        detector_radiance = tau * (
            eps * np.power(temperature, 4) + (1.0 - eps) * reflected
        )
        detector_radiance += (1.0 - tau) * atmosphere
        return np.power(np.maximum(detector_radiance, 0.0), 0.25)

    def temperature_to_image(
        self,
        temperature_frame_K,
        *,
        palette: Optional[str] = None,
        add_noise: bool = True,
    ):
        """Convert a two-dimensional kelvin array to an 8-bit RGB image."""
        self.validate_parameters()
        temperature = self.apparent_temperature(temperature_frame_K)
        if temperature.ndim != 2:
            raise ValueError("temperature_frame_K must be a 2-D array")

        if add_noise and float(self.netd_K) > 0:
            rng = np.random.default_rng(int(self.noise_seed))
            temperature = temperature + rng.normal(
                0.0, float(self.netd_K), temperature.shape
            )

        minimum, maximum = (float(value) for value in self.temperature_range_K)
        normalized = np.clip(
            (temperature - minimum) / (maximum - minimum),
            0.0,
            1.0,
        )
        return self._apply_palette(
            normalized, self.palette if palette is None else palette
        )

    @staticmethod
    def _apply_palette(normalized, palette):
        palette_name = str(palette).lower().replace("-", "_")
        if palette_name in {"white_hot", "grayscale", "grey"}:
            gray = np.rint(normalized * 255.0).astype(np.uint8)
            return np.repeat(gray[..., None], 3, axis=2)
        if palette_name == "black_hot":
            gray = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
            return np.repeat(gray[..., None], 3, axis=2)
        if palette_name not in {"ironbow", "iron", "false_color"}:
            raise ValueError(f"unsupported IR palette: {palette!r}")

        # Compact ironbow approximation. Interpolation provides a smooth
        # display without requiring OpenCV or another image-processing package.
        stops = np.array([0.0, 0.20, 0.45, 0.70, 0.88, 1.0])
        colors = np.array(
            [
                [0, 0, 0],
                [38, 12, 74],
                [132, 22, 86],
                [224, 75, 35],
                [255, 190, 55],
                [255, 255, 240],
            ],
            dtype=np.float64,
        )
        channels = [
            np.interp(normalized, stops, colors[:, channel]) for channel in range(3)
        ]
        return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)

    def temperatures_for_body_ids(self, body_ids):
        """Resolve a PyBullet-style body-ID image to a kelvin frame."""
        if self.thermal_manager is None:
            raise RuntimeError("no ThermalManager is attached to this camera")
        identifiers = np.asarray(body_ids)
        result = np.full(
            identifiers.shape,
            float(self.thermal_manager.ambient),
            dtype=np.float64,
        )
        for body_id in np.unique(identifiers):
            if int(body_id) < 0:
                continue
            result[identifiers == body_id] = self.thermal_manager.get_temperature(
                int(body_id)
            )
        return result

    def setup_live_thermal_view(self, world):
        """Configure this mounted camera to display scene temperature.

        Each registered scene root receives a discrete palette-bin tag.  The
        tag is reevaluated every frame, so an agent's rendered color follows
        its live ``ThermalObject.temperature`` value without altering the
        render state seen by any other camera.
        """
        if self.camera_node is None:
            raise RuntimeError("IR camera must be mounted before thermal setup")

        self._build_thermal_render_states()
        self._register_scene_thermal_nodes(world)
        self.refresh_live_thermal_colors()

        task_name = f"update-{self.model_number}-thermal-view-{id(self)}"
        world.taskMgr.add(self._update_live_thermal_view, task_name)
        return self

    def register_thermal_node(self, node_path, temperature_source, emissivity=None):
        """Associate a Panda3D scene root with a live kelvin value.

        ``temperature_source`` may be a number, a callable returning a number,
        or an object exposing a ``temperature`` attribute.
        """
        if node_path is None or node_path.isEmpty():
            return
        self._thermal_nodes.append(
            {
                "node": node_path,
                "source": temperature_source,
                "emissivity": self.emissivity if emissivity is None else emissivity,
            }
        )

    def refresh_live_thermal_colors(self):
        """Update all scene tags from their current simulated temperatures."""
        minimum, maximum = (float(value) for value in self.temperature_range_K)
        bin_count = max(2, int(self.thermal_palette_bins))
        for registration in self._thermal_nodes:
            node = registration["node"]
            if node.isEmpty():
                continue
            temperature = self._resolve_temperature(registration["source"])
            apparent = float(
                self.apparent_temperature(
                    temperature,
                    emissivity=float(registration["emissivity"]),
                )
            )
            normalized = np.clip((apparent - minimum) / (maximum - minimum), 0, 1)
            palette_bin = int(round(float(normalized) * (bin_count - 1)))
            node.setTag("thermal-palette-bin", str(palette_bin))

    def _build_thermal_render_states(self):
        """Create flat-color states used only while this camera renders."""
        bin_count = max(2, int(self.thermal_palette_bins))
        self.camera_node.setTagStateKey("thermal-palette-bin")

        # Untagged geometry is shown at ambient temperature, preventing RGB
        # textures from leaking into the thermal view.
        ambient_color = self._palette_color_for_temperature(
            self.thermal_manager.ambient if self.thermal_manager else 293.0
        )
        self.camera_node.setInitialState(
            self._make_flat_color_state(ambient_color, priority=50)
        )

        for palette_bin in range(bin_count):
            normalized = palette_bin / float(bin_count - 1)
            rgb = self._apply_palette(np.asarray(normalized), self.palette)
            self.camera_node.setTagState(
                str(palette_bin),
                self._make_flat_color_state(rgb, priority=100),
            )

    def _register_scene_thermal_nodes(self, world):
        self._thermal_nodes.clear()
        materials = ThermalMaterialLibrary.MATERIALS

        terrain = getattr(getattr(world, "terrain", None), "object", None)
        if terrain is not None:
            terrain_material = materials["terrain"]
            self.register_thermal_node(
                terrain, terrain_material["T"], terrain_material["emiss"]
            )

        trees = getattr(getattr(world, "object_loader", None), "static_object", {})
        tree_material = materials["tree"]
        for tree in trees.get("trees", []):
            self.register_thermal_node(tree, tree_material["T"], tree_material["emiss"])

        if self.agent is not None and self.agent.object_node_path is not None:
            thermal_body = getattr(self.agent, "thermal_object", None)
            source = thermal_body if thermal_body is not None else self.agent
            emissivity = getattr(source, "emiss", materials["robot"]["emiss"])
            self.register_thermal_node(
                self.agent.object_node_path,
                source,
                emissivity,
            )

        sky = getattr(getattr(getattr(world, "sky", None), "sky", None), "sky", None)
        if sky is not None and self.thermal_manager is not None:
            self.register_thermal_node(sky, self.thermal_manager.T_sky, 1.0)

    @staticmethod
    def _resolve_temperature(source):
        if callable(source):
            source = source()
        if hasattr(source, "temperature"):
            source = source.temperature
        return float(source)

    def _palette_color_for_temperature(self, temperature_K):
        minimum, maximum = (float(value) for value in self.temperature_range_K)
        apparent = float(self.apparent_temperature(temperature_K))
        normalized = np.clip((apparent - minimum) / (maximum - minimum), 0, 1)
        return self._apply_palette(np.asarray(normalized), self.palette)

    @staticmethod
    def _make_flat_color_state(rgb, priority):
        holder = NodePath(PandaNode("thermal-flat-color-state"))
        color = np.asarray(rgb, dtype=np.float64) / 255.0
        holder.setColor(float(color[0]), float(color[1]), float(color[2]), 1.0, priority)
        holder.setTextureOff(priority)
        holder.setLightOff(priority)
        holder.setMaterialOff(priority)
        holder.setShaderOff(priority)
        return holder.getState()

    def _update_live_thermal_view(self, task):
        self.refresh_live_thermal_colors()
        return task.cont

    def capture_scene_image(self):
        """Capture the Panda3D display region for this mounted camera."""
        if self.display_region is None:
            raise RuntimeError(
                "IR camera must be set up by SensorLoader before capture"
            )
        window_image = PNMImage()
        if not self.display_region.getScreenshot(window_image):
            raise RuntimeError("Panda3D could not capture the IR camera view")
        if window_image.getXSize() == int(
            self.width
        ) and window_image.getYSize() == int(self.height):
            return window_image

        image = PNMImage(int(self.width), int(self.height), 3)
        image.quickFilterFrom(window_image)
        return image

    def get_output(self, temperature_frame_K=None):
        """Return a thermal RGB array or capture the live Panda3D scene view."""
        if temperature_frame_K is not None:
            return self.temperature_to_image(temperature_frame_K)
        return self.capture_scene_image()
