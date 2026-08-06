"""Configurable long-wave infrared camera sensor.

The camera owns two related representations:

* a Panda3D camera node used to view the scene from the agent mount point; and
* a radiometric conversion path that turns a temperature array into a
  displayable thermal image.

Temperatures are kelvin, wavelength limits are micrometres, focal length and
pixel pitch are millimetres and micrometres respectively, and NETD is kelvin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from panda3d.core import ClockObject, NodePath, PandaNode, PNMImage, Shader, Vec4

from sim.environment.thermal.thermal_manager import (
    ThermalManager,
    ThermalMaterialLibrary,
)
from sim.sensors.cameras.camera import Camera
from sim.sensors.sensor import SensorType


class IRCamera(Camera):
    """Agent-mounted radiometric IR camera with configurable optics."""

    def __init__(self, thermal_manager: ThermalManager | None = None):
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
        # The detector can measure the full range above, while its display
        # gain uses a narrower window to reveal scene-scale thermal gradients.
        self.display_temperature_range_K = [260.0, 330.0]
        self.automatic_gain_control = True
        self.agc_min_span_K = 30.0
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
        self.atmospheric_extinction_per_m = 0.00015
        self.surface_temperature_variation_K = 1.0
        self.solar_surface_gain_K = 6.0

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

        self._thermal_nodes = []
        self._thermal_shader = None
        self._thermal_world = None
        self._active_display_range_K = list(self.display_temperature_range_K)
        self.thermal_legend = None

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
        if len(self.display_temperature_range_K) != 2:
            raise ValueError(
                "display_temperature_range_K must contain [minimum, maximum]"
            )
        if float(self.display_temperature_range_K[0]) >= float(
            self.display_temperature_range_K[1]
        ):
            raise ValueError("display temperature minimum must be below maximum")
        if float(self.atmospheric_extinction_per_m) < 0.0:
            raise ValueError("atmospheric_extinction_per_m cannot be negative")
        if float(self.agc_min_span_K) <= 0.0:
            raise ValueError("agc_min_span_K must be positive")
        if not 0.0 < float(self.emissivity) <= 1.0:
            raise ValueError("emissivity must be in the interval (0, 1]")
        if not 0.0 <= float(self.atmospheric_transmission) <= 1.0:
            raise ValueError("atmospheric_transmission must be between 0 and 1")
        return self

    def apparent_temperature(
        self,
        surface_temperature_K,
        *,
        emissivity: float | None = None,
        atmospheric_transmission: float | None = None,
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
        palette: str | None = None,
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

        A single GPU shader computes geometry-aware apparent temperature for
        every fragment. Per-object bulk temperatures are refreshed from the
        thermal simulation without altering other cameras' render states.
        """
        if self.camera_node is None:
            raise RuntimeError("IR camera must be mounted before thermal setup")

        self._thermal_world = world
        self._build_thermal_render_state()
        self._register_scene_thermal_nodes(world)
        from sim.sensors.cameras.thermal_legend import ThermalLegend

        self.thermal_legend = ThermalLegend(world, self)
        self.refresh_live_thermal_colors()

        task_name = f"update-{self.model_number}-thermal-view-{id(self)}"
        world.taskMgr.add(self._update_live_thermal_view, task_name)
        return self

    def refresh_scene_thermal_nodes(self, world=None):
        """Rediscover thermal geometry after all scene agents are loaded."""
        world = self._thermal_world if world is None else world
        if world is None:
            return
        self._register_scene_thermal_nodes(world)
        self.refresh_live_thermal_colors()

    def set_overlay_visible(self, visible):
        """Show or hide the live temperature guide for camera switching."""
        if self.thermal_legend is not None:
            self.thermal_legend.set_visible(bool(visible))

    def register_thermal_node(
        self,
        node_path,
        temperature_source,
        emissivity=None,
        *,
        variation_K=None,
        variation_scale=0.1,
        solar_gain_K=None,
        texture_variation_K=1.0,
        atmosphere=True,
    ):
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
                "variation_K": (
                    self.surface_temperature_variation_K
                    if variation_K is None
                    else float(variation_K)
                ),
                "variation_scale": float(variation_scale),
                "solar_gain_K": (
                    self.solar_surface_gain_K
                    if solar_gain_K is None
                    else float(solar_gain_K)
                ),
                "texture_variation_K": float(texture_variation_K),
                "atmosphere": bool(atmosphere),
                "last_temperature": None,
            }
        )
        
        # This piece of code was mysteriously put here for an unknown reason...and returned errors
        # Thus, it's commented out
        
        # channels = [
        #     np.interp(normalized, stops, colors[:, channel]) for channel in range(3)
        # ]
        # return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)

    def refresh_live_thermal_colors(self):
        """Refresh cheap per-object and per-frame GPU shader inputs."""
        foreground_temperatures = []
        for registration in self._thermal_nodes:
            node = registration["node"]
            if node.isEmpty():
                continue
            temperature = self._resolve_temperature(registration["source"])
            if registration["atmosphere"]:
                foreground_temperatures.append(temperature)
            previous = registration["last_temperature"]
            if previous is None or abs(temperature - previous) >= max(
                float(self.netd_K), 0.001
            ):
                node.setShaderInput(
                    "thermal_object",
                    Vec4(
                        temperature,
                        float(registration["emissivity"]),
                        registration["variation_K"],
                        registration["variation_scale"],
                    ),
                )
                node.setShaderInput(
                    "thermal_effects",
                    Vec4(
                        registration["solar_gain_K"],
                        1.0 if registration["atmosphere"] else 0.0,
                        registration["texture_variation_K"],
                        0.0,
                    ),
                )
                registration["last_temperature"] = temperature

        self._active_display_range_K = self._automatic_display_range(
            foreground_temperatures
        )
        self._update_global_shader_inputs()

    def _automatic_display_range(self, temperatures):
        """Return an AGC window that preserves useful scene contrast."""
        if not self.automatic_gain_control or not temperatures:
            return list(self.display_temperature_range_K)
        low = min(temperatures) - 10.0
        high = max(temperatures) + 10.0
        span = max(high - low, float(self.agc_min_span_K))
        center = (low + high) * 0.5
        return [center - span * 0.5, center + span * 0.5]

    def _build_thermal_render_state(self):
        """Load one fragment shader for the IR camera's render traversal."""
        shader_dir = Path(__file__).resolve().parent / "shaders"
        self._thermal_shader = Shader.make(
            Shader.SL_GLSL,
            (shader_dir / "thermal.vert.glsl").read_text(encoding="utf-8"),
            (shader_dir / "thermal.frag.glsl").read_text(encoding="utf-8"),
        )
        holder = NodePath(PandaNode("thermal-camera-render-state"))
        holder.setShader(self._thermal_shader, 100)
        holder.setLightOff(100)
        holder.setMaterialOff(100)
        ambient = self.thermal_manager.ambient if self.thermal_manager else 293.0
        holder.setShaderInput("thermal_object", Vec4(ambient, 0.95, 0.5, 0.1))
        holder.setShaderInput("thermal_effects", Vec4(2.0, 1.0, 0.0, 0.0))
        self.camera_node.setInitialState(holder.getState())

    def _register_scene_thermal_nodes(self, world):
        self._thermal_nodes.clear()
        materials = ThermalMaterialLibrary.MATERIALS

        terrain = getattr(getattr(world, "terrain", None), "object", None)
        if terrain is not None:
            terrain_material = materials["terrain"]
            self.register_thermal_node(
                terrain,
                terrain_material["T"],
                terrain_material["emiss"],
                variation_K=1.8,
                variation_scale=0.018,
                solar_gain_K=8.0,
                texture_variation_K=3.0,
            )

        trees = getattr(getattr(world, "object_loader", None), "static_object", {})
        tree_material = materials["tree"]
        for tree in trees.get("trees", []):
            self.register_thermal_node(
                tree,
                tree_material["T"],
                tree_material["emiss"],
                variation_K=0.8,
                variation_scale=0.25,
                solar_gain_K=4.0,
                texture_variation_K=1.5,
            )

        agents = list(getattr(world, "agent_list", []))
        if self.agent is not None and self.agent not in agents:
            agents.append(self.agent)
        for agent in agents:
            if agent.object_node_path is None:
                continue
            thermal_body = getattr(agent, "thermal_object", None)
            source = thermal_body if thermal_body is not None else agent
            emissivity = getattr(source, "emiss", materials["robot"]["emiss"])
            self.register_thermal_node(
                agent.object_node_path,
                source,
                emissivity,
                variation_K=0.6,
                variation_scale=1.5,
                solar_gain_K=3.0,
                texture_variation_K=0.75,
            )

        sky_controller = getattr(getattr(world, "sky", None), "sky", None)
        sky = getattr(sky_controller, "sky", None)
        if sky is not None and self.thermal_manager is not None:
            self.register_thermal_node(
                sky,
                self.thermal_manager.T_sky,
                1.0,
                variation_K=0.15,
                variation_scale=0.0002,
                solar_gain_K=0.0,
                texture_variation_K=0.0,
                atmosphere=False,
            )
        sun = getattr(sky_controller, "sun", None)
        if sun is not None:
            self.register_thermal_node(
                sun,
                1000.0,
                1.0,
                variation_K=0.0,
                variation_scale=0.0,
                solar_gain_K=0.0,
                texture_variation_K=0.0,
                atmosphere=False,
            )

    def _update_global_shader_inputs(self):
        world = self._thermal_world
        if world is None:
            return
        manager = self.thermal_manager
        ambient = manager.ambient if manager else 293.0
        sky_temperature = manager.T_sky if manager else 260.0
        minimum, maximum = (float(value) for value in self._active_display_range_K)
        world.render.setShaderInput(
            "thermal_environment",
            Vec4(
                ambient,
                sky_temperature,
                float(self.reflected_temperature_K),
                float(self.atmospheric_temperature_K),
            ),
        )
        world.render.setShaderInput(
            "thermal_camera",
            Vec4(
                minimum,
                maximum,
                float(self.netd_K),
                float(self.atmospheric_extinction_per_m),
            ),
        )
        sun_direction = (0.0, 0.0, 1.0)
        sun_strength = 1.0
        sky = getattr(getattr(world, "sky", None), "sky", None)
        sun = getattr(sky, "sun", None)
        if sun is not None:
            direction = sun.getPos(world.render)
            if direction.lengthSquared() > 0.0:
                direction.normalize()
                sun_direction = (direction.x, direction.y, direction.z)
                sun_strength = max(0.0, direction.z)
        world.render.setShaderInput(
            "thermal_sun",
            Vec4(*sun_direction, sun_strength),
        )
        camera_position = self.camera_nodepath.getPos(world.render)
        world.render.setShaderInput(
            "thermal_camera_position",
            Vec4(camera_position.x, camera_position.y, camera_position.z, 1.0),
        )
        world.render.setShaderInput(
            "thermal_frame", ClockObject.getGlobalClock().getFrameTime()
        )
        world.render.setShaderInput("thermal_palette", self._palette_mode())
        world.render.setShaderInput(
            "thermal_base_transmission", float(self.atmospheric_transmission)
        )
        if self.thermal_legend is not None:
            self.thermal_legend.update(self._active_display_range_K, self.palette)

    def _palette_mode(self):
        palette = str(self.palette).lower().replace("-", "_")
        if palette in {"white_hot", "grayscale", "grey"}:
            return 0.0
        if palette == "black_hot":
            return 1.0
        if palette in {"ironbow", "iron", "false_color"}:
            return 2.0
        raise ValueError(f"unsupported IR palette: {self.palette!r}")

    @staticmethod
    def _resolve_temperature(source):
        if callable(source):
            source = source()
        if hasattr(source, "temperature"):
            source = source.temperature
        return float(source)

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
