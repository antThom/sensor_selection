from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QStackedWidget,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, pyqtSlot
import numpy as np
import pybullet as p
from sim.Sensor.Microphone.microphone import MicrophoneSensor_Uniform
from sim.Sensor.Cameras.depth_camera import DepthCamera
from sim.Sensor.Cameras.ir_camera import IRCamera
from sim.Sensor.Cameras.camera import Camera
from sim.Sensor.Lidar.lidar import Lidar

import matplotlib.pyplot as plt
import io


class CameraViewer(QWidget):
    """GUI to display agents' fixed views (top, side, front) plus a selectable PyBullet sensor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Agent Viewer")

        # Data storage
        self.cam = {}
        self.label = {}
        self.drop_down = {}
        self.sensor_selectors = {}
        self.agent_index = {}
        self.agents_map = {}

        # Layout setup
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Stack for agent pages
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # Default offsets for top/side/front cameras
        self.camera_offsets = {
            "top": {"eye": [0, 0.1, 1], "target": [0, 0, 0]},
            "side": {"eye": [0, 1, 0], "target": [0, 0, 0]},
            "front": {"eye": [0.75, 0, 0], "target": [1, 0, -0.1]},
        }

    # ----------------------------
    # TEAM AND AGENT MANAGEMENT
    # ----------------------------
    def add_team(self, team, team_name):
        """Add all agents of a team to the viewer."""
        selector = QComboBox()
        self.drop_down[team_name] = selector
        self.layout.insertWidget(0, selector)

        for idx, agent_obj in enumerate(team.agents):
            display = f"{team_name} Agent {idx}"
            selector.addItem(display)

            page = self._build_agent_page(team_name, idx, agent_obj)
            self.agent_index[display] = self.stack.count()
            self.stack.addWidget(page)
            self.agents_map[f"{team_name}_{idx}"] = agent_obj

        selector.currentTextChanged.connect(self.on_agent_changed)
        selector.setCurrentIndex(0)

    # ----------------------------
    # PAGE BUILDING
    # ----------------------------
    def _build_agent_page(self, team_name, agent_idx, agent_obj):
        """Create a page showing top/side/front views and one sensor selector."""
        layout = QVBoxLayout()

        # --- Row 1: top/side/front QImage views ---
        row_views = QHBoxLayout()
        for view_name in ["top", "side", "front"]:
            key = f"{team_name}_{agent_idx}_{view_name}"
            row_views.addWidget(self._make_camera_widget(f"{view_name.capitalize()} View", key))
        layout.addLayout(row_views)

        # --- Row 2: sensor dropdown + selected sensor view ---
        sensor_row = QVBoxLayout()

        sensor_selector = QComboBox()
        sensor_selector.addItem("Select Sensor")
        if getattr(agent_obj, "has_sensor", False):
            for s in agent_obj.sensors:
                sensor_selector.addItem(s.name)
                # connect each sensor's frame signal to GUI update slot
                s.signals.new_frame.connect(
                    lambda name, img, ts, t=team_name, a=agent_idx: self.on_new_sensor_frame(t, a, name, img, ts)
                )
        
        sensor_row.addWidget(sensor_selector)

        sensor_key = f"{team_name}_{agent_idx}_selected_sensor"
        sensor_row.addWidget(self._make_camera_widget("Selected Sensor View", sensor_key))

        self.sensor_selectors[f"{team_name}_{agent_idx}"] = sensor_selector
        sensor_selector.currentTextChanged.connect(
            lambda sensor_name, t=team_name, a=agent_idx: self.on_sensor_selected(t, a, sensor_name)
        )

        layout.addLayout(sensor_row)

        # finalize
        page = QWidget()
        page.setLayout(layout)
        return page

    def _make_camera_widget(self, title, key):
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Arial", 12, QFont.Bold))
        lbl_image = QLabel()
        lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl_image.setScaledContents(True)

        # placeholder image
        img = QImage(320, 240, QImage.Format_RGB888)
        img.fill(0xFFFFFF)
        lbl_image.setPixmap(QPixmap.fromImage(img))

        v = QVBoxLayout()
        v.addWidget(lbl_title, alignment=Qt.AlignHCenter)
        v.addWidget(lbl_image)
        w = QWidget()
        w.setLayout(v)

        self.label[key] = lbl_title
        self.cam[key] = lbl_image
        return w

    # ----------------------------
    # SELECTION HANDLERS
    # ----------------------------
    def on_agent_changed(self, display_name):
        """When switching agents, reset the sensor dropdown and clear image."""
        idx = self.agent_index.get(display_name, 0)
        self.stack.setCurrentIndex(idx)

        parts = display_name.split()
        if len(parts) >= 3 and parts[-2] == "Agent":
            team_name = parts[0]
            try:
                agent_idx = int(parts[-1])
            except ValueError:
                return

            # reset dropdown
            selector_key = f"{team_name}_{agent_idx}"
            if selector_key in self.sensor_selectors:
                selector = self.sensor_selectors[selector_key]
                selector.blockSignals(True)
                selector.setCurrentIndex(0)
                selector.blockSignals(False)

            # clear selected sensor view
            sensor_key = f"{team_name}_{agent_idx}_selected_sensor"
            if sensor_key in self.cam:
                self.cam[sensor_key].clear()
                self.label[sensor_key].setText("Selected Sensor View")

    def on_sensor_selected(self, team_name, agent_idx, sensor_name):
        sensor_key = f"{team_name}_{agent_idx}_selected_sensor"
        if not sensor_name or sensor_name == "Select Sensor":
            self.cam[sensor_key].clear()
            self.label[sensor_key].setText("Selected Sensor View")
            return

        agent = self.agents_map[f"{team_name}_{agent_idx}"]
        selected_sensor = next((s for s in agent.sensors if s.name == sensor_name), None)
        if not selected_sensor:
            return

        try:
            data = selected_sensor.get_output()

            # --- CASE 1: RGB Camera ---
            if isinstance(data, Camera):
                h, w, c = data.shape
                qimg = QImage(data.tobytes(), w, h, c * w, QImage.Format_RGB888)
                self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))
                self.label[sensor_key].setText(f"{sensor_name.capitalize()} (Camera)")

            # --- CASE 2: Microphone ---
            elif isinstance(selected_sensor, MicrophoneSensor_Uniform):
                waveform = data.flatten()
                # Convert to spectrogram image
                self._update_mic_plot(sensor_key, selected_sensor, waveform)

                # self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))
                self.label[sensor_key].setText(f"{sensor_name.capitalize()} (Microphone Spectrogram)")

            # --- CASE 3: RGB + Depth Camera ---
            elif isinstance(selected_sensor, DepthCamera):
                rgb = data[0]
                depth = data[1]

                h, w, c = rgb.shape
                qimg = QImage(rgb.tobytes(), w, h, c * w, QImage.Format_RGB888)
                self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))
                self.label[sensor_key].setText(f"{sensor_name.capitalize()} (Depth Camera)")

                # print(f"Depth: {depth[0]}")
            # --- CASE 4: IR Camera ---
            elif isinstance(selected_sensor, IRCamera):
                self.label[sensor_key].setText(f"{sensor_name.capitalize()} (Thermal Camera)")
                h, w, c = data.shape
                qimg = QImage(data.tobytes(), w, h, c * w, QImage.Format_RGB888)
                self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))
            
            # --- CASE 5: LIDAR
            elif isinstance(selected_sensor, Lidar):
                lidar_range = data[0]
                point_cloud = data[1]
                # print(f"Point Cloud: {point_cloud.shape}")
                img = self.lidar_points_to_topdown_rgb(points_sensor=point_cloud, range_sensor=lidar_range, meters_per_pixel=0.5, radius_m=80.0)
                # print("past")
                # lidar_range = data[0]
                # point_cloud = data[1]

                # print(f"Point Cloud: {point_cloud}")

                h, w, c = img.shape
                qimg = QImage(img.tobytes(), w, h, c * w, QImage.Format_RGB888)
                self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))
                self.label[sensor_key].setText(f"{sensor_name.capitalize()} (Lidar Point Cloud)")

            else:
                self.label[sensor_key].setText(f"{sensor_name.capitalize()} (Unknown Type)")

        except Exception as e:
            print(f"Error rendering {sensor_name} for {selected_sensor}: {e}")

    # ----------------------------
    # MICROPHONE VISUALIZATION
    # ----------------------------
    def _update_mic_plot(self, sensor_key, mic, waveform):
        """Render spectrogram directly into the Qt widget (no new window)."""
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from matplotlib.figure import Figure
            import io

            # Create an offscreen matplotlib figure (Agg backend)
            fig = Figure(figsize=(3, 2), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # Plot spectrogram on this offscreen canvas
            ax.specgram(
                waveform,
                Fs=mic.sample_rate,
                NFFT=256,
                noverlap=128,
                cmap="viridis",
            )
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Render to RGBA buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            width, height = fig.get_size_inches() * fig.get_dpi()
            width, height = int(width), int(height)
            qimg = QImage(buf, width, height, QImage.Format_RGBA8888)

            # Set spectrogram image in the GUI widget
            # print(waveform)
            self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))

        except Exception as e:
            print(f"[GUI] Spectrogram render error: {e}")

    # ----------------------------
    # FIXED VIEWS (TOP/SIDE/FRONT)
    # ----------------------------
    def get_camera(self, view_name, pos):
        eye_offset = self.camera_offsets[view_name]["eye"]
        target_offset = self.camera_offsets[view_name]["target"]
        eye = np.array(pos) + np.array(eye_offset)
        target = np.array(pos) + np.array(target_offset)
        return eye.tolist(), target.tolist()

    def update_fixed_views(self, pos, team_name, agent_idx):
        """Render top/side/front views for the given agent."""
        def render(eye, target):
            view = p.computeViewMatrix(eye, target, [0, 0, 1])
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
            _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj)
            img = np.reshape(rgb, (240, 320, 4))[:, :, :3]
            h, w, c = img.shape
            bpl = c * w
            return QImage(img.tobytes(), w, h, bpl, QImage.Format_RGB888)

        for v in ["top", "side", "front"]:
            key = f"{team_name}_{agent_idx}_{v}"
            eye, target = self.get_camera(v, pos)
            qimg = render(eye, target)
            if key in self.cam:
                self.cam[key].setPixmap(QPixmap.fromImage(qimg))

    # ----------------------------
    # AUTO REFRESH
    # ----------------------------
    def start_auto_refresh(self, interval_ms=200):
        """Refresh fixed views + selected sensor view periodically."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_current_agent)
        self.timer.start(interval_ms)

    def refresh_current_agent(self):
        """Update the current agent's 3 views and selected sensor view."""
        for team_name, combo in self.drop_down.items():
            display = combo.currentText()
            if not display or "Agent" not in display:
                continue
            parts = display.split()
            if len(parts) < 3:
                continue
            agent_idx = int(parts[-1])
            agent = self.agents_map.get(f"{team_name}_{agent_idx}")
            if agent is None:
                continue

            # update the fixed top/side/front views
            if isinstance(agent.position, tuple):
                pos = list(agent.position)
            else:
                pos = agent.position.flatten().tolist()

            self.update_fixed_views(pos, team_name, agent_idx)

            # refresh selected sensor (if any)
            selector = self.sensor_selectors.get(f"{team_name}_{agent_idx}")
            if selector:
                sensor_name = selector.currentText()
                if sensor_name and sensor_name != "Select Sensor":
                    self.on_sensor_selected(team_name, agent_idx, sensor_name)

    def lidar_points_to_topdown_rgb(self, points_sensor, range_sensor, meters_per_pixel=0.05, radius_m=20.0):
        """
        points_sensor: (N,3) float or (H,W,3); NaNs for invalid
        Assumes sensor frame: +X forward, +Y left (common). Uses XY plane.
        Returns an RGB image (uint8) with points drawn as white.
        """

        pts = np.asarray(points_sensor, dtype=np.float32).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        # Keep within radius
        x, y = pts[:, 0], pts[:, 1]
        mask = (x*x + y*y) <= (radius_m * radius_m)
        x, y = x[mask], y[mask]

        size = int(np.ceil((2 * radius_m) / meters_per_pixel))
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # map (x,y) to pixels: center is sensor
        cx = cy = size // 2
        u = (cx + (y / meters_per_pixel)).astype(np.int32)   # y -> image x
        v = (cy - (x / meters_per_pixel)).astype(np.int32)   # x forward -> up

        ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        img[v[ok], u[ok]] = 255  # white points

        # optional: draw axes (simple)
        img[cy, :, 1] = 80
        img[:, cx, 1] = 80
        return img


    @pyqtSlot(str, object, float)
    def on_new_sensor_frame(self, team_name, agent_idx, sensor_name, data, timestamp):
        """Live update when a threaded sensor emits new_frame."""
        sensor_key = f"{team_name}_{agent_idx}_selected_sensor"
        agent = self.agents_map.get(f"{team_name}_{agent_idx}")
        if not agent:
            return

        sensor = next((s for s in agent.sensors if s.name == sensor_name), None)
        if not sensor:
            return

        if isinstance(sensor, MicrophoneSensor_Uniform):
            waveform = np.array(data).flatten()
            self._update_mic_plot(sensor_key, sensor, waveform)
        elif isinstance(data, np.ndarray) and data.ndim == 3:
            h, w, c = data.shape
            qimg = QImage(data.tobytes(), w, h, c * w, QImage.Format_RGB888)
            self.cam[sensor_key].setPixmap(QPixmap.fromImage(qimg))