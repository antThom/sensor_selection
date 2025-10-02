from PyQt5.QtWidgets import (
QApplication, 
QLabel, 
QWidget, 
QVBoxLayout, 
QHBoxLayout, 
QComboBox, 
QStackedWidget,
QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import sim.print_helpers as ph
import os, sys
import time
import pybullet as p
import pybullet_data
import numpy as np

class CameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Agent Viewer")

        # storage
        self.cam = {}                 # key -> QLabel for image
        self.label = {}               # key -> QLabel for title
        self.drop_down = {}           # team_name -> QComboBox
        self.agent_index = {}         # display_name -> stacked index
        # self.layout = QVBoxLayout()
        # self.container = QWidget()
        

        self.camera_offsets = {
            "top"  :     {"eye": [0, 0.1, 1], "target": [0, 0, 0]},
            "side" :     {"eye": [0, 1, 0], "target": [0, 0, 0]},
            "rear" :     {"eye": [-10, 0, 5], "target": [0, 0, 0]},
            "front":     {"eye": [0.75, 0, 0], "target": [1, 0, -0.1]},
        }

        # root layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # one stack to hold all agent pages
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

    # def add_team(self,team,team_name):
    #     # for name in range(team_name):
    #     self.drop_down[team_name] = QComboBox()
    #     self.add_agent(name=team_name,team=team)
    #     self.drop_down.currentTextChanged.connect(self.on_agent_changed)
    #     self.drop_down.setCurrentIndex(0)

    # === Public API you already call ===
    def add_team(self, team, team_name):
        # top bar for this team: agent selector
        selector = QComboBox()
        self.drop_down[team_name] = selector
        self.layout.insertWidget(0, selector)  # put selector above the stack

        # build one page per agent
        for idx, _agent_obj in enumerate(team.agents):
            display = f"{team_name} Agent {idx}"
            selector.addItem(display)

            page = self._build_agent_page(team_name, idx)  # 3 camera widgets
            self.agent_index[display] = self.stack.count()
            self.stack.addWidget(page)

        # connect AFTER items are added
        selector.currentTextChanged.connect(self.on_agent_changed)
        selector.setCurrentIndex(0)  # show first agent by default

    # Called by the combo box
    def on_agent_changed(self, display_name):
        idx = self.agent_index.get(display_name, 0)
        self.stack.setCurrentIndex(idx)

    # === Building UI pieces ===
    def _build_agent_page(self, team_name, agent_idx):
        """Create one page containing the 3 views for a single agent."""
        views = ["top", "side", "front"]  # must match camera_offsets keys
        row = QHBoxLayout()
        for v in views:
            key = f"{team_name}_{agent_idx}_{v}"
            row.addWidget(self._make_camera_widget(f"{v.capitalize()} View", key))
        page = QWidget()
        page.setLayout(row)
        return page
    
    def _make_camera_widget(self, title_text, key):
        title = QLabel(title_text)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        img_label = QLabel()
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # placeholder image
        image = QImage(320, 240, QImage.Format_RGB888)
        image.fill(0xFFFFFF)
        img_label.setPixmap(QPixmap.fromImage(image))
        img_label.setScaledContents(True)

        v = QVBoxLayout()
        v.addWidget(title, alignment=Qt.AlignHCenter)
        v.addWidget(img_label)

        w = QWidget()
        w.setLayout(v)

        self.label[key] = title
        self.cam[key] = img_label
        return w

    # === Rendering helpers (kept close to yours) ===
    def get_camera(self, view_name, pos):
        eye_offset = self.camera_offsets[view_name]["eye"]
        target_offset = self.camera_offsets[view_name]["target"]
        return self.camera_view(pos, eye_offset, target_offset)

    def camera_view(self, pos, eye_offset, target_offset):
        eye = np.array(pos) + np.array(eye_offset)
        target = np.array(pos) + np.array(target_offset)
        return eye.tolist(), target.tolist()

    # === Update just one agent's three views ===
    def update_views(self, pos, team_name, agent_num):
        """
        Render top/side/front once and push to the corresponding labels for this agent.
        Only the visible page shows them, but you can call this regardless.
        """
        def render(eye, target):
            view = p.computeViewMatrix(eye, target, [0, 0, 1])
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
            _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj)
            img = np.reshape(rgb, (240, 320, 4))[:, :, :3]
            h, w, c = img.shape
            bpl = c * w
            return QImage(img.tobytes(), w, h, bpl, QImage.Format_RGB888)

        # compute all three images once
        top_eye, top_target = self.get_camera("top", pos)
        side_eye, side_target = self.get_camera("side", pos)
        front_eye, front_target = self.get_camera("front", pos)

        q_top   = render(top_eye, top_target)
        q_side  = render(side_eye, side_target)
        q_front = render(front_eye, front_target)

        # set them on the specific agent's widgets
        keys = {
            "top":   f"{team_name}_{agent_num}_top",
            "side":  f"{team_name}_{agent_num}_side",
            "front": f"{team_name}_{agent_num}_front",
        }
        if keys["top"] in self.cam:
            self.cam[keys["top"]].setPixmap(QPixmap.fromImage(q_top))
        if keys["side"] in self.cam:
            self.cam[keys["side"]].setPixmap(QPixmap.fromImage(q_side))
        if keys["front"] in self.cam:
            self.cam[keys["front"]].setPixmap(QPixmap.fromImage(q_front))
    # Old code
    # def add_agent(self,name,team):
    #     views = {"Top","Side","Front"}
    #     # when adding an agent add a new item to the drop down list
    #     for idx, agent in enumerate(team.agents):
    #         self.drop_down[name].addItem(f"{name} Agent {idx}")
    #         team_layout = QVBoxLayout()
    #         team_layout.addWidget(self.drop_down[name])
    #         self.layout.addLayout(team_layout)
    #         cam_layout = QHBoxLayout()
    #         for view in views:
    #             cam_layout = self.add_camera(name=view,origin=None,target=None, cam_layout=cam_layout, agent=idx, team_name=name)
    #             self.layout.addLayout(cam_layout)
            

    #     # self.container.setLayout(self.layout)
    #     # self.setCentralWidget(self.container)
    #     self.setLayout(self.layout)
  
    # def add_camera(self, name, origin, target, agent, team_name, cam_layout=QHBoxLayout()):
    #     unit_layout = QVBoxLayout()
    #     # Add view to camera offsets
    #     if name not in self.camera_offsets:
    #         self.camera_offsets[name] = {"eye": origin, "target": target}
        
    #     # title label
    #     self.label[f"{team_name}_{agent}_{name}"] = QLabel(f"{name} View")
    #     self.label[f"{team_name}_{agent}_{name}"].setFont(QFont("Arial", 12, QFont.Bold))
        
    #     # image lable
    #     self.cam[f"{team_name}_{agent}_{name}"] = QLabel()
    #     image = QImage(240, 320, QImage.Format_RGB888)
    #     image.fill(0xFFFFFF)  # White color
    #     self.cam[f"{team_name}_{agent}_{name}"].setPixmap(QPixmap.fromImage(image))
    #     self.cam[f"{team_name}_{agent}_{name}"].setScaledContents(True)  # Scale the image to fit the label

    #     unit_layout.addWidget(self.label[f"{team_name}_{agent}_{name}"])
    #     unit_layout.addWidget(self.cam[f"{team_name}_{agent}_{name}"])
    #     # self.cam.append(QLabel())

    #     container = QWidget()
    #     container.setLayout(unit_layout)
    #     cam_layout.addWidget(container)

    #     # self.layout.addLayout(cam_layout)
    #     # self.setLayout(self.layout)
        
    #     return cam_layout

    # def __del__(self):
    #     print(f"{ph.RED}QLabel (cam1) deleted{ph.RESET}")

    # def get_camera(self, view_name, pos):
    #     eye_offset = self.camera_offsets[view_name]["eye"]
    #     target_offset = self.camera_offsets[view_name]["target"]
    #     return self.camera_view(pos, eye_offset, target_offset)
    
    # def camera_view(self, pos, eye_offset, target_offset):
    #     eye = np.array(pos) + np.array(eye_offset)
    #     target = np.array(pos) + np.array(target_offset)
    #     return eye.tolist(), target.tolist()

    # def update_views(self, pos, team_name, agent_num):
    #     def render_camera(eye, target):
    #         view = p.computeViewMatrix(eye, target, [0, 0, 1])
    #         proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
    #         _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj)
    #         img = np.reshape(rgb, (240, 320, 4))[:, :, :3]
    #         return img
        
    #     result_cam   = [val for key, val in self.cam.items() if f"{team_name}_{agent_num}_" in key]
    #     result_label = [val for key, val in self.label.items() if f"{team_name}_{agent_num}_" in key]
    #     for cam, label in zip(result_cam, result_label):
    #         # eye, target = self.get_camera(label, pos)
        
    #         top_eye, top_target = self.get_camera("top", pos)
    #         side_eye, side_target = self.get_camera("side", pos)
    #         front_eye, front_target = self.get_camera("front", pos)
        
    #         top_img  = render_camera(top_eye, top_target)
    #         side_img = render_camera(side_eye, side_target)
    #         front_img = render_camera(front_eye, front_target)

    #         def to_qimage(img):
    #             height, width, channels = img.shape
    #             bytes_per_line = channels * width
    #             return QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            
    #         cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
    #         cam.setPixmap(QPixmap.fromImage(to_qimage(side_img)))
    #         cam.setPixmap(QPixmap.fromImage(to_qimage(front_img)))
    #     # for cam in self.cam:
    #     #         cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
    #     # self.top_cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
    #     # self.side_cam.setPixmap(QPixmap.fromImage(to_qimage(side_img)))
    #     # self.front_cam.setPixmap(QPixmap.fromImage(to_qimage(front_img)))
