from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
import sim.print_helpers as ph
import os, sys
import time
import pybullet as p
import pybullet_data
import numpy as np

class CameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Viewer")

        self.cam = []
        self.label = []
        self.layout = QVBoxLayout()

        self.camera_offsets = {
            "top"  :     {"eye": [0, 0.1, 1], "target": [0, 0, 0]},
            "side" :     {"eye": [0, 1, 0], "target": [0, 0, 0]},
            "rear" :     {"eye": [-10, 0, 5], "target": [0, 0, 0]},
            "front":     {"eye": [0.75, 0, 0], "target": [1, 0, -0.1]},
        }
    
    def add_camera(self,name,origin,target):

        # Add view to camera offsets
        if name not in self.camera_offsets:
            self.camera_offsets[name] = {"eye": origin, "target": target}

        self.label.append(QLabel(f"{name} View"))
        self.label[-1].setFont(QFont("Arial", 12, QFont.Bold))
        self.cam.append(QLabel())

        cam_layout = QVBoxLayout()
        cam_layout.addWidget(self.label[-1])
        cam_layout.addWidget(self.cam[-1])

        self.layout.addLayout(cam_layout)
        self.setLayout(self.layout)

    def __del__(self):
        print(f"{ph.RED}QLabel (cam1) deleted{ph.RESET}")

    def get_camera(self, view_name, pos):
        eye_offset = self.camera_offsets[view_name]["eye"]
        target_offset = self.camera_offsets[view_name]["target"]
        return self.camera_view(pos, eye_offset, target_offset)
    
    def camera_view(self, pos, eye_offset, target_offset):
        eye = np.array(pos) + np.array(eye_offset)
        target = np.array(pos) + np.array(target_offset)
        return eye.tolist(), target.tolist()

    def update_views(self, pos):
        def render_camera(eye, target):
            view = p.computeViewMatrix(eye, target, [0, 0, 1])
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
            _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj)
            img = np.reshape(rgb, (240, 320, 4))[:, :, :3]
            return img
        
        for cam, label in zip(self.cam,self.label):
            eye, target = self.get_camera(label, pos)
        
        top_eye, top_target = self.get_camera("top", pos)
        side_eye, side_target = self.get_camera("side", pos)
        front_eye, front_target = self.get_camera("front", pos)
        
        top_img  = render_camera(top_eye, top_target)
        side_img = render_camera(side_eye, side_target)
        front_img = render_camera(front_eye, front_target)

        def to_qimage(img):
            height, width, channels = img.shape
            bytes_per_line = channels * width
            return QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)

        for cam in self.cam:
            cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
        # self.top_cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
        # self.side_cam.setPixmap(QPixmap.fromImage(to_qimage(side_img)))
        # self.front_cam.setPixmap(QPixmap.fromImage(to_qimage(front_img)))
