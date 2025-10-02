import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
import json
import os, sys
from sim.Environment import environment as ENV
from sim.Agent import agent as AGENT
from sim.Agent import team as TEAM
from sim.Sensor import sensor as SENSOR
from gui.gui import CameraViewer
import time
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
import sim.print_helpers as ph

# class CameraViewer(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Multi-Camera Viewer")

#         self.cam = []
#         self.label = []
#         self.layout = QVBoxLayout()

#         # self.top_cam_label  = QLabel("Top View")
#         # self.top_cam_label.setFont(QFont("Arial", 12, QFont.Bold))
#         # self.top_cam = QLabel()

#         # self.side_cam_label = QLabel("Side View")
#         # self.side_cam_label.setFont(QFont("Arial", 12, QFont.Bold))
#         # self.side_cam = QLabel()

#         # self.front_cam_label = QLabel("Front View")
#         # self.front_cam_label.setFont(QFont("Arial", 12, QFont.Bold))
#         # self.front_cam = QLabel()

#         # top_cam_layout = QVBoxLayout()
#         # top_cam_layout.addWidget(self.top_cam_label)
#         # top_cam_layout.addWidget(self.top_cam)

#         # side_cam_layout = QVBoxLayout()
#         # side_cam_layout.addWidget(self.side_cam_label)
#         # side_cam_layout.addWidget(self.side_cam)

#         # front_cam_layout = QVBoxLayout()
#         # front_cam_layout.addWidget(self.front_cam_label)
#         # front_cam_layout.addWidget(self.front_cam)

#         # self.layout = QVBoxLayout()
#         # self.layout.addLayout(top_cam_layout)
#         # self.layout.addLayout(side_cam_layout)
#         # self.layout.addLayout(front_cam_layout)
#         # self.setLayout(self.layout)

#         self.camera_offsets = {
#             "top"  :     {"eye": [0, 0.1, 1], "target": [0, 0, 0]},
#             "side" :     {"eye": [0, 1, 0], "target": [0, 0, 0]},
#             "rear" :     {"eye": [-10, 0, 5], "target": [0, 0, 0]},
#             "front":     {"eye": [0.75, 0, 0], "target": [1, 0, -0.1]},
#         }
    
#     def add_camera(self,name,origin,target):

#         # Add view to camera offsets
#         if name not in self.camera_offsets:
#             self.camera_offsets[name] = {"eye": origin, "target": target}

#         self.label.append(QLabel(f"{name} View"))
#         self.label[-1].setFont(QFont("Arial", 12, QFont.Bold))
#         self.cam.append(QLabel())

#         cam_layout = QVBoxLayout()
#         cam_layout.addWidget(self.label[-1])
#         cam_layout.addWidget(self.cam[-1])

#         self.layout.addLayout(cam_layout)
#         self.setLayout(self.layout)

#     def __del__(self):
#         print(f"{ph.RED}QLabel (cam1) deleted{ph.RESET}")

#     def get_camera(self, view_name, pos):
#         eye_offset = self.camera_offsets[view_name]["eye"]
#         target_offset = self.camera_offsets[view_name]["target"]
#         return self.camera_view(pos, eye_offset, target_offset)
    
#     def camera_view(self, pos, eye_offset, target_offset):
#         eye = np.array(pos) + np.array(eye_offset)
#         target = np.array(pos) + np.array(target_offset)
#         return eye.tolist(), target.tolist()

#     def update_views(self, pos):
#         def render_camera(eye, target):
#             view = p.computeViewMatrix(eye, target, [0, 0, 1])
#             proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
#             _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj)
#             img = np.reshape(rgb, (240, 320, 4))[:, :, :3]
#             return img
        
#         for cam, label in zip(self.cam,self.label):
#             eye, target = self.get_camera(label, pos)
        
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

#         for cam in self.cam:
#             cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
#         # self.top_cam.setPixmap(QPixmap.fromImage(to_qimage(top_img)))
#         # self.side_cam.setPixmap(QPixmap.fromImage(to_qimage(side_img)))
#         # self.front_cam.setPixmap(QPixmap.fromImage(to_qimage(front_img)))

class SensorSelection_Env(gym.Env):
    def __init__(self, config_file=""):   
        super().__init__()
        with open(config_file, "r") as f:
            config_data = json.load(f)
        self.config = config_data

        self.render_mode = self.config.get("render", False)
        self.do_plots    = self.config.get("do_plots", False)
        self.time_limit  = self.config.get("time_limit", 30.0)
        self.save_plots  = self.config.get("save_plots", False)
        self.seed        = self.config.get("seed", 42)
        self.sim_dt      = self.config.get("sim_dt", 0.001)
        np.random.seed(self.seed)


        
        # time.sleep(5)

        # self.physics_client = p.connect(p.DIRECT) # Turn off the GUI until the environment is loaded
        self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        # Load Terrain.obj
        with open(self.config['scene']['terrain']) as t:
            terrain_data = json.load(t)
        self.terrain = terrain_data
        
        # Action space: discrete 4-directional movement
        self.action_space = spaces.Discrete(4)
        # Observation space: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)

        self.blue_team = TEAM.Team(team_name="blue",config=self.config['blue_agent'])
        self.red_team = TEAM.Team(team_name="red",config=self.config['red_agent'])

        self.goal_pos = np.zeros(2)

        # Load Camera Views
        self.init_cameras()

    def run_sim(self,model):
        model.learn(total_timesteps=10000)
        model.save("ppo_pybullet_goal")

        # Test the trained model
        obs, _ = self.reset()
        total_reward = 0
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = self.step(action)
            total_reward += reward
            if done:
                print(f"✅ Goal reached! Total reward: {total_reward:.2f}")
                break
        self.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        # Freeze the GUI Rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # time.sleep(2)

        # Enable Shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        """ Load the Terrain"""
        # base_dir = os.getcwd()
        # terrain_file_name = os.path.join(base_dir,os.path.normpath(self.terrain["Layered Surface"]["Mesh"]))
        
        self.env = ENV.Environment(self.terrain)
        # Unfreeze the GUI
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        """ Load the agents"""
        for agent in self.blue_team.agents:
            agent._reset_states(terrain_bound=self.env.terrain['terrain_bounds'][0,:],physicsClient=self.physics_client,team=self.blue_team.team_color)
            
            
        for agent in self.red_team.agents:
            agent._reset_states(terrain_bound=self.env.terrain['terrain_bounds'][0,:],physicsClient=self.physics_client,team=self.red_team.team_color)
        
        """ Set Camera """
        # Set initial camera parameters
        self.camera_distance = 2
        self.camera_yaw = 0
        self.camera_pitch = -30
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=tuple(self.blue_team.agents[0].position.T.tolist()[0]))
        # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=self.blue_agent[0].position.T)
        if self.camera_viewer["blue"].isVisible() and self.camera_viewer["red"].isVisible():
            self.camera_viewer["blue"].show()
            self.camera_viewer["red"].show()
        self.goal_pos = np.random.uniform(-5, 5, size=2)
        return self._get_observation(), {}
        # return {}

    def _get_observation(self):
        pos, _ = p.getBasePositionAndOrientation(self.blue_team.agents[0].id, physicsClientId=self.physics_client)
        return np.array([pos[0], pos[1], pos[2], self.goal_pos[0], self.goal_pos[1]], dtype=np.float32)

    def step(self, action):
        agent_pos, _ = p.getBasePositionAndOrientation(self.blue_team.agents[0].id, physicsClientId=self.physics_client)
        self.blue_team.agents[0].position[:] = np.array(agent_pos).reshape(3,1)
        
        # Interpret action
        dx, dy = 0, 0
        if action == 0: dy = 1   # forward
        elif action == 1: dy = -1 # backward
        elif action == 2: dx = -1 # left
        elif action == 3: dx = 1  # right

        # Move agent
        p.resetBaseVelocity(self.blue_team.agents[0].id, linearVelocity=[dx, dy, 0], physicsClientId=self.physics_client)
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=agent_pos)
        p.stepSimulation(physicsClientId=self.physics_client)
        if self.camera_viewer["blue"].isVisible() and self.camera_viewer["red"].isVisible():
            for idx, agent in enumerate(self.blue_team.agents):
                pos, _ = p.getBasePositionAndOrientation(agent.id, physicsClientId=self.physics_client)
                self.camera_viewer["blue"].update_views(pos=pos, team_name="blue", agent_num=idx)

            for idx, agent in enumerate(self.red_team.agents):
                pos, _ = p.getBasePositionAndOrientation(agent.id, physicsClientId=self.physics_client)
                self.camera_viewer["red"].update_views(pos=pos, team_name="red", agent_num=idx)
        time.sleep(self.sim_dt)

        obs = self._get_observation()
        agent_pos = obs[:2]
        distance = np.linalg.norm(agent_pos - self.goal_pos)
        reward = -distance
        terminated = distance < 0.5
        truncated = False

        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)


    def init_cameras(self):
        self.app = QApplication(sys.argv)
        self.camera_viewer = {"blue": CameraViewer(), "red": CameraViewer()}
        self.camera_viewer["blue"].add_team(self.blue_team,self.blue_team.team)
        self.camera_viewer["red"].add_team(self.red_team,self.red_team.team)
        self.camera_viewer["blue"].show()
        self.camera_viewer["red"].show()
    
    # def add_agent_views(self):
    #     # Put these in the teams class
    #     self.app = QApplication(sys.argv)
    #     self.camera_viewer = CameraViewer()
    #     self.camera_viewer.show()
    #     """ Blue Team Camera """
    #     # Top View
    #     self.camera_viewer.add_camera("top",None,None)
    #     # Side View
    #     self.camera_viewer.add_camera("side",None,None)
    #     # Front View
    #     self.camera_viewer.add_camera("front",None,None)


    #     """ Red Team Camera """
    #     # Top View
    #     self.camera_viewer.add_camera("top",None,None)
    #     # Side View
    #     self.camera_viewer.add_camera("side",None,None)
    #     # Front View
    #     self.camera_viewer.add_camera("front",None,None)


