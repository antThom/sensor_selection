import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
# from matplotlib import pyplot as plt
import json
import os, sys
from sim.Environment import environment as ENV
from sim.Agent import agent as AGENT
from sim.Agent import team as TEAM
from sim.Sensor import sensor as SENSOR
from sim.Sensor.Microphone.microphone import MicrophoneSensor_Uniform
from sim.Environment.Thermal.thermal_manager import ThermalManager
from sim.Sound.audio_mixer import AudioMixer
from gui.gui import CameraViewer
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QApplication
import sim.print_helpers as ph

class SensorSelection_Env(gym.Env):
    def __init__(self, config_file=""):   
        super().__init__()
        with open(config_file, "r") as f:
            config_data = json.load(f)
        self.config = config_data

        self.render_mode  = self.config.get("render", False)
        self.do_plots     = self.config.get("do_plots", False)
        self.time_limit   = self.config.get("time_limit", 30.0)
        self.save_plots   = self.config.get("save_plots", False)
        self.seed         = self.config.get("seed", 42)
        self.sim_dt       = self.config.get("sim_dt", 0.001)
        self.ambient_temp = self.config.get("ambient_temp", 293.0)
        self.sky_temp     = self.config.get("sky_temp", 260.0)
        self.time_of_day  = self.config.get("time_of_day", 8)
        self.id           = []
        np.random.seed(self.seed)

        # self.physics_client = p.connect(p.DIRECT) # Turn off the GUI until the environment is loaded
        self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        # Freeze the GUI Rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.sim_dt)

        # Load Terrain.obj
        with open(self.config['scene']['terrain']) as t:
            terrain_data = json.load(t)
        self.terrain = terrain_data

        # --- Initialize Thermal Manager ---
        self.sim_time = datetime(2025, 5, 24, self.time_of_day, 0, 0)
        self.thermal_manager = ThermalManager(ambient_K=self.ambient_temp, T_sky=self.sky_temp, time_of_day=self.sim_time)
        
        """ Load the Terrain"""        
        self.env = ENV.Environment(self.terrain, time_of_day=12, thermal=self.thermal_manager)

        # --- Initialize AudioMixer ---
        self.audio_mixer = AudioMixer(sample_rate=44100, dt=self.sim_dt)

        # Create teams
        self.blue_team = TEAM.Team(team_name="blue", config=self.config['blue_agent'], thermal=self.thermal_manager)
        self.red_team = TEAM.Team(team_name="red", config=self.config['red_agent'], thermal=self.thermal_manager)

        # Add all sound sources to mixer
        for agent in self.blue_team.agents + self.red_team.agents:
            if hasattr(agent, "sound"):
                self.audio_mixer.add_source(agent.sound)
        
        # Give microphones access to the shared mixer
        for agent in self.blue_team.agents + self.red_team.agents:
            if getattr(agent, "has_sensor", False):
                for sensor in agent.sensors:
                    if isinstance(sensor, MicrophoneSensor_Uniform):
                        sensor.mixer = self.audio_mixer
                        # Start the threaded sampling loop
                        sensor.start_capture(rate_hz=1.0 / self.sim_dt)

                # Action space: discrete 4-directional movement
        num_sensors = self.blue_team.getNumSensors()
        sensor_combinations = 2**num_sensors[0] - 1
        self.action_space = spaces.MultiDiscrete([sensor_combinations] * self.blue_team.Num_agents)

        # Observation space: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
        
        """ Set Camera """
        # Set initial camera parameters
        self.camera_distance = 2
        self.camera_yaw = 0
        self.camera_pitch = -30
        
        # Enable Shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

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
        p.setRealTimeSimulation(0)
        
        """ Load the agents"""
        for agent in self.blue_team.agents:
            agent._reset_states(terrain_bound=self.env.terrain['terrain_bounds'][0,:],physicsClient=self.physics_client,team=self.blue_team.team_color)
            
            
        for agent in self.red_team.agents:
            agent._reset_states(terrain_bound=self.env.terrain['terrain_bounds'][0,:],physicsClient=self.physics_client,team=self.red_team.team_color)
        
        # Unfreeze the GUI
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=tuple(self.blue_team.agents[0].position.T.tolist()[0]))
        if self.camera_viewer["blue"].isVisible() and self.camera_viewer["red"].isVisible():
            self.camera_viewer["blue"].show()
            self.camera_viewer["red"].show()


        return self._get_observation(), {}

    def _get_observation(self):
        pos, _ = p.getBasePositionAndOrientation(self.blue_team.agents[0].agent_id, physicsClientId=self.physics_client)
        return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)

    def step(self, action):
        # Update Thermal
        irradiance = max(0, np.sin(2*np.pi*(self.sim_time.timestamp()))) 
        self.thermal_manager.update(self.sim_dt, irradiance)
        
        # Update audio field
        self.audio_mixer.update()

        # Step 1: Get agent states
        blue_states = self.blue_team.get_states(self.physics_client)
        red_states = self.red_team.get_states(self.physics_client)
        
        # p.addUserDebugLine(src.pos.tolist(), mic.agent.position.flatten().tolist(), [1, 0, 0], 2, 0.1)

        # Step 2: Select the sensor set
        self.blue_team.assignSenor(action)

        # Step 3: Choose the dynamics inputs
        
        # Step 4: Move the agent
        
        # Interpret action
        dx, dy = 0, 0
        if action == 0: dy = 1   # forward
        elif action == 1: dy = -1 # backward
        elif action == 2: dx = -1 # left
        elif action == 3: dx = 1  # right

        # Move agent
        p.resetBaseVelocity(self.blue_team.agents[0].agent_id, linearVelocity=[dx, dy, 0], physicsClientId=self.physics_client)
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=blue_states[0]['pos'])
        p.stepSimulation(physicsClientId=self.physics_client)
        if self.camera_viewer["blue"].isVisible() and self.camera_viewer["red"].isVisible():
            for idx, agent in enumerate(self.blue_team.agents):
                pos, _ = p.getBasePositionAndOrientation(agent.agent_id, physicsClientId=self.physics_client)
                self.camera_viewer["blue"].update_fixed_views(pos=pos, team_name="blue", agent_idx=idx)

            for idx, agent in enumerate(self.red_team.agents):
                pos, _ = p.getBasePositionAndOrientation(agent.agent_id, physicsClientId=self.physics_client)
                self.camera_viewer["red"].update_fixed_views(pos=pos, team_name="red", agent_idx=idx)
        self.sim_time += timedelta(seconds=self.sim_dt)

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
        self.camera_viewer["blue"].start_auto_refresh(interval_ms=500)
        self.camera_viewer["red"].start_auto_refresh(interval_ms=500)


    
    