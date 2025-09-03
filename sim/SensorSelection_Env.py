import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
import json
import os
from sim.Environment import environment as ENV
from sim.Agent import agent as AGENT
from sim.Sensor import sensor as SENSOR
import time

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
        np.random.seed(self.seed)

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

        self.blue_agent = [AGENT.Agent(agent) for agent in self.config['blue_agent'].values()]
        self.red_agent = [AGENT.Agent(agent) for agent in self.config['red_agent'].values()]

        self.goal_pos = np.zeros(2)

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
        
        # Enable Shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        """ Load the Terrain"""
        base_dir = os.getcwd()
        terrain_file_name = os.path.join(base_dir,os.path.normpath(self.terrain["Layered Surface"]["Mesh"]))
        self.env = ENV.Environment(terrain_file_name)

        texture_file_name = os.path.join(base_dir,os.path.normpath(self.terrain["Layered Surface"]["Texture"]))
        tex_id = p.loadTexture(texture_file_name)
        p.changeVisualShape(self.env.terrain, -1, textureUniqueId=tex_id)
        p.changeVisualShape(self.env.terrain, -1, rgbaColor=[1,1,1,1], specularColor=[0.1,0.1,0.1])


        """ Load the agents"""
        for agent in self.blue_agent:
            agent._reset_states(terrain_bound=self.env.terrain_bounds,physicsClient=self.physics_client)
            # agent._load_file()
            
        for agent in self.red_agent:
            agent._reset_states(terrain_bound=self.env.terrain_bounds,physicsClient=self.physics_client)
        
        """ Set Camera """
        # Set initial camera parameters
        self.camera_distance = 10
        self.camera_yaw = 50
        self.camera_pitch = -30
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=self.blue_agent[0].position.T)
        # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=self.blue_agent[0].position.T)
        self.goal_pos = np.random.uniform(-5, 5, size=2)
        return self._get_observation(), {}
        # return {}

    def _get_observation(self):
        pos, _ = p.getBasePositionAndOrientation(self.blue_agent[0].id, physicsClientId=self.physics_client)
        return np.array([pos[0], pos[1], pos[2], self.goal_pos[0], self.goal_pos[1]], dtype=np.float32)

    def step(self, action):
        agent_pos, _ = p.getBasePositionAndOrientation(self.blue_agent[0].id, physicsClientId=self.physics_client)
        self.blue_agent[0].position[:] = np.array(agent_pos).reshape(3,1)
        
        # Interpret action
        dx, dy = 0, 0
        if action == 0: dy = 20   # forward
        elif action == 1: dy = -20 # backward
        elif action == 2: dx = -20 # left
        elif action == 3: dx = 20  # right

        # Move agent
        p.resetBaseVelocity(self.blue_agent[0].id, linearVelocity=[dx, dy, 0], physicsClientId=self.physics_client)
        p.resetDebugVisualizerCamera(cameraDistance=self.camera_distance, cameraYaw=self.camera_yaw, cameraPitch=self.camera_pitch, cameraTargetPosition=agent_pos)
        p.stepSimulation(physicsClientId=self.physics_client)
        time.sleep(1./240.)

        obs = self._get_observation()
        agent_pos = obs[:2]
        distance = np.linalg.norm(agent_pos - self.goal_pos)
        reward = -distance
        terminated = distance < 0.5
        truncated = False

        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)
