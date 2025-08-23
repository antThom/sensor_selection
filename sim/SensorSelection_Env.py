import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
import json
import os

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
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)

        self.agent = None
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
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        
        # Enable Shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        base_dir = os.getcwd()
        terrain_file_name = os.path.join(base_dir,os.path.normpath(self.terrain["Layered Surface"]["Mesh"]))
        header, height_data = self.read_asc_file(terrain_file_name)

        # Normalize / scale height
        height_data = height_data.astype(np.float32)
        height_scale = 1.0  # vertical exaggeration if needed

        # Flatten row-major
        heightfield_data = height_data.flatten()

        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[header["cellsize"], header["cellsize"], height_scale],
            heightfieldTextureScaling=header["ncols"] / 2,
            heightfieldData=heightfield_data,
            numHeightfieldRows=header["nrows"],
            numHeightfieldColumns=header["ncols"]
        )

        terrain = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0]
        )

        texture_file_name = os.path.join(base_dir,os.path.normpath(self.terrain["Layered Surface"]["Texture"]))
        tex_id = p.loadTexture(texture_file_name)
        p.changeVisualShape(terrain, -1, textureUniqueId=tex_id)
        p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1], specularColor=[0.1,0.1,0.1])


        p.resetDebugVisualizerCamera(cameraDistance=50, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        self.agent = p.loadMJCF("quadrotor.xml")[0]
        # self.agent = p.loadURDF("quadrotor.urdf", basePosition=[0, 0, -10], physicsClientId=self.physics_client)
        # self.target = p.loadURDF("quadrotor.urdf", basePosition=[100, 0, -10], physicsClientId=self.physics_client)
        self.goal_pos = np.random.uniform(-5, 5, size=2)
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.agent, physicsClientId=self.physics_client)
        return np.array([pos[0], pos[1], self.goal_pos[0], self.goal_pos[1]], dtype=np.float32)

    def step(self, action):
        # Interpret action
        dx, dy = 0, 0
        if action == 0: dy = 0.2   # forward
        elif action == 1: dy = -0.2 # backward
        elif action == 2: dx = -0.2 # left
        elif action == 3: dx = 0.2  # right

        # Move agent
        p.resetBaseVelocity(self.agent, linearVelocity=[dx, dy, 0], physicsClientId=self.physics_client)
        p.stepSimulation(physicsClientId=self.physics_client)

        obs = self._get_obs()
        agent_pos = obs[:2]
        distance = np.linalg.norm(agent_pos - self.goal_pos)
        reward = -distance
        terminated = distance < 0.5
        truncated = False

        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)

    def read_asc_file(self,filepath):
        with open(filepath, 'r') as f:
            header = {}
            for _ in range(6):
                key, value = f.readline().strip().split()
                header[key] = float(value) if '.' in value else int(value)

            data = np.loadtxt(f)

        return header, data