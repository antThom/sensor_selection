# main.py
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from stable_baselines3 import PPO
import threading
from ursina import *


# --- Ursina Setup (Visualization Only) ---
app = Ursina(borderless=False)
camera.position = (0, 10, -20)
camera.rotation_x = 30
ursina_agent = Entity(model='cube', color=color.azure, scale=(1,1,1))
ursina_goal = Entity(model='sphere', color=color.red, scale=(1,1,1))


# --- PyBullet + Gym Environment ---
class UrsinaBulletEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action: forward, backward, left, right
        self.action_space = spaces.Discrete(4)
        # Observation: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self):
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # Load agent
        self.agent = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.1], physicsClientId=self.physics_client)
        self.goal_pos = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
        return self._get_obs()

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.agent, physicsClientId=self.physics_client)
        return np.array([pos[0], pos[1], self.goal_pos[0], self.goal_pos[1]], dtype=np.float32)

    def step(self, action):
        pos, _ = p.getBasePositionAndOrientation(self.agent, physicsClientId=self.physics_client)
        x, y = pos[0], pos[1]

        # Action mapping
        dx, dy = 0, 0
        if action == 0: dy = 0.2   # forward
        elif action == 1: dy = -0.2 # back
        elif action == 2: dx = -0.2 # left
        elif action == 3: dx = 0.2  # right

        p.resetBaseVelocity(self.agent, linearVelocity=[dx, dy, 0], physicsClientId=self.physics_client)
        p.stepSimulation(physicsClientId=self.physics_client)

        obs = self._get_obs()
        agent_pos = obs[:2]
        distance = np.linalg.norm(agent_pos - self.goal_pos)

        reward = -distance
        done = distance < 0.5
        return obs, reward, done, {}

    def render_to_ursina(self):
        # Sync pybullet agent pos -> ursina agent
        pos, _ = p.getBasePositionAndOrientation(self.agent, physicsClientId=self.physics_client)
        ursina_agent.position = (pos[0], 0.5, pos[1])
        ursina_goal.position = (self.goal_pos[0], 0.5, self.goal_pos[1])

    def close(self):
        p.disconnect(physicsClientId=self.physics_client)


# --- RL Training ---
env = UrsinaBulletEnv()

model = PPO("MlpPolicy", env, verbose=1)
training_done = False


def train():
    global training_done
    model.learn(total_timesteps=10000)
    training_done = True

# Train in a thread so Ursina doesn't freeze
threading.Thread(target=train).start()


# --- Ursina Main Loop ---
def update():
    if not training_done:
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        _, _, done, _ = env.step(action)
        if done:
            env.reset()
        env.render_to_ursina()

app.run()
