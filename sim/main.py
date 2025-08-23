import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

class PyBulletGoalEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)

        # Action space: discrete 4-directional movement
        self.action_space = spaces.Discrete(4)
        # Observation space: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)

        self.agent = None
        self.goal_pos = np.zeros(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        self.agent = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.1], physicsClientId=self.physics_client)
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


if __name__ == "__main__":
    env = PyBulletGoalEnv(render=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_pybullet_goal")

    # Test the trained model
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done:
            print(f"✅ Goal reached! Total reward: {total_reward:.2f}")
            break
    env.close()
