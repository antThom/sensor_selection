import gym
from gym import spaces
import pygame
import math
import numpy as np
import os
import time as t
from itertools import product


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"],"render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_preclude=0, num_sensor=1):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
    
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        
        # Define the number of preclusions and preclude regions locations
        self.num_preclude = num_preclude
        self._precluded_locations = []
        for _ in range(self.num_preclude):
            self._precluded_locations.append(np.array([-1, -1], dtype=np.int32))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # The Action Space is the number of sensor set to choose from
        self.action_space = spaces.Discrete(math.factorial(num_sensor))
        # self.action_space = spaces.Discrete(2**num_sensor-1)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_sensor_set = {}
        all_combo = self.combinations_of_three(min_val=0, max_val=1, num_sensor=num_sensor)
        for index, set in enumerate(all_combo):
            self._action_to_sensor_set[index] = set

        self._target_action_to_direction = {
            0: np.array([1, 0]),    # Left
            1: np.array([0, 1]),    # Up
            2: np.array([-1, 0]),   # Right
            3: np.array([0, -1]),   # Down
            4: np.array([1, 1]),    # 45 deg
            5: np.array([-1, 1]),   # 135 deg
            6: np.array([-1, -1]),  # 225 deg
            7: np.array([1, -1]),   # 315 deg
        }

        self.end_time = 30

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def combinations_of_three(self, min_val=0, max_val=1, num_sensor=1):
        """
        Generates all possible combinations of three integers within a given range.

        Args:
            min_val: The minimum integer value (inclusive).
            max_val: The maximum integer value (inclusive).

        Returns:
            A list of tuples, where each tuple represents a combination of three integers.
        """
        all_combo = list(product(range(min_val, max_val + 1), repeat=num_sensor))
        null_set = (0,)*num_sensor
        full_set = (1,)*num_sensor
        all_combo.remove(null_set)
        all_combo.remove(full_set)
        return all_combo

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # Choose the locations of the precluded regions
        for i in range(self.num_preclude):
            self._precluded_locations[i] = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, actions, time):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = {}
        direction['agent'] = self._action_to_sensor_set[actions['agent']]
        direction['target'] = self._target_action_to_direction[actions['target']]
        # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction['agent'], 0, self.size - 1
        # )

        self._target_location = np.clip(
            self._target_location + direction['target'], 0, self.size - 1
        )
        
        # An episode is done iff the target reaches the agent or after a set number of time_steps
        terminated = np.array_equal(self._agent_location, self._target_location) or time >= self.end_time
        
        observation = self._get_obs()
        info = self._get_info()

        reward = -info['distance']
        reward += 10 if terminated else 0  # Binary sparse rewards
        

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Now we draw the precluded regions
        for i in range(self.num_preclude):
            pygame.draw.rect(
                canvas,
                (0, 255, 0, 100),
                pygame.Rect(
                    pix_square_size * self._precluded_locations[i],
                    (pix_square_size, pix_square_size),
                ),
            )

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class KalmanFilterGrid:
    def __init__(self, grid_size, dt, process_noise_std, measurement_noise_std):
        self.grid_size = grid_size
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

        self.state_dim = 4
        self.measurement_dim = 2 # This number is variable

        # State transition matrix (predict step)
        self.F = np.array([[1, dt, 0, 0], 
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])

        # Measurement matrix (update step)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        # Process noise covariance matrix
        self.Q = np.eye(self.state_dim) * process_noise_std**2

        # Measurement noise covariance matrix
        self.R = np.eye(self.measurement_dim) * measurement_noise_std**2

        # Initial state estimate (mean and covariance)
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim)

    def predict(self):
        # Predict next state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
      # Compute Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state estimate and covariance
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.state_dim) - np.dot(K, self.H)), self.P)
        
    def get_filtered_grid(self):
        return np.floor(self.x[:2]), np.ceil(self.P.diagonal())
        # return self.x[:self.grid_size**2].reshape((self.grid_size, self.grid_size))

class BlueAgent:
    def __init__(self,env,learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, num_sensors=2, dt=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.states = []
        self.num_sensors = num_sensors
        self.actions = env.action_space
        self.num_actions = math.factorial(num_sensors)
        self.q_table = np.zeros((env.size, env.size, self.num_actions))
        self.KF = KalmanFilterGrid(grid_size=env.size, dt=dt, process_noise_std=1, measurement_noise_std=1)

    def choose_action(self,state):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        """ Get Estimate from the Kalman Filter """
        # Get Measurement
        self.measurement = (state['target'] + 2*np.random.randn(2)).reshape((2,1))
        # Predict
        self.KF.predict()
        # Update
        self.KF.update(self.measurement)
        # Use Grid Filter
        est_state = self.KF.get_filtered_grid()
        if np.random.uniform(0, 1) <= self.exploration_rate:
            # Explore
            action = np.random.choice(self.actions.n)
        else:
            # Exploit
            # greedy action
            if isinstance(state,dict):
                STATE = state['agent']
            action = np.argmax(STATE)
            # for a in self.actions:
                # if the action is deterministic
                # nxt_reward = self.state_values[self.State.nxtPosition(a)]
                # if nxt_reward >= mx_nxt_reward:
                #     action = a
                #     mx_nxt_reward = nxt_reward
        return action
    
    def get_action_map(self, action):
        return self.env._action_to_sensor_set[action]
    
    def update_q_value(self, state, action, reward, next_state):
        if isinstance(state, tuple):
            STATE = tuple(state[0]['agent'])
        else:
            STATE = tuple(state['agent'])

        max_future_q = np.max(self.q_table[STATE])  # Best Q-value for next state
        
        current_q = self.q_table[STATE][action]
        # Q-learning formula
        self.q_table[STATE][action] = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
    
class RedAgent:    
    def __init__(self, env, action_set=4):
        self.env = env
        self.states = []
        self.actions = range(action_set)
        self.world_size = env.size
    
    def choose_action(self,state):
        # choose action with most expected value
        action = np.random.choice(self.actions)
        return action
    
    def get_action_map(self, action):
        return self.env._target_action_to_direction[action]
    
if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,50'
    dt = 0.1
    grid_size = 5
    num_preclude = 10
    num_sensors  = 3
    render_mode = 'human'
    env = GridWorldEnv(render_mode=render_mode,size=grid_size,num_preclude=num_preclude,num_sensor=num_sensors)
    
    agent  = BlueAgent(env=env,exploration_rate=0.5, num_sensors=num_sensors, dt=dt)
    target = RedAgent(env=env, action_set=4)

    # Training
    episodes = 1
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        actions = {}
        time = 0.0

        while not done:
            env.render()
            print(f"Time: {time}")
            actions['agent'] = agent.choose_action(state) # Choose an action
            actions['target'] = target.choose_action(state) # Choose an action
            print(f"\tAction: {env._action_to_sensor_set[actions['agent']]}")
            print(f"\tPre-State: {state['agent']}")
            next_state, reward, done, info = env.step(actions, time)  # Take the action and observe next state, reward
            print(f"\tPost-State: {next_state['agent']}")
            agent.update_q_value(state, actions['agent'], reward, next_state) # Update Q-values
            state = next_state
            time += dt
            t.sleep(1)