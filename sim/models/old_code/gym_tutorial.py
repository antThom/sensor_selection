import gym
from gym import spaces
import pygame
import numpy as np
import os
import time as t


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"],"render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_preclude=0):
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

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

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
    
    def step(self, actions):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = {}
        direction['agent'] = self._action_to_direction[actions['agent']]
        direction['target'] = self._action_to_direction[actions['target']]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction['agent'], 0, self.size - 1
        )

        self._target_location = np.clip(
            self._target_location + direction['target'], 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
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

class BlueAgent:
    def __init__(self,learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, size=4, action_set=4):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.states = []
        self.actions = range(action_set)
        self.q_table = np.zeros((size, size, 4))
        # self.env = GridWorldEnv()

    def choose_action(self,state):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exploration_rate:
            # Explore
            action = np.random.choice(self.actions)
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
    def __init__(self, action_set=4, size=4):
        self.states = []
        self.actions = range(action_set)
        self.world_size = size
    
    def choose_action(self,state):
        # choose action with most expected value
        action = np.random.choice(self.actions)
        return action
    
if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = '100,50'
    grid_size = 5
    num_preclude = 10
    render_mode = 'human'
    env = GridWorldEnv(render_mode=render_mode,size=grid_size,num_preclude=num_preclude)
    
    agent  = BlueAgent(size=grid_size,exploration_rate=0.5,action_set=4)
    target = RedAgent(action_set=4)

    # Training
    episodes = 1
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        actions = {}
        time = 0.0
        dt   = 0.1

        while not done:
            env.render()
            print(f"Time: {time}")
            actions['agent'] = agent.choose_action(state) # Choose an action
            actions['target'] = target.choose_action(state) # Choose an action
            print(f"\tAction: {env._action_to_direction[actions['agent']]}")
            print(f"\tPre-State: {state['agent']}")
            next_state, reward, done, info = env.step(actions)  # Take the action and observe next state, reward
            print(f"\tPost-State: {next_state['agent']}")
            agent.update_q_value(state, actions['agent'], reward, next_state) # Update Q-values
            state = next_state
            time += dt
            t.sleep(1)