import os, subprocess, time, signal
import gym
import numpy as np  # Erco
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

log = logging.getLogger("miniproject_snake")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=[15,15], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.snake_size = snake_size
        self.n_snakes = n_snakes
        self.current_snake = 0  # Erco: turn-based multi-player snake
        self.n_foods = n_foods
        self.viewer = None
        self.action_space = spaces.Discrete(4)  # Erco: replaced Discrete(4) by spaces.Discrete(4); needed to use stable-baselines
        # Erco: coordinate-based state space; added the complete line below; this is needed to use stable-baselines
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.grid_size[1], self.grid_size[0]), dtype=np.uint8)
        self.random_init = random_init

    # Erco: added a function that translates state in pixels to state in coordinates
    def to_coord(self, state):
        state = state[0::self.unit_size]
        state = [ state[i][0::self.unit_size] for i in range(len(state)) ]
        coordstate = np.zeros(self.grid_size)
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j][1] == 255:  
                    coordstate[i][j] = 0  # space
                elif state[i][j][0] == 1:  
                    coordstate[i][j] = 1  # body
                elif state[i][j][2] == 255:  
                    coordstate[i][j] = 3  # food
                elif state[i][j][0] == 255:  
                    coordstate[i][j] = 2  # head
                else:
                    assert True, "unexpected state"
        return coordstate

    def step(self, action):
        # the action passed to step if for the "current" snake; change it to a list with -1 (no-move) for the other snakes
        if self.n_snakes > 1:
            actions = np.ones(self.n_snakes) * -1  # Erco: -1 is NOMOVE
            actions[self.current_snake] = action  # Erco
            action = actions.tolist()  # Erco
            self.current_snake += 1  # Erco
            self.current_snake %= self.n_snakes
        self.last_obs, rewards, done, info = self.controller.step(action)
        coord_obs = self.to_coord(self.last_obs)  # Erco
        return coord_obs, rewards, done, info  # Erco

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        coord_obs = self.to_coord(self.last_obs)  # Erco
        return coord_obs  # Erco

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        pass
