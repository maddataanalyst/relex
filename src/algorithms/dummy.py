import gym
import numpy as np

import src.algorithms.commons as algo_common

from typing import Tuple


class DummyAgent(algo_common.RLAgent):

    """
    A dummy agent to be used as a baseline.
    """

    def __init__(self, env: gym.Env, name:str = 'dummy'):
        self.env = env
        self.name = name

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        a = self.env.action_space.sample()
        a = np.array(a).reshape(self.env.action_space.shape)
        return a, None
