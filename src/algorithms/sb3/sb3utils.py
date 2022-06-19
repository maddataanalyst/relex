import numpy as np
from typing import Tuple
from src.algorithms.commons import RLAgent, ActionClipper


class BaselinesWrapper(RLAgent):
    """
    A wrapper class for Stable baselines 3 agent
    """

    def __init__(self, baselines_agent, action_clipper: ActionClipper=None):
        super().__init__(action_clipper)
        self.agent = baselines_agent

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        return self.agent.predict(s, *args, **kwargs)
