from abc import ABC

import gym
import numpy as np
import logging
import abc
from dataclasses import dataclass
from typing import List, Tuple
from time import time

@dataclass
class TimeCheckResult:
    should_stop_training: bool
    should_break_loop: bool


def check_train_time(t0: float, max_train_sec: int) -> TimeCheckResult:
    t_now = time()
    delta_t = t_now - t0
    if delta_t >= max_train_sec:
        return TimeCheckResult(True, True)
    else:
        return TimeCheckResult(False, False)


class ActionClipper(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def clip_action(action: np.array, env: gym.wrappers.TimeLimit, *args, **kwargs):
        raise NotImplementedError


class DefaultActionClipper(ActionClipper):

    @staticmethod
    def clip_action(action: np.array, env: gym.wrappers.TimeLimit, *args, **kwargs):
        a_clip = np.clip(action, env.action_space.low, env.action_space.high)
        return a_clip


class RLAgent:
    """
    A base class that defines an interface for all concrete implementations
    """

    def __init__(self, action_clipper: ActionClipper, name: str = "Agent"):
        self.action_clipper = action_clipper
        self.name = name

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        """
        A base function for choosing an action, given state and (potentially) other parameters.

        Parameters
        ----------
        s: np.array
            State

        Returns
        -------
        Tuple[np.array, np.array]
            A tuple representing: (action to be taken: np.array, action logprob: np.array)
        """
        raise NotImplementedError("Implement me")

    def learn(
            self,
            s: np.array,
            a: np.array,
            a_logprob: np.array,
            r: np.array,
            sprime: np.array,
            done: np.array,
            vs: np.array,
            v_sprime: np.array,
            gamma: float,
            learning_step: int,
            *args,
            **kwargs) -> float:
        """
        Learning step function.

        Parameters
        ----------
        s: np.array
            States of environment.

        a: np.array
            Actions taken in the env.

        a_logprob: np.array
            Logprob(a).

        r: np.array
            Reward signal.

        sprime: np.array
            Next states after transition s--->a--->sprime.

        done: np.array
            Is episode done after transition?

        vs: np.array
            V(s) - state value.

        v_sprime: np.array
            V(sprime) - sprime value.

        gamma: float
            Discounting parameter

        learning_step: int
            Learning iteration step.

        Returns
        -------
        float
            Total oss after learning step.
        """
        raise NotImplementedError("Implement me")

    def train(self,
              env: gym.wrappers.TimeLimit,
              nepisodes: int,
              gamma: float = 0.9,
              max_steps: int = 500,
              max_train_sec: int = 900,
              print_interval: int = 10,
              average_n_last: int = 30,
              scaler: object = None,
              epochs: int = 3,
              clip_action: bool = True,
              batch_size: int = 32,
              log: logging.Logger = logging.getLogger("ppo_logger"),
              *args,
              **kwargs) -> np.array:
        """
        Training function for RL Agent. Runs it against the specified environment.

        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            Env to run against.

        nepisodes: int
            Total number of episodes to run.

        gamma: float
            Discounting factor.

        max_steps: int
            Max. number of steps per episode.

        max_train_sec: int
            Max total training time. Terminates training after the deadline.

        print_interval: int
            Print every n iterations.

        average_n_last: int
            How many recent rewards to average in log.

        scaler: object
            Scaling object.

        epochs: int
            Total number of epochs for neural net training.

        clip_action: bool
            Should actions be clipped?

        batch_size: int
            Batch size to train NN.

        log: logging.Logger
            Logging object.

        Returns
        -------
        np.array
            Episode scores.
        """
        raise NotImplementedError("Implement me")

    def make_step(self, s: np.array, env: gym.wrappers.TimeLimit, clip_action: bool, scaler: object) -> Tuple[np.array, float, bool, float, np.array]:
        """
        Performs a single transition step.

        Parameters
        ----------
        clip_action: bool
            Should action be clipped to meet env requirements?

        env: gym.wrappers.TimeLimit
            Env to operate on

        s: np.array
            Current state

        scaler: object
            Scaler object

        Returns
        -------
        Tuple[np.array, float, bool, float, np.array]
            A tuple of a, logprob(a), done, r, sprime

        """
        import src.algorithms.algo_utils as autils

        a, a_logprob = self.choose_action(s)
        a_clip = self.action_clipper.clip_action(a, env) if clip_action else a
        sprime, r, done, _ = env.step(np.atleast_1d(a_clip)) if env.action_space.shape != () else env.step(a_clip)
        sprime = np.atleast_1d(autils.normalize_state(sprime, scaler))
        return a, a_logprob, done, r, sprime

