from typing import Tuple
from abc import ABC
import numpy as np


class Advantage(ABC):
    """
    A generic class for calculating advantages given rewards, state-values
    and episode-end signals.
    """

    NAME = 'Advantage'

    @staticmethod
    def calc_advantage_and_returns(
            rewards: np.array,
            vs: np.array,
            vs_prime: np.array,
            dones: np.array,
            gamma: float,
            *args,
            **kwargs
    ) -> Tuple[np.array, np.array]:
        """
        Calculates advantages and returns.

        Parameters
        ----------
        rewards: np.array
            1d vector of rewards.
        vs: np.array
            State-values v(s)

        vs_prime: np.array
            Next state values v(s')

        dones: np.array
            Is episode done in a given timestep?

        gamma: float
            Disount factor

        Returns
        -------
        Tuple[np.array, np.array]
            A tuple of advantages (N,) and returns (N,)
        """
        raise NotImplementedError


class GAE(Advantage):

    NAME = 'GAE'

    @staticmethod
    def calc_advantage_and_returns(
            rewards: np.array,
            vs: np.array,
            vs_prime: np.array,
            dones: np.array,
            gamma: float,
            lambda_: float = 0.9,
            normalize: bool = True,
            norm_eps: float = 1e-4,
            *args,
            **kwargs) -> Tuple[np.array, np.array]:
        """
        Calcualtes advantages and returns.

        Parameters
        ----------
        rewards: np.array
            1d vector of rewards.
        vs: np.array
            State-values v(s)

        vs_prime: np.array
            Next state values v(s')

        dones: np.array
            Is episode done in a given timestep?

        gamma: float
            Disount factor

        lambda_: float
            A factor for GAE discounting.

        normalize: bool
            Should advantages be normalized? It can reduce variance in certain
            tasks.

        norm_eps: float
            Normalizing constant

        Returns
        -------
        Tuple[np.array, np.array]
            A tuple of advantages (N,) and returns (N,)

        References
        -------
        [1] https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
            Fantastic explanation and tutorial code, that inspired this implementation.

        [2] Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016).
            High-dimensional continuous control using generalized advantage estimation.
            4th International Conference on Learning Representations, ICLR 2016 - Conference Track Proceedings.
        """
        if rewards.ndim != 1 or vs.ndim != 1 or dones.ndim != 1:
            raise ValueError(
                f"Adv. estimation vectors should have 1 dim. Encuontered: r:{rewards.shape}, vs:{vs.shape}, d:{dones.shape}"
            )
        N = rewards.shape[0]

        # additional advantage = 0 for the terminal/last state
        advantage = np.zeros(N + 1)

        for t in reversed(range(N)):
            delta = rewards[t] + (gamma * vs_prime[t] * (1 - dones[t])) - vs[t]
            advantage[t] = delta + (gamma * lambda_ * advantage[t + 1] * (1 - dones[t]))

        if normalize:
            advantage[:N] = (advantage[:N] - np.mean(advantage[:N])) / (advantage[:N].std() + 1e-6)

        value_target = advantage[:N] + vs

        return advantage[:N], value_target


class BaselineAdvantage(Advantage):

    NAME = 'Baseline'

    @staticmethod
    def calc_advantage_and_returns(
            rewards: np.array,
            vs: np.array,
            vs_prime: np.array,
            dones: np.array,
            gamma: float,
            *args,
            **kwargs) -> Tuple[np.array, np.array]:
        if rewards.ndim != 1 or vs.ndim != 1 or dones.ndim != 1:
            raise ValueError(
                f"Adv. estimation vectors should have 1 dim. Encuontered: r:{rewards.shape}, vs:{vs.shape}, d:{dones.shape}"
            )
        N = rewards.shape[0]
        running_discount = 0.0
        G = np.zeros_like(rewards)
        for idx in reversed(range(N)):
            r = rewards[idx]
            running_discount = r + gamma * running_discount
            G[idx] = running_discount

        advantages = G - vs
        return advantages, G

    def __str__(self):
        return "Baseline"

