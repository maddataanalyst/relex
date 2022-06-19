from random import sample

import pandas as pd
import numpy as np
import gym
from gym import spaces
import tensorflow as tf
import src.algorithms.dummy as dummy
import scipy.optimize as opt
import src.algorithms.commons as acommons

from tensorflow_probability.python.distributions import TruncatedNormal
from tensorflow_probability.python.distributions import Distribution
from typing import List, Tuple

ACTION_PROJ_1 = 0
ACTION_PROJ_2 = 1
ACTION_RESERVE = 2
ACTION_DISMISS = 3

"""
This module contains the resource allocation environment that was used for experiments. 
It tests the applicability of reinforcement learning methods to discrete, sequential optimization tasks 
in e.g. project management.
"""


def create_projects(max_resource: int = 100, max_payout: float = 1000, stochastic: bool = True, mean_proba: float = 0.5,
                    std: float = 0.2, min_proj_time: int = 1, max_proj_time: int = 5, size: int = 10000,
                    seed: int = 123) -> pd.DataFrame:
    """
    Builds random projects that will be used by the simulator. Each project consists of the probability of success
     resource demand, and payout per resource unit.

    Parameters
    ----------
    max_resource: int
        Maximal number of resources, that can be required by the proejct.
    max_payout: int
        Maximal payout per project.
    stochastic: bool
        Is the environment stochastic? If no - then p(success) = 1. for each project.
    mean_proba: float
        Average success chance per project.
    std: float
        Standard deviation per project.
    min_proj_time: int
        Minimal time
    max_proj_time
    size
    seed

    Returns
    -------

    """
    np.random.seed(seed)
    data = {}
    for i in range(2):
        proj_allocs = np.random.randint(0, max_resource, size=size)
        proj_payouts = np.random.uniform(1, max_payout, size=size)  # randint(1, max_payout, size=size)
        proj_proba = np.clip(np.random.normal(mean_proba, std, size=size), 0., 1.) if stochastic else [1.0] * size
        proj_time = np.random.randint(min_proj_time, max_proj_time, size=(size,))
        data[f'proj_{i + 1}_alloc'] = proj_allocs
        data[f'proj_{i + 1}_payouts'] = proj_payouts
        data[f'proj_{i + 1}_proba'] = proj_proba
        data[f'proj_{i + 1}_time'] = proj_time
    return pd.DataFrame.from_dict(data)


class SimpleProjectsEnv(gym.Env):
    """
    Generalized implementation of projects env. It uses the proportional allocation of resources.
    It's a base class for later implementations (e.g., discrete projects allocation env).
    """
    ACTION_DISMISS = 2

    def __init__(
            self,
            start_resource: int = 100,
            start_cash: float = 1000,
            upkeep_cost=-10,
            min_payout: float = -100,
            max_payout: float = 100.,
            payout_mean: float = 0.0,
            payout_std: float = 50.,
            size: int = 10000,
            max_balance: float = 50000,
            stochastic: bool = False,
            seed: int = 123,
            balance_is_reward: float = False,
            projects: pd.DataFrame = None,
            normalize_from: str = 'tanh',
            distrib: Distribution = TruncatedNormal(0.5, 0.2, 0.0, 1.0)):
        """
        Initializes project environment simulator.

        Parameters
        ----------
        start_resource: int
            Initial resources for an agent.
        start_cash: float
            Initial cash resources.
        upkeep_cost: float
            Upkeep cost of idle resources.
        min_payout: int
            Minimal payout per project.
        max_payout: int
            Maximal payout per project.
        payout_mean: float
            Mean - a parameter for payout distribution.
        payout_std: float
            Standard deviation - a parameter for payout distribution.
        size: int
            Size of the environment. Number of time steps/decision points.
        max_balance: float
            Maximal balance for an agent.
        stochastic: bool
            Is env stochastic. If not then P(success) = 1 for each project.
        seed: int
            Random seed.
        balance_is_reward: bool
            Is balance a reward signal for an agent? If not - single payouts are used as rewards.
        projects: pd.DataFrame
            Predefined projects data frame if needed.
        normalize_from: str
            In case of proportional alloocation: re-normalize raw allocation outputs from a given function,
            to match values in range [0,1] and sum to 1. Used for proportional allocaation.
        distrib: Distribution
            A distribution to be used for building projects probability.
        """
        super(SimpleProjectsEnv, self).__init__()

        self.payout_mean = payout_mean
        self.payout_std = payout_std
        self.distrib = distrib
        if (normalize_from != 'tanh') and (normalize_from != 'logits'):
            raise ValueError("Invalid normalization")
        self.normalize_from = normalize_from
        self.stochastic = stochastic
        self.start_resource = start_resource
        self.start_cash = start_cash
        self.upkeep_cost = upkeep_cost
        self.max_payout = max_payout

        self.size = size
        self.max_balance = max_balance
        self.balance_is_reward = balance_is_reward

        if projects is None:
            self._build_projects(max_payout, min_payout, size, start_resource, stochastic, payout_mean, payout_std,
                                 seed)
        else:
            self.projects = projects
            self.size = projects.shape[0]

        self.reward_range = (0, max_payout)
        self.current_step = -1
        self.current_balance = start_cash
        self.current_resources = start_resource
        self.current_proj = None

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()
        self.reset()

    def _build_observation_space(self) -> spaces.Box:
        """
        Builds an observation space for each timestep. It contains the following elements:
        proj. A alloc, proj. A payout, P(success) proj. A,
        proj. B alloc, proj. B payout, P(success) proj. B,
        available resources, available cash.

        Returns
        -------
        spaces.Box
            A Box gym space with state elements.
        """
        return spaces.Box(
            low=np.array([
                0., 0., 0.,  # proj. a - alloc, payout, probability
                0., 0., 0.,  # proj. b - alloc, payout, probability
                0.,  # available resources
                0.,  # available cash
            ]),
            high=np.array([
                self.start_resource, self.max_payout, 1.,  # proj 1
                self.start_resource, self.max_payout, 1.,  # proj 2
                self.start_resource,  # resources
                self.max_balance  # balance
            ])
        )

    def _build_action_space(self) -> spaces.Box:
        """
        Builds an action space. In case of proportional allocation, it uses three possible actions:
        [%] allocation to proj. A, proj. B, resources kept idle.

        Returns
        -------
        spaces.Box
            Box space with predefined actions.
        """
        return spaces.Box(low=np.array([0., 0., 0.]), high=np.array([1., 1., 1.]))

    def _build_projects(self,
                        max_payout: int,
                        min_payout: int,
                        size: int,
                        start_resource: int,
                        stochastic: bool,
                        payout_mean: float,
                        payout_std: float,
                        seed: int = None):
        """
        Builds random projects that will be used by the simulator. Each project consists of the probability of success
        resource demand, and payout per resource unit.

        Parameters
        ----------
        max_payout: int
            Maximal payout per project.
        min_payout: int
            Minimal payout per project.
        size: int
            Size of the environment. Number of time steps/decision points.
        start_resource: int
            Initial resources for an agent.
        payout_mean: float
            Mean - a parameter for payout distribution.
        payout_std: float
            Standard deviation - a parameter for payout distribution.
        stochastic: bool
            Is env stochastic. If not then P(success) = 1 for each project.
        seed: int
            Random seed.

        Returns
        -------

        """
        if seed:
            np.random.seed(seed)
        payout_distrib = TruncatedNormal(payout_mean, payout_std, min_payout, max_payout)
        proj_1_alloc = np.random.randint(0, start_resource, size=size)
        proj_1_payouts = payout_distrib.sample(sample_shape=size).numpy().squeeze()
        proj_1_proba = self.distrib.sample(sample_shape=size).numpy().squeeze() if stochastic else np.ones(shape=size)
        proj_2_alloc = np.random.randint(0, start_resource, size=size)
        proj_2_payouts = payout_distrib.sample(sample_shape=size).numpy().squeeze()
        proj_2_proba = self.distrib.sample(sample_shape=size).numpy().squeeze() if stochastic else np.ones(shape=size)
        self.projects = pd.DataFrame({
            "proj_1_alloc": proj_1_alloc,
            "proj_1_payouts": proj_1_payouts,
            "proj_1_proba": proj_1_proba,
            "proj_2_alloc": proj_2_alloc,
            "proj_2_payouts": proj_2_payouts,
            "proj_2_proba": proj_2_proba
        })

    def reset(self):
        self.current_step = -1
        self.current_balance = self.start_cash
        self.current_resources = self.start_resource
        self.current_proj = None
        return self._next_observation()

    def step(self, a: np.array):
        action_val = self._take_action(a)
        obs = self._next_observation()
        done = self.current_balance <= 0.0 or self.current_resources <= 0
        if self.current_step == (self.size - 1):
            done = True
        reward = self.current_balance if self.balance_is_reward else action_val
        return obs, reward, done, {}

    def _next_observation(self) -> np.array:
        """
        Returns the next observation (projects A, B), subsequent time step.

        Returns
        -------
        np.array
            An array that describes the next state.
        """
        self.current_step += 1

        current_proj = self.projects.iloc[self.current_step]
        results = []
        results.extend([current_proj.proj_1_alloc, current_proj.proj_1_payouts, current_proj.proj_1_proba])
        results.extend([current_proj.proj_2_alloc, current_proj.proj_2_payouts, current_proj.proj_2_proba])

        results.append(self.current_resources)
        results.append(self.current_balance)
        self.current_proj = current_proj
        return np.array(results)

    def _take_action(self, a: np.array) -> float:
        """
        Performs an action in the environment. The action in this env. is expected to be a vector, produced from
        an agent. According to the property 'normalize_from' - actions vector is an output of a predefined function
        e.g. tanh or softmax. Then it is re-normalized to sum(alloc) = 1 and each alloc element should be in range [0,1].

        Parameters
        ----------
        a: np.array
            An array that describes action to be taken in the environment.

        Returns
        -------
        float
            Action value, value of the step taken - reward obtained from action. E.g. project payout, or penalty for
            an idle resources.
        """
        alloc = None
        if self.normalize_from == 'tanh':
            alloc = (a - (-1)) / 2.
        else:
            alloc = tf.nn.softmax(a).numpy().squeeze()
        alloc = (alloc * self.current_resources).astype(int)
        reserve = 0.
        if alloc[ACTION_PROJ_1] > self.current_proj.proj_1_alloc:
            diff = alloc[ACTION_PROJ_1] - self.current_proj.proj_1_alloc
            reserve += diff
            alloc[ACTION_PROJ_1] = self.current_proj.proj_1_alloc

        if alloc[ACTION_PROJ_2] > self.current_proj.proj_2_alloc:
            diff = alloc[ACTION_PROJ_2] - self.current_proj.proj_2_alloc
            reserve += diff
            alloc[ACTION_PROJ_2] = self.current_proj.proj_2_alloc

        proj_1_success = np.random.binomial(1, self.current_proj.proj_1_proba) if self.stochastic else 1.
        proj_2_success = np.random.binomial(1, self.current_proj.proj_2_proba) if self.stochastic else 1.
        proj_payout = (alloc[ACTION_PROJ_1] * self.current_proj.proj_1_payouts * proj_1_success) + (alloc[
                                                                                                        ACTION_PROJ_2] * self.current_proj.proj_2_payouts * proj_2_success)
        res_costs = reserve * self.upkeep_cost
        action_value = proj_payout + res_costs
        self.current_balance += action_value
        self.current_balance = np.clip(self.current_balance, -1 * self.max_balance, self.max_balance)
        self.current_resources -= alloc[self.ACTION_DISMISS]
        return action_value


class DiscreteProjectsEnv(SimpleProjectsEnv):
    """
    It is a discretized version of the project allocation env - where actions are discrete instead
    of continuous ([%] allocation). The remaining logic is the same as before.
    """
    A_PROJ_1 = 0
    A_PROJ_2 = 1
    A_PROJ_BOTH = 2
    A_WAIT = 3
    A_REDUCE_RES_10PERC = 4
    A_REDUCE_RES_25PERC = 5
    A_REDUCE_RES_50PERC = 6
    A_INCREASE_RES_10PERC = 7
    A_INCREASE_RES_25PERC = 8

    def __init__(
            self,
            start_resource: int = 100,
            start_cash: float = 1000,
            upkeep_cost=-10,
            min_payout: float = -100,
            max_payout: float = 100.,
            payout_mean: float = 0.0,
            payout_std: float = 50.,
            size: int = 10000,
            max_balance: float = 50000,
            stochastic: bool = False,
            seed: int = 123,
            balance_is_reward: float = False,
            projects: pd.DataFrame = None,
            increase_resource_cost: float = -20,
            normalize_from: str = 'tanh',
            distrib_prob: TruncatedNormal = TruncatedNormal(0.5, 0.2, 0.0, 1.0)):
        super().__init__(start_resource, start_cash, upkeep_cost, min_payout, max_payout, payout_mean, payout_std, size,
                         max_balance, stochastic, seed, balance_is_reward, projects, normalize_from, distrib_prob)
        self.increase_resource_cost = increase_resource_cost

    def _build_action_space(self):
        return spaces.Discrete(9)

    def _take_action(self, a: np.array) -> float:
        """
        In case of a discrete environment, the action vector is a single number, that describes a possible action.
        Allowed values are integers in range 0-8.

        Parameters
        ----------
        a: np.array
            Vector of actions. In case of a discrete env - a single integer.

        Returns
        -------
        float
            Action value, value of the step taken - reward obtained from action. E.g. project payout, or penalty for
            an idle resources.
        """
        if a.shape != ():
            raise ValueError(f"A has invalid shape. Expected (), received: {a.shape}")
        a = int(a)
        reserve = 0.0
        score = 0.0
        if a == self.A_PROJ_1:
            proj_1_score, proj_1_alloc = self.get_project_alloc_score(self.current_proj.proj_1_proba,
                                                                      self.current_proj.proj_1_alloc,
                                                                      self.current_proj.proj_1_payouts,
                                                                      self.current_resources)
            score += proj_1_score
            reserve += (self.current_resources - proj_1_alloc)
        elif a == self.A_PROJ_2:
            proj_2_score, proj_2_alloc = self.get_project_alloc_score(self.current_proj.proj_2_proba,
                                                                      self.current_proj.proj_2_alloc,
                                                                      self.current_proj.proj_2_payouts,
                                                                      self.current_resources)
            score += proj_2_score
            reserve += (self.current_resources - proj_2_alloc)
        elif a == self.A_PROJ_BOTH:
            half_res = self.current_resources // 2
            proj_1_score, proj_1_alloc = self.get_project_alloc_score(self.current_proj.proj_1_proba,
                                                                      self.current_proj.proj_1_alloc,
                                                                      self.current_proj.proj_1_payouts,
                                                                      half_res)
            proj_2_score, proj_2_alloc = self.get_project_alloc_score(self.current_proj.proj_2_proba,
                                                                      self.current_proj.proj_2_alloc,
                                                                      self.current_proj.proj_2_payouts,
                                                                      half_res)
            score += (proj_1_score + proj_2_score)
            reserve += (self.current_resources - (proj_1_alloc + proj_2_alloc))
        elif a == self.A_WAIT:
            reserve = self.current_resources
        elif a == self.A_REDUCE_RES_10PERC:
            self.current_resources = int(self.current_resources * 0.9)
            reserve = self.current_resources
        elif a == self.A_REDUCE_RES_25PERC:
            self.current_resources = int(self.current_resources * 0.75)
            reserve = self.current_resources
        elif a == self.A_REDUCE_RES_50PERC:
            self.current_resources = int(self.current_resources * 0.5)
            reserve = self.current_resources
        elif a == self.A_INCREASE_RES_10PERC:
            increase_cost, increase_size = self._calculate_size_increase(0.1)
            reserve = self.current_resources
            self.current_resources += increase_size
            score = -1 * increase_cost
        elif a == self.A_INCREASE_RES_25PERC:
            increase_cost, increase_size = self._calculate_size_increase(0.25)
            reserve = self.current_resources
            self.current_resources += increase_size
            score = -1 * increase_cost

        cost = reserve * self.upkeep_cost
        action_value = score + cost
        self.current_balance += action_value
        self.current_balance = np.clip(self.current_balance, -1 * self.max_balance, self.max_balance)

        return action_value

    def _calculate_size_increase(self, ratio: float) -> Tuple[float, int]:
        """
        An internal helper method to calculate the cost for resource increase.

        Parameters
        ----------
        ratio: float
            Resource increase ratio.

        Returns
        -------
        Tuple[float, int]
            A tuple that describes the increase cost and increased size in integers.

        """
        increase_size = int(self.current_resources * ratio)
        increase_cost = -1 * increase_size * self.increase_resource_cost
        return increase_cost, increase_size

    def get_project_alloc_score(self, proj_proba, proj_alloc, proj_payout, resources):
        alloc = np.minimum(proj_alloc, resources)
        success = np.random.binomial(1, proj_proba) if self.stochastic else 1.
        score = alloc * proj_payout * success
        return score, alloc


class OptimizerAgent(dummy.DummyAgent):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        probas = np.array([s[2], s[5]])
        rewards = np.array([s[1], s[4]])
        allocs = np.array([s[0], s[3]])
        expected_rewards = allocs * rewards * probas

        resources = s[-2]
        expected_vals = probas * expected_rewards

        bounds = [(0, 1.), (0., 1.)]
        A_ub = np.array([
            [resources, 0.],
            [0., resources],
            [resources, resources]
        ])
        b_ub = np.array([
            np.min([allocs[0], resources]),
            np.min([allocs[1], resources]),
            resources
        ])
        sol = opt.linprog(-1 * expected_rewards * resources, A_ub=A_ub, b_ub=b_ub)
        perc_alloc = np.array(list(sol.x.round(2)) + [0.])
        alloc_scaled_m1_to_1 = (perc_alloc * 2.) - 1.

        return alloc_scaled_m1_to_1, None


class DiscreteProjectOptimizerAgent(dummy.DummyAgent):

    def __init__(self, env: DiscreteProjectsEnv, name: str = "Optimization agent"):
        super().__init__(env, name)
        self.projects_env = env

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        resources = s[-2]
        rewards = np.array([s[1], s[4]])
        probabilities = np.array([s[2], s[4]])
        allocs = np.array([
            np.minimum(s[0], resources),
            np.minimum(s[3], resources)
        ])
        outcomes = allocs * rewards * probabilities

        half_resources = int(resources // 2)
        half_alloc = np.array([
            np.minimum(s[0], half_resources),
            np.minimum(s[3], half_resources)
        ])
        half_outcome = np.sum(half_alloc * rewards * probabilities)
        results = np.append(outcomes, half_outcome)

        wait_res = self.projects_env.upkeep_cost * self.projects_env.current_resources
        results = np.append(results, wait_res)

        for perc in [0.1, 0.25, 0.5]:
            decrease_and_wait = self.projects_env.upkeep_cost * int(self.projects_env.current_resources * (1. - perc))
            results = np.append(results, decrease_and_wait)

        for perc in [0.1, 0.25]:
            increase_size_cost, _ = self.projects_env._calculate_size_increase(perc)
            total_cost = (-1 * increase_size_cost) + self.projects_env.upkeep_cost * self.projects_env.current_resources
            results = np.append(results, total_cost)

        actions = np.array([self.projects_env.A_PROJ_1, self.projects_env.A_PROJ_2, self.projects_env.A_PROJ_BOTH,
                            self.projects_env.A_WAIT, self.projects_env.A_REDUCE_RES_10PERC,
                            self.projects_env.A_REDUCE_RES_25PERC, self.projects_env.A_REDUCE_RES_50PERC,
                            self.projects_env.A_INCREASE_RES_10PERC, self.projects_env.A_INCREASE_RES_25PERC])

        action = actions[np.argmax(results)]
        return action, None


class ActionClipperProba(acommons.ActionClipper):

    @staticmethod
    def clip_action(action: np.array, env: gym.wrappers.TimeLimit, *args, **kwargs):
        action_0_to_1 = (action - (-1.)) / 2.
        return action_0_to_1
