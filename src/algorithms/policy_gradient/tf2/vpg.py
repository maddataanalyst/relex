import logging
import random
import time
from typing import Tuple, List

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

import src.consts as consts
import src.algorithms.advantages as adv
import src.algorithms.algo_utils as autils
import src.algorithms.commons as rl_commons
import src.algorithms.memory_samplers as mbuff
import src.models.base_models.tf2.policy_nets as policy_nets
import src.models.base_models.tf2.value_nets as vnets
import src.algorithms.policy_gradient.tf2.policy_gradient_commons as pgc
from src.algorithms.policy_gradient.tf2.policy_gradient_commons import PolicyGradientBase
from src.algorithms.commons import ActionClipper


class VPG(PolicyGradientBase):

    """Vanilla policy gradient implementation - the basic VPG algorithm with optional possible extensions, like
    different advantage estimation or N-steps method utilization.

    References
    -------
    [1] Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12.
    [2] Schulman, J. (2016). Optimizing expectations: From deep reinforcement learning to stochastic computation graphs (Doctoral dissertation, UC Berkeley).
    [3] https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(
            self,
            policy_net: policy_nets.BaseStochasticPolicyNet,
            value_net: vnets.ValueNet,
            advantage: adv.Advantage,
            action_clipper: ActionClipper = rl_commons.DefaultActionClipper(),
            entropy_coef: float = 1e-3,
            policy_opt: krs.optimizers.Optimizer = krs.optimizers.Adam(1e-3),
            value_opt: krs.optimizers.Optimizer = krs.optimizers.Adam(1e-3)):
        super(VPG, self).__init__(policy_net, value_net, advantage, action_clipper, entropy_coef, policy_opt, value_opt)

    def train(
            self,
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
            log: logging.Logger = logging.getLogger("vpg_logger"),
            lambda_: float = 0.95,
            *args,
            **kwargs) -> np.array:
        all_scores = []
        t0 = time.time()
        losses = []
        learning_step = 0

        buffer = mbuff.SimpleMemory(max_size=max_steps)
        stop_training = False

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0] if type(env.action_space) == gym.spaces.Box else 1

        for ep in range(nepisodes):

            s = autils.normalize_state(env.reset(), scaler)
            done = False
            ep_score = 0

            for step in range(max_steps):
                t = time.time()
                delta_t = t - t0
                if delta_t >= max_train_sec:
                    stop_training = True
                    break
                if done:
                    break

                a, a_logprob, done, r, sprime = self.make_step(s, env, clip_action, scaler)
                vs = self.value_net.state_value(np.atleast_2d(s)).numpy().squeeze()
                vs_prime = self.value_net.state_value(np.atleast_2d(sprime)).numpy().squeeze()
                buffer.store_transition(s, a, r, sprime, vs, vs_prime, a_logprob, done)

                s = sprime
                ep_score += r

            memory_sample = buffer.sample(max_steps, dim_state, dim_action)
            batch_s, batch_a, \
            batch_r, batch_done, \
            batch_sprime, batch_vs, \
            batch_vs_prime, batch_logprob = memory_sample

            batch_s = batch_s[0]
            batch_a = batch_a[0]
            batch_r = batch_r[0]
            batch_done = batch_done[0]
            batch_sprime = batch_sprime[0]
            batch_vs = batch_vs[0]
            batch_vs_prime = batch_vs_prime[0]
            batch_logprob = batch_logprob[0]

            loss = self.learn(
                batch_s, batch_a, batch_logprob,
                batch_r, batch_sprime, batch_done,
                batch_vs, batch_vs_prime, gamma, learning_step)
            learning_step += 1
            losses.append(loss)
            tf.summary.scalar("Loss", data=loss, step=learning_step)

            if ep % print_interval == 0:
                autils.log_progress(max_train_sec, average_n_last, all_scores, t0, losses, ep, nepisodes, log,
                                    kwargs.get(consts.MLFLOW_LOG, True))

            if stop_training:
                break
            all_scores.append(ep_score)

        return np.array(all_scores)

