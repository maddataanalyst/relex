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
import src.models.base_models.tf2.value_nets as vnet

from src.algorithms.commons import ActionClipper, check_train_time
from src.algorithms.policy_gradient.tf2.policy_gradient_commons import PolicyGradientBase

BATCH_TD_TARGETS_KWARG = 'batch_td_targets'
BATCH_GAE_KWARG = 'batch_gae'


class ActorCrtic(PolicyGradientBase):
    """Single-threaded implementation of generic Actor-Critic models with possible settings for different
    advantage estimations.

    References
    -------
    [1] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937). PMLR.
    [2] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press, ch 12.2
    """

    def __init__(
            self,
            actor: policy_nets.BasePolicyNet,
            critic: vnet.ValueNet,
            actor_opt: krs.optimizers.Optimizer,
            critic_opt: krs.optimizers.Optimizer,
            advantage: adv.Advantage,
            entropy_coef: float = 1e-3,
            action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper(),
            name: str = "AC"):
        super(ActorCrtic, self).__init__(actor, critic, advantage, action_clipper, entropy_coef, actor_opt, critic_opt, name)
        self.actor = actor
        self.critic = critic
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt

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
            clip_action: bool = True,
            log: logging.Logger = logging.getLogger("ppo_logger"),
            n_steps: int = 25,
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

        clip_action: bool
            Should actions be clipped?

        log: logging.Logger
            Logging object.

        n_steps: int
            Number of steps for Actor Critic. Defaults to None.
            Nsteps = max_steps results in a Monte Carlo AC with advantage estimation.
            NSteps < max_steps results in a "n-step" actor critic with advantage estimation.
            Nsteps = 1 is a single step TD(0) actor critic with advantage estimation.

        Returns
        -------
        np.array
            Episode scores.
        """
        all_scores = []
        t0 = time.time()
        losses = []
        learning_step = 0
        stop_training = False

        memory = mbuff.SimpleMemory(n_steps)

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0] if type(env.action_space) == gym.spaces.Box else 1

        for ep in range(nepisodes):
            s = autils.normalize_state(env.reset(), scaler)
            done = False
            ep_score = 0

            for step in range(max_steps):
                time_check_result = check_train_time(t0, max_train_sec)
                stop_training = time_check_result.should_stop_training
                if time_check_result.should_break_loop:
                    break

                if done:
                    break

                a, a_logprob, done, r, sprime = self.make_step(s, env, clip_action, scaler)
                vs = self.critic.state_value(np.atleast_2d(s)).numpy().squeeze()
                vs_prime = self.critic.state_value(np.atleast_2d(sprime)).numpy().squeeze()
                memory.store_transition(s, a, r, sprime, vs, vs_prime, a_logprob, done)

                if (memory.actual_size >= n_steps) or (done and memory.actual_size > 0):
                    memory_sample = memory.sample(n_steps, dim_state, dim_action)
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

                    memory.clear_all()

                s = sprime
                ep_score += r

            all_scores.append(ep_score)
            if ep % print_interval == 0:
                autils.log_progress(max_train_sec, average_n_last, all_scores, t0, losses, ep, nepisodes, log,
                                    kwargs.get(consts.MLFLOW_LOG, True))

            if stop_training:
                break

        return np.array(all_scores)
