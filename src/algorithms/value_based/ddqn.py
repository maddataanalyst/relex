import logging
import time
from typing import Tuple, List

import tensorflow as tf
import tensorflow.keras as krs
import gym
import numpy as np

import src.algorithms.commons as acommons
import src.algorithms.commons as rl_commons
import src.models.base_models.tf2.q_nets as qnets
import src.algorithms.memory_samplers as mbuff
import src.algorithms.algo_utils as autils
from src import consts


class DDQN(acommons.RLAgent):
    """
    Double Dueling Deep Q-network implementaion, related to paper:
    Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

    """
    def __init__(self,
                 online_qnet: qnets.QNet,
                 target_qnet: qnets.QNet,
                 buffer: mbuff.SimpleMemory,
                 eps_start: float,
                 eps_min: float,
                 eps_decay_rate: float = 0.1,
                 net_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 loss_function: str = 'mean_squared_error',
                 action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper(),
                 polyak_tau: float = 0.005,
                 name: str = "DDQN",
                 target_update_frequency: int = 5):
        super().__init__(action_clipper, name)
        self.online_qnet = online_qnet
        self.net_optimizer = net_optimizer
        self.loss_function = loss_function
        self.target_qnet = target_qnet
        self.eps_min = eps_min
        self.eps_decay_rate = eps_decay_rate
        self.eps_start = eps_start
        self.buffer = buffer
        self.polyak_tau = polyak_tau
        self.target_update_frequency = target_update_frequency

        self.target_qnet.net.set_weights(self.online_qnet.net.get_weights())

        self.online_qnet.net.compile(loss=self.loss_function, optimizer=self.net_optimizer)
        self.target_qnet.net.compile(loss=self.loss_function, optimizer=self.net_optimizer)

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        epsilon = kwargs.get(consts.EPSILON, 0.0)
        p = np.random.rand()
        if p < epsilon:
            a = np.random.randint(self.online_qnet.action_dim)
            return np.array(a), None
        else:
            prediction = self.online_qnet.state_action_value(np.atleast_2d(s)).numpy()
            return prediction.argmax(axis=1).squeeze(), None

    def train(self,
              env: gym.wrappers.TimeLimit,
              nepisodes: int,
              gamma: float = 0.9,
              max_steps: int = 500,
              max_train_sec: int = 900,
              print_interval: int = 10,
              average_n_last: int = 30,
              scaler: object = None,
              clip_action: bool = True,
              batch_size: int = 32,
              warmup_batches: int = 5,
              log: logging.Logger = logging.getLogger("dqn_logger"), *args, **kwargs) -> np.array:
        all_scores = []
        t0 = time.time()
        losses = []
        learning_step = 0

        warmup_size = warmup_batches * batch_size
        stop_training = False

        if type(env.observation_space) == gym.spaces.Discrete:
            dim_state = env.observation_space.n
        else:
            dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0] if type(env.action_space) == gym.spaces.Box else 1

        epsilon_decays = autils.decay_schedule_epsilon(self.eps_start, self.eps_min, self.eps_decay_rate, max_steps=nepisodes)

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

                epsilon = epsilon_decays[ep]
                a, a_logprob, done, r, sprime = self.make_step(s, env, clip_action, scaler, epsilon=epsilon)
                self.buffer.store_transition(s, a, r, sprime, None, None, a_logprob, done)

                if self.buffer.actual_size >= warmup_size:
                    memory_sample = self.buffer.sample(batch_size, dim_state, dim_action, preserve_order=False)
                    batch_s, batch_a, \
                    batch_r, batch_done, \
                    batch_sprime, _, \
                    _, batch_logprob = memory_sample
                    loss = self.learn(
                        batch_s, batch_a, batch_logprob,
                        batch_r, batch_sprime, batch_done,
                        None, None, gamma, learning_step, batch_size=batch_size)
                    learning_step += 1
                    losses.append(loss)
                    tf.summary.scalar("Loss", data=loss, step=learning_step)
                s = sprime
                ep_score += r

            if ep % self.target_update_frequency == 0:
                qnets.polyak_tau_update_networks(self.target_qnet.net, self.online_qnet.net, self.polyak_tau)

            if ep % print_interval == 0:
                autils.log_progress(max_train_sec, average_n_last, all_scores, t0, losses, ep, nepisodes, log,
                                    kwargs.get(consts.MLFLOW_LOG, True))

            if stop_training:
                break
            all_scores.append(ep_score)
        return np.array(all_scores)

    def learn(self,
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
              *args, **kwargs) -> float:
        indices = np.arange(s.shape[0])
        sprime_target_q_vals = self.target_qnet.state_action_value(sprime).numpy()
        max_q_val = np.max(sprime_target_q_vals, axis=1)

        # y = ri + gamma * max a' Q-target(s', a')
        targets = r + (gamma * (max_q_val * (1. - done)))

        # Loss = 1/N Q(s,a) - y
        expected = self.online_qnet.state_action_value(s).numpy()
        expected[indices, a.squeeze()] = targets

        loss = self.online_qnet.net.train_on_batch(s, expected)

        return loss
