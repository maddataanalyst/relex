import logging
import time
from typing import Tuple, List

import tensorflow as tf
import tensorflow.keras as krs
import gym
import numpy as np

import src.algorithms.commons as acommons
import src.algorithms.commons as rl_commons
import src.models.base_models.tf2.policy_nets as pi_net
import src.models.base_models.tf2.q_nets as qnets
import src.algorithms.memory_samplers as mbuff
import src.algorithms.algo_utils as autils
from src import consts


class DDPG(acommons.RLAgent):
    """
    DDPG implementation. Paper:
    Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, January). Deterministic policy gradient algorithms. In International conference on machine learning (pp. 387-395). PMLR.
    """
    def __init__(self,
                 actor_net: pi_net.DeterministicPolicyNet,
                 actor_net_target: pi_net.DeterministicPolicyNet,
                 critic_net: qnets.QSANet,
                 critic_net_target: qnets.QSANet,
                 buffer: mbuff.SimpleMemory,
                 a_min: float,
                 a_max: float,
                 net_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 loss_function: str = 'mean_squared_error',
                 action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper(),
                 polyak_tau: float = 0.005,
                 noise_std: float = 0.1,
                 name: str = "DDPG",
                 target_update_frequency: int = 5):
        super().__init__(action_clipper, name)
        self.actor_net = actor_net
        self.actor_net_target = actor_net_target
        self.critic_net = critic_net
        self.critic_net_target = critic_net_target

        self.net_optimizer = net_optimizer
        self.loss_function = loss_function
        self.a_min = a_min
        self.a_max = a_max
        self.buffer = buffer
        self.polyak_tau = polyak_tau
        self.target_update_frequency = target_update_frequency

        self.noise_std = noise_std

        self.critic_net_target.net.set_weights(self.critic_net.net.get_weights())
        self.actor_net_target.set_weights(self.actor_net.get_weights())

        self.actor_net.compile(loss=self.loss_function, optimizer=self.net_optimizer)
        self.actor_net_target.compile(loss=self.loss_function, optimizer=self.net_optimizer)

        self.critic_net.net.compile(loss=self.loss_function, optimizer=self.net_optimizer)
        self.critic_net_target.net.compile(loss=self.loss_function, optimizer=self.net_optimizer)

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        a, _ = self.actor_net.policy(np.atleast_2d(s))
        a = a.numpy()
        a_noisy = a + np.random.normal(0.0, scale=self.noise_std, size=(1, self.actor_net.a_dim))
        a_clip = np.clip(a_noisy, self.a_min, self.a_max)
        return a_clip.squeeze(), None

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

                a, _, done, r, sprime = self.make_step(s, env, clip_action, scaler)
                self.buffer.store_transition(s, a, r, sprime, None, None, None, done)

                if self.buffer.actual_size >= warmup_size:
                    memory_sample = self.buffer.sample(batch_size, dim_state, dim_action, preserve_order=False)
                    batch_s, batch_a, \
                    batch_r, batch_done, \
                    batch_sprime, _, \
                    _, _ = memory_sample
                    loss = self.learn(
                        batch_s, batch_a, None,
                        batch_r, batch_sprime, batch_done,
                        None, None, gamma, learning_step, batch_size=batch_size)
                    learning_step += 1
                    losses.append(loss)
                    tf.summary.scalar("Loss", data=loss, step=learning_step)
                s = sprime
                ep_score += r

            if ep % self.target_update_frequency == 0:
                qnets.polyak_tau_update_networks(self.actor_net_target, self.actor_net, self.polyak_tau)
                qnets.polyak_tau_update_networks(self.critic_net_target.net, self.critic_net.net, self.polyak_tau)

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
        sprime_action, _ = self.actor_net_target.policy(sprime)
        sprime_action = sprime_action.numpy()
        sprime_target_q_vals = self.critic_net_target.state_action_value(sprime, sprime_action).numpy().squeeze()

        # y = ri + gamma * Q-target(s', a') * (1-done)
        targets_critic = r + (gamma * (sprime_target_q_vals * (1. - done)))

        with tf.GradientTape() as tape:
            mse = krs.losses.MeanSquaredError()
            s_qval = self.critic_net.state_action_value(s, a, training=True)
            loss_critic = mse(targets_critic, s_qval)
        grad_critic = tape.gradient(loss_critic, self.critic_net.net.trainable_variables)
        self.net_optimizer.apply_gradients(zip(grad_critic, self.critic_net.net.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions, _ = self.actor_net.policy(s)
            vs = self.critic_net.state_action_value(s, new_actions, training=True)
            loss_actor = -1 * tf.math.reduce_mean(vs)
        grad_actor = tape.gradient(loss_actor, self.actor_net.trainable_variables)
        self.net_optimizer.apply_gradients(zip(grad_actor, self.actor_net.trainable_variables))

        return 0.5 * (loss_critic + loss_actor)
