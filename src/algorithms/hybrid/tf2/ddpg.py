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
    Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.
    """
    def __init__(self,
                 actor_net: pi_net.DeterministicPolicyNet,
                 actor_net_target: pi_net.DeterministicPolicyNet,
                 critic_net: qnets.QSANet,
                 critic_net_target: qnets.QSANet,
                 buffer: mbuff.SimpleMemory,
                 a_min: float,
                 a_max: float,
                 actor_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 critic_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 loss_function: str = 'mean_squared_error',
                 action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper(),
                 polyak_tau: float = 0.005,
                 noise_gen: autils.NoiseGenerator = autils.OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1)),
                 name: str = "DDPG",
                 target_update_frequency: int = 1):
        super().__init__(action_clipper, name)
        self.noise_gen = noise_gen
        self.actor_net = actor_net
        self.target_actor = actor_net_target
        self.critic_net = critic_net
        self.target_critic = critic_net_target

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.loss_function = loss_function
        self.a_min = a_min
        self.a_max = a_max
        self.buffer = buffer
        self.polyak_tau = polyak_tau
        self.target_update_frequency = target_update_frequency

        self.noise_std = noise_gen

        # self.target_critic.net.set_weights(self.critic_net.net.get_weights())
        # self.target_actor.set_weights(self.actor_net.get_weights())

    def choose_action(self, s: np.array, deterministic: bool = False, *args, **kwargs) -> Tuple[np.array, np.array]:
        """
        A function for choosing an action using a policy net.
        Parameters
        ----------
        s: np.array
            Current state for which to choose an action.
        deterministic: bool
            Should action be deterministic or have injected noise?

        Returns
        -------
        Tuple[np.array, np.array]
            Tuple of action and log prob (always None in this case).
        """
        a, _ = self.actor_net.policy(np.atleast_2d(s))
        a = a.numpy()
        if not deterministic:
            noise = self.noise_gen.get_noise(a).squeeze()
            self.noise_gen.update_noise_params()
            a_noisy = np.clip(a + noise, self.a_min, self.a_max)
        else:
            a_noisy = a
        return a_noisy.squeeze(), None

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
        """
        A training function for DDPG algorithm.

        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            Env to train on.
        nepisodes: int
            Number of episodes to train on.
        gamma: float
            Discount factor.
        max_steps: int
            Maximum number of steps per episode.
        max_train_sec: int
            Maximum training time.
        print_interval: int
            Logging interval.
        average_n_last: int
            Smoothing window of n scores when logging.
        scaler: object
            Any scaling object
        clip_action: bool
            Should action be clipped back to the original scale of env.
        batch_size: int
            Batch size for memory buffer.
        warmup_batches: int
            Number of batches treated as a 'warmup' for the algorithm.
        log: logging.Logger
            Logger

        Returns
        -------
        np.array
            Episode scores.
        """
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

                if self.buffer.actual_size >= warmup_size or done:
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

                if learning_step % self.target_update_frequency == 0:
                    self.update_networks()

            if ep % print_interval == 0:
                autils.log_progress(max_train_sec, average_n_last, all_scores, t0, losses, ep, nepisodes, log,
                                    kwargs.get(consts.MLFLOW_LOG, True))

            if stop_training:
                break
            all_scores.append(ep_score)
        return np.array(all_scores)

    def update_networks(self):
        qnets.polyak_tau_update_networks(self.target_actor, self.actor_net, self.polyak_tau)
        qnets.polyak_tau_update_networks(self.target_critic.net, self.critic_net.net, self.polyak_tau)

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
        """
        Learning function - applied for a single batch of experiences.

        Parameters
        ----------
        s: np.array
            Current states.
        a: np.array
            Executed actions
        a_logprob: np.array
            Action log probs (always None for DDPG).
        r: np.array
            Rewards
        sprime: np.array
            Next states
        done: np.array
            Is episode done?
        vs: np.array
            V(st)
        v_sprime: np.array
            V(s t+1)
        gamma: float
            Discount factor
        learning_step: int
            Which learning iteration is this.

        Returns
        -------
        float
            Averaged actor and critic loss.
        """
        sprime_action, _ = self.target_actor.policy(sprime)

        sprime_target_q_vals = tf.squeeze(self.target_critic.state_action_value(sprime, sprime_action))

        # y = ri + gamma * Q-target(s', a') * (1-done)
        td_target = r + (gamma * (sprime_target_q_vals * (1. - done)))

        loss_critic = self.optimize_critic(a, s, td_target)
        loss_actor = self.optimize_actor(s)

        return 0.5 * (loss_critic + loss_actor)

    def optimize_actor(self, s: np.array) -> float:
        """
        Helper function for optimizing actor network.

        Parameters
        ----------
        s: np.array
            Current states array.

        Returns
        -------
        float
            Actor loss
        """
        with tf.GradientTape() as tape:
            new_actions = self.actor_net(np.atleast_2d(s))
            vs = self.critic_net.net([s, new_actions], training=True)
            loss_actor = -1 * tf.math.reduce_mean(vs)
        grad_actor = tape.gradient(loss_actor, self.actor_net.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grad_actor, self.actor_net.trainable_variables))
        return loss_actor

    def optimize_critic(self, a: np.array, s: np.array, td_target: np.array) -> float:
        """
        Helper function for optimizing critic network.

        Parameters
        ----------
        a: np.array
            Executed actions.
        s: np.array
            Current states
        td_target: np.array
            TD-target gamma * r + V(s t+1) - V(s t)

        Returns
        -------
        float
            Critic loss.
        """
        with tf.GradientTape() as tape:
            mse = krs.losses.MeanSquaredError()
            s_qval = self.critic_net.net([s, a], training=True)
            loss_critic = mse(td_target, s_qval)
        grad_critic = tape.gradient(loss_critic, self.critic_net.net.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grad_critic, self.critic_net.net.trainable_variables))
        return loss_critic
