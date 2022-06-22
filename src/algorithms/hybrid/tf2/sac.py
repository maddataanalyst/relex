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
import src.models.base_models.tf2.q_nets as q_nets


class SAC(rl_commons.RLAgent):
    #TODO: refactor to common class, extract shared code
    #TODO: SAC is really slow. Need to run it with profiler & refactor/speed up

    """Implementation of SAC algorithm with possible extensions
    """

    def __init__(self,
                 actor: policy_nets.BasePolicyNet,
                 critic1: q_nets.QSANet,
                 critic2: q_nets.QSANet,
                 target_critic1: q_nets.QSANet,
                 target_critic2: q_nets.QSANet,
                 actor_opt: krs.optimizers.Optimizer,
                 crtitic_opt: krs.optimizers.Optimizer,
                 buffer: mbuff.SimpleMemory,
                 alpha_start: float = 1.0,
                 auto_alhpa: bool = True,
                 alpha_opt: krs.optimizers.Optimizer = krs.optimizers.Adam(),
                 entropy_coef: float = 1e-3,
                 action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper()):
        super(SAC, self).__init__(action_clipper)
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2
        self.actor_opt = actor_opt
        self.value_opt = crtitic_opt
        self.auto_alpha = auto_alhpa
        self.alpha_opt = alpha_opt
        self.alpha_start = alpha_start
        self.buffer = buffer
        self.alpha = tf.Variable(alpha_start, dtype=np.float32)
        self.log_alpha = tf.Variable(np.log(alpha_start), dtype=np.float32)

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
              log: logging.Logger = logging.getLogger("sac_logger"),
              warmup_batches: int = 5,
              polyak_tau: float = 0.005,
              **kwargs) -> np.array:
        all_scores = []
        t0 = time.time()
        losses = []
        learning_step = 0

        warmup_size = warmup_batches * batch_size
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

                self.buffer.store_transition(s, a, r, sprime, None, None, a_logprob, done)

                if self.buffer.actual_size >= warmup_size:
                    memory_sample = self.buffer.sample(batch_size, dim_state, dim_action)
                    batch_s, batch_a, \
                    batch_r, batch_done, \
                    batch_sprime, _, \
                    _, batch_logprob = memory_sample
                    sampling_loss = 0.0
                    for batch_i in range(batch_s.shape[0]):
                        loss = self.learn(
                            batch_s[batch_i], batch_a[batch_i], batch_logprob[batch_i],
                            batch_r[batch_i], batch_sprime[batch_i], batch_done[batch_i],
                            None, None, gamma, learning_step)
                        sampling_loss += loss
                    learning_step += 1
                    losses.append(loss)
                    tf.summary.scalar("Loss", data=loss, step=learning_step)

                    if self.auto_alpha:
                        for batch_i in range(batch_s.shape[0]):
                            self.tune_alpha(batch_s[batch_i], learning_step)

                    self.update_target_nets(polyak_tau)
                s = sprime
                ep_score += r
            if ep % print_interval == 0:
                autils.log_progress(max_train_sec, average_n_last, all_scores, t0, losses, ep, nepisodes, log,
                                    kwargs.get(consts.MLFLOW_LOG, True))

            if stop_training:
                break
            all_scores.append(ep_score)

        return np.array(all_scores)

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
            *args, **kwargs) -> float:

        td_target = self.build_td_target(sprime, r, done, gamma).numpy()
        loss_critic1 = self.optimize_critic(self.critic1, s, a, td_target, "critic1", learning_step)
        loss_critic2 = self.optimize_critic(self.critic2, s, a, td_target, "critic2", learning_step)
        loss_actor = self.optimize_actor(s, learning_step)
        return np.mean([loss_critic1, loss_critic2, loss_actor])

    def build_td_target(self, sprime, r, done, gamma):
        N = sprime.shape[0]
        sprime_a_tild, sprime_a_tild_logprob = self.actor.policy(sprime)
        sprime_a_tild_logprob = tf.squeeze(sprime_a_tild_logprob)
        vs_prime_target1 = tf.squeeze(self.target_critic1.state_action_value(sprime, sprime_a_tild))
        vs_prime_target2 = tf.squeeze(self.target_critic2.state_action_value(sprime, sprime_a_tild))

        target_v_sprime_val = tf.minimum(vs_prime_target1, vs_prime_target2) - tf.multiply(self.alpha,
                                                                                           sprime_a_tild_logprob)
        td_target = r.squeeze() + (gamma * target_v_sprime_val * (1. - done.squeeze()))

        tf.debugging.assert_shapes([
            (vs_prime_target1, (N,)),
            (vs_prime_target2, (N,)),
            (td_target, (N,))
        ])

        return td_target

    def optimize_critic(self,
                        critic: q_nets.QSANet,
                        states: np.array,
                        actions: np.array,
                        td_targets: np.array,
                        name: str,
                        learning_step: int):
        N = states.shape[0]
        with tf.GradientTape() as tape:
            s_qvals = tf.squeeze(critic.state_action_value(states, actions, training=True))

            tf.debugging.assert_shapes([
                (td_targets, (N,)),
                (s_qvals, (N,))
            ])

            loss = krs.losses.mean_squared_error(tf.reshape(td_targets, (N,)), tf.reshape(s_qvals, (N,)))
        grad = tape.gradient(loss, critic.net.trainable_variables)
        self.value_opt.apply_gradients(zip(grad, critic.net.trainable_variables))
        for idx, param in enumerate(critic.net.trainable_variables):
            tf.debugging.check_numerics(param, f"{name} net contains NaNs in weights")
        tf.summary.scalar(f"{name} loss", data=loss, step=learning_step)
        return loss

    def optimize_actor(self, s: np.ndarray, learning_step):
        N = s.shape[0]
        with tf.GradientTape() as tape:
            a_tilde, log_prob = self.actor.policy(s)
            critic1_val = self.critic1.state_action_value(s, a_tilde)
            critic1_val = tf.squeeze(critic1_val, 1)
            critic2_val = self.critic2.state_action_value(s, a_tilde)
            critic2_val = tf.squeeze(critic2_val, 1)

            target_critic_val = tf.math.minimum(critic1_val, critic2_val)
            tf.debugging.assert_shapes([
                (critic1_val, (N,)),
                (critic2_val, (N,)),
                (log_prob, (N,)),
            ])
            alpha_logprob = tf.multiply(self.alpha, log_prob)
            actor_loss = -1 * tf.math.reduce_mean(target_critic_val - alpha_logprob)
        grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grad, self.actor.trainable_variables))

        for idx, param in enumerate(self.actor.trainable_variables):
            tf.debugging.check_numerics(param, "Actor net contains NaNs in weights")
        tf.summary.scalar("Actor loss", data=actor_loss, step=learning_step)
        return actor_loss

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        a, a_logprob = self.actor.policy(np.atleast_2d(s))
        a = a.numpy().squeeze()
        a_logprob = a_logprob.numpy().squeeze()
        return a, a_logprob

    def update_target_nets(self, polyak_tau: float):
        for target_net, actual_net in zip([self.target_critic1, self.target_critic2], [self.critic1, self.critic2]):
            for target_weights, new_weights in zip(target_net.net.trainable_weights, actual_net.net.trainable_weights):
                target_weights.assign((new_weights * polyak_tau) + ((1. - polyak_tau) * target_weights))

    def tune_alpha(self, states, learning_step):
        with tf.GradientTape() as tape:
            _, sa_tilde_logprob = self.actor.policy(states)
            alpha_loss = -1. * tf.reduce_mean(self.log_alpha * sa_tilde_logprob)
        alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(alpha_grad, [self.log_alpha]))
        self.alpha.assign(tf.exp(self.log_alpha))
        tf.summary.scalar("Alpha loss", alpha_loss, learning_step, 'Alpha loss')
