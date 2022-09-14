import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

import src.algorithms.algo_utils as autils
import src.algorithms.commons as rl_commons
import src.algorithms.hybrid.tf2.ddpg as ddpg
import src.algorithms.memory_samplers as mbuff
import src.models.base_models.tf2.policy_nets as pi_net
import src.models.base_models.tf2.q_nets as qnets


class TD3(ddpg.DDPG):
    """
    TD3 implementation. Paper:
    Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, January). Deterministic policy gradient algorithms. In International conference on machine learning (pp. 387-395). PMLR.
    """
    def __init__(self,
                 actor_net: pi_net.DeterministicPolicyNet,
                 actor_net_target: pi_net.DeterministicPolicyNet,
                 critic_net: qnets.QSANet,
                 critic_net_target: qnets.QSANet,
                 critic_net2: qnets.QSANet,
                 critic_net2_target: qnets.QSANet,
                 buffer: mbuff.SimpleMemory,
                 a_min: float,
                 a_max: float,
                 actor_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 critic_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 loss_function: str = 'mean_squared_error',
                 action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper(),
                 polyak_tau: float = 0.005,
                 noise_gen: autils.NoiseGenerator = autils.OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1)),
                 name: str = "TD3",
                 target_update_frequency: int = 1):
        super().__init__(actor_net, actor_net_target, critic_net, critic_net_target, buffer, a_min, a_max, actor_optimizer, critic_optimizer, loss_function, action_clipper, polyak_tau, noise_gen, name="TD3", target_update_frequency=target_update_frequency)
        self.critic_net2 = critic_net2
        self.target_critic2 = critic_net2_target

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
            Action log probs (always None for TD3).
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

        sprime_target_1 = tf.squeeze(self.target_critic.state_action_value(sprime, sprime_action))
        sprime_target_2 = tf.squeeze(self.target_critic2.state_action_value(sprime, sprime_action))
        targets_12 = tf.stack((sprime_target_1, sprime_target_2), axis=1)
        # Equation: Q-target = min Q target {1,2}(s', a')
        target_min = tf.reduce_min(targets_12, axis=1)

        # y = ri + gamma * Q-target(s', a') * (1-done)
        td_target = r + (gamma * (target_min * (1. - done)))

        loss_critic = self.optimize_critic(a, s, td_target)
        if learning_step % 2 == 0:
            loss_actor = self.optimize_actor(s)
            return 0.5 * (loss_critic + loss_actor)
        else:
            return loss_critic

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
        Helper function for optimizing critic networks.

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
        loss_critic_1 = self.optimize_critic_net(a, s, td_target, self.critic_net)
        loss_critic_2 = self.optimize_critic_net(a, s, td_target, self.critic_net2)
        return 0.5 * (loss_critic_1 + loss_critic_2)

    def optimize_critic_net(self, a: np.array, s: np.array, td_target: np.array, critic: qnets.QSANet) -> float:
        """
        Because TD3 has two critic networks - this function optimizes the critic network provided.

        Parameters
        ----------
        a: np.array
            Actions.
        s: np.array
            States
        td_target: np.array
            Calculated TD-targets
        critic: qnets.QSANet
            Critic network

        Returns
        -------
        float
            Loss value
        """
        with tf.GradientTape() as tape:
            mse = krs.losses.MeanSquaredError()
            s_qval = critic.net([s, a], training=True)
            loss_critic = mse(td_target, s_qval)
        grad_critic = tape.gradient(loss_critic, critic.net.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grad_critic, critic.net.trainable_variables))
        return loss_critic
