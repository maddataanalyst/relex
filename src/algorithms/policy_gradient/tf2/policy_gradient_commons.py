from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

import src.algorithms.advantages as adv
import src.algorithms.commons as rl_commons
import src.models.base_models.tf2.policy_nets as policy_nets
import src.models.base_models.tf2.value_nets as vnets
from src.algorithms.commons import ActionClipper


class PolicyGradientBase(rl_commons.RLAgent):

    """A class that gathers common functionalities for all policy gradient algorithms. Factors-out some common
     abstractions like optimize policy net, optimize value net, etc.
    """

    def __init__(
            self,
            policy_net: policy_nets.BaseStochasticPolicyNet,
            value_net: vnets.ValueNet,
            advantage: adv.Advantage,
            action_clipper: ActionClipper = rl_commons.DefaultActionClipper(),
            entropy_coef: float = 1e-3,
            policy_opt: krs.optimizers.Optimizer = krs.optimizers.Adam(1e-3),
            value_opt: krs.optimizers.Optimizer = krs.optimizers.Adam(1e-3),
            name: str = 'agent'):
        super(PolicyGradientBase, self).__init__(action_clipper, name)
        self.policy_net = policy_net
        self.value_net = value_net
        self.advantage = advantage
        self.entropy_coef = entropy_coef
        self.policy_opt = policy_opt
        self.value_opt = value_opt

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        """
        Generic function for choosing action in a given state.
        Parameters
        ----------
        s: np.array
            Current state
        Returns
        -------
        Tuple[np.array, np.array]
            Tuple of selected action and action log prob

        """
        a, a_logprob = self.policy_net.policy(np.atleast_2d(s))
        a = a.numpy().squeeze()
        a_logprob = a_logprob.numpy().squeeze()
        return a, a_logprob

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
            learning_step: int, *args, **kwargs) -> float:
        """
        Generic learning function that performs a single learning iteration.

        Parameters
        ----------
        s: np.array
            Array of current states, either a single state or a batch of states.
            Dim: (N, dim s)
        a: np.array
            Array of actions selected in states s.
            Dim: (N, dim s)
        a_logprob: np.array
            Log probs of actions a in states s.
            Dim: (N, )
        r: np.array
            Rewards collected in the process.
            Dim: (N, )
        sprime: np.array
            An array of sprimes after executing action a in state s.
            Dim: (N, dim s)
        done: np.array
            An array of indicators if episode was done after (s,a,sprime) series.
            Dim: (N, )
        vs: np.array
            An array of state-values V(s).
            Dim: (N, )
        v_sprime: np.array
            An array of sprime-state values V(s')
            Dim:: (N, )
        gamma: float
            Discounting factor for rewards.
        learning_step: float
            Learning parameter for algorithm.
        
        Returns
        -------
        float
            Total loss, combined policy and state value loss.

        """
        advantages, td_targets = self.advantage.calc_advantage_and_returns(r, vs, v_sprime, done, gamma, *args,
                                                                           **kwargs)
        policy_loss = self.optimize_policy(
            advantage_vals=advantages.astype(np.float32),
            states=s,
            actions=a)
        value_loss = self.optimize_value(
            value_optimizer=self.value_opt,
            states=s,
            target_vals=td_targets)

        tf.summary.scalar("Policy loss", policy_loss, step=learning_step)
        tf.summary.scalar("Value loss", value_loss, step=learning_step)
        total_loss = policy_loss.numpy() + np.mean(value_loss.numpy())
        return total_loss

    def optimize_policy(
            self,
            advantage_vals: np.array,
            states: np.array,
            actions: np.array, *args, **kwargs) -> tf.Tensor:
        """
        Generic policy network optimization function.
        Parameters
        ----------
        advantage_vals: np.array
            Advantage values calculated using any available method
            e.g. A(s,a) = Q(s,a) - V(s).
            Dim should be (N, ).
            Available advantage methods are described in paper: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438
        states: np.array
            Array of states s. Dim should be (N, dim S).
        actions: np.array
            Actions performed in state s. Dim should be (N, dim A).

        Returns
        -------
        tf.Tensor
            Tensor of policy total loss. Dim should be: (N,)

        """
        N = states.shape[0]
        tf.debugging.assert_shapes([
            (advantage_vals, (N,))
        ])
        with tf.GradientTape() as tape:
            a_logprobs = tf.squeeze(self.policy_net.get_sa_logprob(states, actions))
            loss = tf.reduce_mean(-1 * a_logprobs * advantage_vals)

            entropy_val = self.policy_net.get_entropy(states)
            entropy_loss = -1 * self.entropy_coef * tf.reduce_mean(entropy_val)

            total_loss = loss + entropy_loss
        delta_pi = tape.gradient(total_loss, self.policy_net.trainable_variables)
        self.policy_opt.apply_gradients(zip(delta_pi, self.policy_net.trainable_variables))
        for idx, param in enumerate(self.policy_net.trainable_variables):
            tf.debugging.check_numerics(param, "Policy net contains NaNs in weights")

        return total_loss

    def optimize_value(
            self,
            value_optimizer: krs.optimizers.Optimizer,
            states: np.array,
            target_vals: np.array) -> tf.Tensor:
        """
        Generic state-value optimization function.

        Parameters
        ----------
        value_optimizer: krs.optimzers.Optimizer
            Optimizer selected for a value-function.
        states: np.array
            Array of states. Dim: (N, dim s).
        target_vals: np.array
            V(s) target values, eg. TD targets.
            Dim: (N, )

        Returns
        -------
        tf.Tensor
            Tensor of total loss for state-value.
        """
        N, dim_s = states.shape
        with tf.GradientTape() as tape:
            state_v = self.value_net.state_value(states)
            tf.debugging.assert_shapes([
                (state_v, (N,))
            ])
            loss = krs.losses.mean_squared_error(target_vals, state_v)
        tf.debugging.check_numerics(loss, "Value loss is NAN or inf!")
        value_grad = tape.gradient(loss, self.value_net.net.trainable_variables)
        value_optimizer.apply_gradients(zip(value_grad, self.value_net.net.trainable_variables))

        for idx, param in enumerate(self.value_net.net.trainable_variables):
            tf.debugging.check_numerics(param, "Policy net contains NaNs in weights")
        return loss
