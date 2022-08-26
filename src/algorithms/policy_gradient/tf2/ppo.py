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

from src.algorithms.policy_gradient.tf2.policy_gradient_commons import PolicyGradientBase

OLD_LOGPROB_PARAM = 'old_logprob'

BATCH_TD_TARGETS_KWARG = 'batch_td_targets'

BATCH_GAE_KWARG = 'batch_gae'


class PPO(PolicyGradientBase):

    """Implementation of a single-threaded PPO algorithm with possible optional extension (e.g. different advantage
    estimations).

    References
    -------
    [1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
    [2] https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(
            self,
            policy_net: policy_nets.BaseStochasticPolicyNet,
            value_net: vnet.ValueNet,
            policy_opt: krs.optimizers.Optimizer,
            value_opt: krs.optimizers.Optimizer,
            advantage: adv.Advantage,
            clipping_eps: float = 1e-6,
            n_agents: int = 3,
            entropy_coef: float = 1e-3,
            action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper()):
        super(PPO, self).__init__(policy_net, value_net, advantage, action_clipper, entropy_coef, policy_opt, value_opt)
        self.n_agents = n_agents
        self.clipping_eps = clipping_eps

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
              log: logging.Logger = logging.getLogger("ppo_logger"),
              lambda_: float = 0.95,
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

        epochs: int
            Total number of epochs for neural net training.

        clip_action: bool
            Should actions be clipped?

        batch_size: int
            Batch size to train NN.

        log: logging.Logger
            Logging object.

        lambda_: float
            GAE discounting factor.

        Returns
        -------
        np.array
            Episode scores.
        """
        all_scores = []
        t0 = time.time()
        losses = []
        learning_step = 0

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0] if type(env.action_space) == gym.spaces.Box else 1

        collection_steps = nepisodes // self.n_agents
        memories = [mbuff.SimpleMemory(max_steps) for _ in range(self.n_agents)]
        for step in range(collection_steps):

            t_now = time.time()
            delta_t = t_now - t0
            if delta_t > max_train_sec:
                break

            step_scores = []
            for mem in memories:
                score = self.run_env(env, max_steps, scaler, mem, clip_action)
                step_scores.append(score)
            avg_score = np.mean(step_scores)
            tf.summary.scalar("Ep score", data=avg_score, step=step)
            all_scores.append(avg_score)

            for epoch in range(epochs):
                mem_losses = []
                mem_indices = sorted(np.arange(len(memories)), key=lambda _: random.random())
                for mem_idx in mem_indices:
                    mem = memories[mem_idx]
                    memory_sample = mem.sample(mem.actual_size, dim_state, dim_action)
                    bidx = 0
                    n_obs_trajectory = memory_sample.states[bidx].shape[0]
                    batch_start = np.arange(0, n_obs_trajectory, batch_size)
                    if n_obs_trajectory > 1:
                        traj_gae, traj_td_targets = self.advantage.calc_advantage_and_returns(
                            memory_sample.rewards[bidx],
                            memory_sample.svals[bidx],
                            memory_sample.sprime_vals[bidx],
                            memory_sample.dones[bidx],
                            gamma,
                            lambda_=lambda_)

                        for start_i in batch_start:
                            from_to = slice(start_i, start_i + batch_size)
                            bi_state = memory_sample.states[bidx][from_to]
                            bi_action = memory_sample.actions[bidx][from_to]
                            bi_logprob = memory_sample.logprobs[bidx][from_to]
                            bi_reward = memory_sample.rewards[bidx][from_to]
                            bi_sprime = memory_sample.sprimes[bidx][from_to]
                            bi_done = memory_sample.dones[bidx][from_to]
                            bi_sval = memory_sample.svals[bidx][from_to]
                            bi_sprime_val = memory_sample.sprime_vals[bidx][from_to]

                            batch_gae = traj_gae[from_to]
                            batch_td_targets = traj_td_targets[from_to]
                            loss = self.learn(
                                bi_state,
                                bi_action,
                                bi_logprob,
                                bi_reward,
                                bi_sprime,
                                bi_done,
                                bi_sval,
                                bi_sprime_val,
                                gamma,
                                learning_step,
                                lambda_,
                                batch_gae=batch_gae,
                                batch_td_targets=batch_td_targets
                            )
                            mem_losses.append(loss)
                            learning_step += 1
                            tf.summary.scalar("Loss", data=loss, step=learning_step)
                if len(mem_losses) > 0:
                    losses.append(np.mean(mem_losses))
            for mem in memories:
                mem.clear_all()
            if step % print_interval == 0:
                autils.log_progress(max_train_sec, average_n_last, all_scores, t0, losses, step, collection_steps, log,
                                    kwargs.get(consts.MLFLOW_LOG, True))

        return np.array(all_scores)

    def run_env(
            self,
            env: gym.wrappers.TimeLimit,
            max_steps: int,
            scaler: object,
            memory: mbuff.SimpleMemory,
            clip_action: bool = True) -> float:
        """
        Function for running single environment instance. Used to collect trajectories, aligned with current policy.
        Direct implementation of lines 2,3 and 4 in Algorithm 1 from paper [1].

        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            Gym env to run in.

        max_steps: int
            Maximal number of steps

        scaler: object
            Scaler object

        memory: mbuff.SimpleMemory
            Memory collector to store transitions.

        clip_action: bool
            Should actions be clipped.
        Returns
        -------
        float
            Episode score
        """
        s = env.reset()
        s = autils.normalize_state(s, scaler)
        done = False
        ep_score = 0

        for step in range(max_steps):

            if done:
                break

            a, a_logprob, done, r, sprime = self.make_step(s, env, clip_action, scaler)

            vs = self.value_net.state_value(np.atleast_2d(s)).numpy().squeeze()
            v_sprime = self.value_net.state_value(np.atleast_2d(sprime)).numpy().squeeze()

            memory.store_transition(s, a, r, sprime, vs, v_sprime, a_logprob, done)
            s = sprime
            ep_score += r
        return ep_score

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
            lambda_: float = 0.95,
            *args,
            **kwargs) -> float:
        """
        Learning step function.

        Parameters
        ----------
        s: np.array
          States of environment.

        a: np.array
          Actions taken in the env.

        a_logprob: np.array
          Logprob(a).

        r: np.array
          Reward signal.

        sprime: np.array
          Next states after transition s--->a--->sprime.

        done: np.array
          Is episode done after transition?

        vs: np.array
          V(s) - state value.

        v_sprime: np.array
          V(sprime) - sprime value.

        gamma: float
        Discounting parameter

        learning_step: int
            Learning iteration step.

        lambda_: float
            Lambda parameter for GAE calculation.

        Returns
        -------
        float
          Total oss after learning step.
        """
        batch_gae = kwargs[BATCH_GAE_KWARG]
        batch_td_targets = kwargs[BATCH_TD_TARGETS_KWARG]

        policy_loss = self.optimize_policy(batch_gae.astype(np.float32), s, a.squeeze(), old_logprob=a_logprob)
        value_loss = self.optimize_value(self.value_opt, np.atleast_2d(s), batch_td_targets.astype(np.float32))
        tf.summary.scalar("Policy loss", policy_loss, step=learning_step)
        tf.summary.scalar("Value loss", value_loss, step=learning_step)
        total_loss = policy_loss.numpy() + np.mean(value_loss.numpy())
        return total_loss

    def optimize_policy(
            self,
            advantage_vals: np.array,
            states: np.array,
            actions: np.array,
            *args,
            **kwargs) -> tf.Tensor:
        if "old_logprob" not in kwargs:
            raise ValueError("PPO policy optimizer requires old_logprobs!")
        old_logprob = kwargs[OLD_LOGPROB_PARAM]
        return self.optimize_policy_ppo(advantage_vals, states, actions, old_logprob)

    def optimize_policy_ppo(self,
                        advantage_vals: np.array,
                        states: np.array,
                        actions: np.array,
                        old_logprob: np.array) -> tf.Tensor:
        """
        Policy optimization step. Directly implements equation (7) from paper [1]. Easy explanation of
        equations derivation can be found in tutorial [2].

        Parameters
        ----------
        advantage_vals: np.array
            Array of advantage estimations.

        states: np.array
            Array of states encountered

        actions: np.array
            Taken actions.

        old_logprob: np.array
            Previously taken actions logprob(a).

        Returns
        -------
        tf.Tensor
            Policy loss.

        References
        -------
        [1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
        [2] https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        N = states.shape[0]
        with tf.GradientTape() as tape:
            new_logprob = tf.squeeze(self.policy_net.get_sa_logprob(states, actions))
            ratio = tf.exp(new_logprob - old_logprob)
            clip_val = tf.clip_by_value(ratio, 1.0 - self.clipping_eps, 1.0 + self.clipping_eps)

            tf.debugging.assert_shapes([
                (new_logprob, (N,)),
                (old_logprob, (N,)),
                (ratio, (N,)),
                (clip_val, (N,))
            ])

            surrogate_loss = -1. * tf.minimum(ratio * advantage_vals, clip_val * advantage_vals)
            entropy = tf.reduce_mean(self.policy_net.get_entropy(states))
            policy_loss = tf.reduce_mean(surrogate_loss) - self.entropy_coef * entropy
        tf.debugging.check_numerics(policy_loss, "Policy loss is NAN or inf")
        policy_grad = tape.gradient(policy_loss, self.policy_net.trainable_variables)
        self.policy_opt.apply_gradients(zip(policy_grad, self.policy_net.trainable_variables))

        for idx, param in enumerate(self.policy_net.trainable_variables):
            tf.debugging.check_numerics(param, "Policy net contains NaNs in weights")

        return policy_loss

