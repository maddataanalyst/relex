import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

from tensorflow_probability.python.distributions import MultivariateNormalDiag, Normal, Categorical
from typing import List, Tuple


class BasePolicyNet(krs.Model):
    """Abstract base class for policy networks"""
    def __init__(
            self,
            a_dim: int,
            h_sizes: List[int],
            h_act: str = 'relu',
            out_act: str = 'linear',
            hidden_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal()):
        super(BasePolicyNet, self).__init__()
        self.a_dim = a_dim
        self.h_sizes = h_sizes
        self.h_act = h_act
        self.out_act = out_act
        self.hidden_initializer = hidden_initializer
        self.out_initializer = out_initializer
        self._build_network()


    def _build_network(self):
        self.dense_layers = []
        for h_size in self.h_sizes:
            self.dense_layers.append(krs.layers.Dense(h_size, self.h_act, kernel_initializer=self.hidden_initializer))
        self.out_layer = krs.layers.Dense(self.a_dim, self.out_act, kernel_initializer=self.out_initializer)

    def policy(self, s: np.array, training: bool = True, *args, **kwargs) -> Tuple[
        tf.Tensor, tf.Tensor]:
        """
        Calculates a policy for a given state.

        Parameters
        ----------
        s: np.array
            Current state.
        training: bool
            Is this action calculated in a training mode?

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Tuple (a, None)
            Action is an array of (1xAction dim).
            Second argument is for compatibility with Stochastic Policy, where log prob is returned.
        """
        raise NotImplementedError("Implement me")


class DeterministicPolicyNet(BasePolicyNet):
    """Base class for deterministic policy implementations"""

    def __init__(
            self,
            a_dim: int,
            h_sizes: List[int],
            a_max: int,
            h_act: str = 'relu',
            out_act: str = 'linear',
            hidden_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal()):
        self.a_max = a_max
        super(DeterministicPolicyNet, self).__init__(a_dim, h_sizes, h_act, out_act, hidden_initializer, out_initializer)

    def call(self, input, *args, **kwargs):
        hidden_out = input
        for dense_l in self.dense_layers:
            hidden_out = dense_l(hidden_out)
        out = self.out_layer(hidden_out) * self.a_max
        return out

    def policy(self, s: np.array, training: bool = True, *args, **kwargs) -> Tuple[
        tf.Tensor, tf.Tensor]:

        """
        Calculates a DETERMINISTIC policy for a given state.

        Parameters
        ----------
        s: np.array
            Current state.
        training: bool
            Is this action calculated in a training mode?

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Tuple (a, None)
            Action is an array of (1xAction dim).
            Second argument is for compatibility with Stochastic Policy, where log prob is returned.
        """
        action = self.call(np.atleast_2d(s))
        return action, None


class BaseStochasticPolicyNet(BasePolicyNet):
    """
    Base class for policy networks - forms an API and fundamental operations.
    """

    def __init__(
            self,
            a_dim: int,
            h_sizes: List[int],
            h_act: str = 'relu',
            out_act: str = 'linear',
            hidden_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal()):
        super(BaseStochasticPolicyNet, self).__init__(a_dim, h_sizes, h_act, out_act, hidden_initializer, out_initializer)
        self.a_dim = a_dim
        self.h_sizes = h_sizes
        self.h_act = h_act
        self.out_act = out_act
        self.hidden_initializer = hidden_initializer
        self.out_initializer = out_initializer
        self._build_network()

    def _build_network(self, *args, **kwargs):
        """
        Builds a model network.
        """
        self.dense_layers = []
        for h_size in self.h_sizes:
            self.dense_layers.append(krs.layers.Dense(h_size, self.h_act, kernel_initializer=self.hidden_initializer))
        self.out_layer = krs.layers.Dense(self.a_dim, self.out_act, kernel_initializer=self.out_initializer)

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()

    def policy(self, s: np.array, training: bool = True, deterministic_action: bool = False, *args, **kwargs) -> Tuple[
        tf.Tensor, tf.Tensor]:
        """
        Implements a pi(a|s) operation - returning an action given a state.

        Parameters
        ----------
        s: np.array
            An array of states (N x dim state)

        training: bool
            Is the operation performed during training or not.

        deterministic_action: bool
            Defaults False. Should the action returned be deterministic?

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing an action and its log-probability.
        """
        distrib = self._get_distrib(s)
        action = distrib.sample()
        action_log_prob = distrib.log_prob(action)
        return action, action_log_prob

    def get_sa_logprob(self, s: np.array, a: np.array, *args, **kwargs) -> tf.Tensor:
        """
        Given past states and actions taken - returns log probabilities from a policy distribution. Required by some
        policy gradient algorithms during optimization steps.

        Parameters
        ----------
        s: np.array
            An array of states (N x dim state)

        a: np.array
            An array of actions (N x dim action)

        Returns
        -------
        tf.Tensor
            Log probabilities tensor (N,)
        """
        distrib = self._get_distrib(s)
        if distrib.event_shape.ndims != 0:
            a = np.reshape(a, (distrib.batch_shape[0], distrib.event_shape[0]))
        else:
            a = np.reshape(a, (distrib.batch_shape[0],))
        action_log_probs = distrib.log_prob(a)
        return action_log_probs

    def get_entropy(self, s: np.array) -> tf.Tensor:
        """
        Gets the entropy of a policy distribution for a set of states.

        Parameters
        ----------
        s: np.array
            An array of states (N x dim state)

        Returns
        -------
        tf.Tensor
            A Tensor/vector of entropy values for each state, (N,).

        """
        distrib = self._get_distrib(s)
        return distrib.entropy()

    def _get_distr_params(self, s: np.array):
        raise NotImplementedError()

    def _get_distrib(self, s: np.array):
        raise NotImplementedError()

    def build_graph(self):
        x = krs.layers.Input(shape=(self.dense_layers[0].input_dim))
        return krs.models.Model(inputs=[x], outputs=self.call(x))


class ContinuousPolicyNet(BaseStochasticPolicyNet):
    """
    Policy network for continuous action spaces. Instead of operating on the
    """

    def __init__(self, state_dim: int,
                 a_dim: int,
                 h_sizes: List[int],
                 h_act: str = 'relu',
                 mean_out_act: str = 'linear',
                 hidden_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
                 out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
                 log_std: float = -0.7,
                 std_min: float = 1e-6,
                 std_max: float = 1.0):
        super(ContinuousPolicyNet, self).__init__(a_dim, h_sizes, h_act, mean_out_act, hidden_initializer, out_initializer)
        self.std_min = std_min
        self.std_max = std_max
        self.log_std = tf.Variable(
            name="action_log_std",
            initial_value=tf.ones((a_dim, ), dtype='float32') * log_std,
            trainable=True,
        )

    def call(self, inputs: np.array, training=None, mask=None):
        last_out = inputs
        for dense_l in self.dense_layers:
            last_out = dense_l(last_out)
        mean_out = self.out_layer(last_out)
        log_std_out = tf.ones_like(mean_out) * self.log_std
        std = tf.clip_by_value(tf.exp(log_std_out), self.std_min, self.std_max)
        return mean_out, std

    def _get_distr_params(self, s: np.array):
        return self.call(s)

    def _get_distrib(self, s: np.array):
        mu, std = self._get_distr_params(s)
        distrib = MultivariateNormalDiag(mu, scale_diag=std)
        return distrib


class ContinuousPolicyNetReparam(ContinuousPolicyNet):

    def __init__(self, state_dim: int,
                 a_dim: int,
                 h_sizes: List[int],
                 h_act: str = 'relu',
                 mean_out_act: str = 'linear',
                 hidden_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
                 out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
                 log_std: float = -0.7,
                 std_min: float = 1e-6,
                 std_max: float = 1.0,
                 reparam_eps: float = 1e-5):
        super(ContinuousPolicyNetReparam, self).__init__(state_dim, a_dim, h_sizes, h_act, mean_out_act, hidden_initializer, out_initializer, log_std, std_min, std_max)
        self.reparam_eps = reparam_eps

    def _get_distrib(self, s: np.array):
        mu, std = self._get_distr_params(s)
        distrib = Normal(loc=mu, scale=std)
        return distrib

    def policy(
            self,
            s: np.array,
            training: bool = True,
            deterministic_action: bool = False,
            *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        distrib = self._get_distrib(s)
        action = distrib.sample()
        original_logprob = distrib.log_prob(action)

        reparam_logprob, squashed_a = self._calc_reparametrized_logprobs(action, original_logprob)
        return squashed_a, reparam_logprob

    def _calc_reparametrized_logprobs(self, action, action_log_prob):
        squashed_a = tf.math.tanh(action)
        log_probs = action_log_prob - tf.math.log(1. - tf.math.pow(squashed_a, 2) + self.reparam_eps)
        log_probs = tf.reduce_sum(log_probs, 1)
        return log_probs, squashed_a

    def get_sa_logprob(self, s: np.array, a: np.array, *args, **kwargs) -> tf.Tensor:
        distrib = self._get_distrib(s)
        original_logprob = distrib.log_prob(a)
        reparam_logprob, _ = self._calc_reparametrized_logprobs(a, original_logprob)
        return reparam_logprob


class DiscretePolicyNet(BaseStochasticPolicyNet):
    """
    An implementation of a policy network for a continuous action spaces policy.
    """

    def __init__(
            self,
            a_dim: int,
            h_sizes: List[int],
            h_act: str = 'relu',
            out_act: str = 'softmax',
            hidden_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal()):
        super().__init__(a_dim, h_sizes, h_act, out_act, hidden_initializer, out_initializer)

    def call(self, inputs, training=None, mask=None):
        last_out = inputs
        for dense_l in self.dense_layers:
            last_out = dense_l(last_out)
        out = self.out_layer(last_out)
        return out

    def _get_distr_params(self, s: np.array):
        return self(s)

    def _get_distrib(self, s: np.array):
        prob = self._get_distr_params(s)
        distrib = Categorical(probs=prob)
        return distrib
