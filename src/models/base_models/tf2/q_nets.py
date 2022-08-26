import tensorflow as tf
import numpy as np
import tensorflow.keras as krs

from enum import Enum
from typing import List, Tuple


class AdvantageNorm(Enum):
    MAX = 0
    MEAN = 1


def polyak_tau_update_networks(target_net: krs.models.Model, online_net: krs.models.Model, polyak_tau: float = 0.005):
    """
    Implementation of Polyak Averaging procedure - for blending target networks and acting networks.

    Parameters
    ----------
    target_net: krs.models.Model
        Model to blend weights into.

    online_net: krs.models.Model
        Keras model that gives weights.

    polyak_tau: float
        Blending parameter. Should be between [0-1]

    References
    -------
    Piché, A., Marino, J., Marconi, G. M., Pal, C., & Khan, M. E. (2021). Beyond Target Networks: Improving Deep $ Q $-learning with Functional Regularization. arXiv preprint arXiv:2106.02613.
    """
    if polyak_tau < 0 or polyak_tau > 1:
        raise ValueError("Polyak tau needs to be in range [0, 1]")
    for online_w, target_w in zip(online_net.trainable_weights, target_net.trainable_weights):
        target_w.assign((polyak_tau * online_w) + ((1. - polyak_tau) * target_w))


class QNet:
    """
    Q net implementation for DISCRETE action spaces, that takes state and outputs Q-values for all actions.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            h_sizes: List[int],
            h_act: str = 'relu',
            out_act: str = 'linear',
            h_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            batch_norm: bool = False
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h_sizes = h_sizes
        self._build_net(h_sizes, h_act, out_act, h_initializer, out_initializer, batch_norm)

    def _build_net(self, h_sizes: List[int], h_act: str, out_act: str, h_initializer: krs.initializers.Initializer,
                   out_initializer: krs.initializers.Initializer, batch_norm: bool = False) -> krs.Model:
        """
        Prepares an internal network structure.

        Parameters
        ----------
        h_sizes: List[int]
            List of hidden layer sizes.

        h_act: str
            Hidden layer activations.

        out_act: str
            Output layer activation.

        h_initializer, out_initializer: krs.initializers.Initializer
            Keras initializers.

        batch_norm: bool
            Should batch norm be added?

        Returns
        -------
        krs.Model
            Model network.
        """
        model = krs.models.Sequential()
        if batch_norm:
            model.add(krs.layers.BatchNormalization(input_dim=self.state_dim))
        else:
            for l_idx, h_size in enumerate(h_sizes):
                if l_idx == 0:
                    model.add(krs.layers.Dense(h_size, kernel_initializer=h_initializer, activation=h_act,
                                               input_dim=self.state_dim))
                else:
                    model.add(krs.layers.Dense(h_size, kernel_initializer=h_initializer, activation=h_act))
        model.add(krs.layers.Dense(units=self.action_dim, activation=out_act, kernel_initializer=out_initializer))
        self.net = model

    def state_action_value(self, s: np.ndarray, *args, **kwargs) -> tf.Tensor:
        """
        Estimates Q(s, a) - state-action function, given S for all actions: Q(s,a1), Q(s,a2), etc.

        Parameters
        ----------
        s: np.array
            State array (N x dim state)

        Returns
        -------
        tf.Tensor
            State action values (N,dim-A)
        """
        q_val = self.net(s, *args, **kwargs)
        return q_val


class QSANet:
    """
    Basic Q-net implementation, that takes s,a and outputs Q(s,a) value (single value).
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            shared_sizes: List[int],
            a_sizes: List[int],
            state_sizes: List[int],
            h_act: str = 'relu',
            out_act: str = 'linear',
            initializer: krs.initializers.Initializer = krs.initializers.HeNormal()
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._build_net(a_sizes, state_sizes, shared_sizes, h_act, out_act, initializer)

    def _build_net(self, a_sizes: List[int], s_sizes: List[int], shared_sizes: List[int], h_act: str, out_act: str,
                   initializer: krs.initializers.Initializer) -> krs.Model:
        """
        Prepares an internal network structure.

        Parameters
        ----------
        a_sizes: List[int]
            List of action-branch sizes.

        s_sizes: List[int]
            List of state-branch sizes.

        shared_sizes: List[int]
            List of hidden layer sizes.

        h_act: str
            Hidden layer activations.

        out_act: str
            Output layer activation.

        Returns
        -------
        krs.Model
            Model network.
        """
        a_inp = krs.Input(shape=(self.action_dim,))
        a_last_out = a_inp
        for a_size in a_sizes:
            a_last_out = krs.layers.Dense(units=a_size, activation=h_act)(a_last_out)

        # state branch
        s_inp = krs.Input(shape=(self.state_dim,))
        s_last_out = s_inp
        for s_size in s_sizes:
            s_last_out = krs.layers.Dense(units=s_size, activation=h_act)(s_last_out)

        sa_out = krs.layers.concatenate([a_last_out, s_last_out])
        last_sa_out = sa_out
        for h_sz_stack in shared_sizes:
            last_sa_out = krs.layers.Dense(h_sz_stack, activation=h_act)(last_sa_out)
        out = krs.layers.Dense(1, out_act)(last_sa_out)
        self.net = krs.Model([s_inp, a_inp], out)

    def state_action_value(self, s: np.ndarray, a: np.array, *args, **kwargs) -> tf.Tensor:
        """
        Estimates Q(s, a) - state-action function.

        Parameters
        ----------
        s: np.array
            State array (N x dim state)

        a: np.array
            action array (N x dim action)

        Returns
        -------
        tf.Tensor
            State values (N,)
        """
        q_val = self.net([s, a], *args, **kwargs)
        return q_val


class DuelingNetModel(krs.Model):
    """Wrapper for Keras model, for building dueling networks - networks with two outputs:
        1. First - V(s)
        2. Second - Advantage"""

    def __init__(self, state_dim: int,
                 action_dim: int,
                 h_sizes: List[int],
                 h_act: str = 'relu',
                 out_act: str = 'linear',
                 h_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
                 out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
                 batch_norm: bool = False,
                 advantage_norm: AdvantageNorm = AdvantageNorm.MEAN):
        super().__init__()
        self.advantage_norm = advantage_norm
        if batch_norm:
            self.batch_norm_l = krs.layers.BatchNormalization()
        self.batch_norm = batch_norm
        self.hidden_layers = [krs.layers.Dense(h_size, kernel_initializer=h_initializer, activation=h_act) for h_size in
                              h_sizes]
        self.v_layer = krs.layers.Dense(units=1, activation=out_act, kernel_initializer=out_initializer)
        self.adv_layer = krs.layers.Dense(units=action_dim, activation=out_act, kernel_initializer=out_initializer)

    def call(self, input_data) -> tf.Tensor:
        out1 = input_data
        if self.batch_norm:
            out1 = self.batch_norm_l(input_data)

        hidden_out = out1
        for layer in self.hidden_layers:
            hidden_out = layer(hidden_out)

        v_s = self.v_layer(hidden_out)
        adv_raw = self.adv_layer(hidden_out)

        adv_norm = self._normalize_advantage(adv_raw)

        q_sa = v_s + (adv_raw - adv_norm)
        return q_sa

    def _normalize_advantage(self, adv_raw: tf.Tensor) -> tf.Tensor:
        """
        Advantage normalization function. Two possible options are mentioned in literature - mean, and max
        normalization.

        Parameters
        ----------
        adv_raw: tf.Tensor
            Raw advantage values

        Returns
        -------
        tf.Tensor
            Normalized advantage.

        References
        -------
        Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

        """
        if self.advantage_norm == AdvantageNorm.MEAN:
            adv_norm = tf.math.reduce_mean(adv_raw, axis=1, keepdims=True)

        elif self.advantage_norm == AdvantageNorm.MAX:
            adv_norm = tf.math.reduce_max(adv_raw, axis=1, keepdims=True)
        return adv_norm

    def advantage(self, state: tf.Tensor) -> tf.Tensor:
        """
        Advantage calculation function: A(s,a) = Q(s,a) - V(s).

        Parameters
        ----------
        state: tf.Tensor
            Current state s(t)

        Returns
        -------
        tf.Tensor
            Advantage.
        """
        out1 = state
        if self.batch_norm:
            out1 = self.batch_norm_l(state)

        hidden_out = out1
        for layer in self.hidden_layers:
            hidden_out = layer(hidden_out)

        adv_raw = self.adv_layer(hidden_out)
        return adv_raw


class DuelingQNet(QNet):

    """Agent/training algorithm for dueling DQN."""

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            h_sizes: List[int],
            h_act: str = 'relu',
            out_act: str = 'linear',
            h_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            out_initializer: krs.initializers.Initializer = krs.initializers.HeNormal(),
            batch_norm: bool = False,
            advantage_norm: AdvantageNorm = AdvantageNorm.MEAN
    ):
        self.advantage_norm = advantage_norm
        self.action_dim = action_dim
        super().__init__(state_dim, action_dim, h_sizes, h_act, out_act, h_initializer, out_initializer, batch_norm)

    def _build_net(self, h_sizes: List[int], h_act: str, out_act: str, h_initializer: krs.initializers.Initializer,
                   out_initializer: krs.initializers.Initializer, batch_norm: bool = False) -> krs.Model:
        """
        Prepares an internal network structure for dueling netwrok - outputs V(s) and Q(s,a).

        Parameters
        ----------
        h_sizes: List[int]
            List of hidden layer sizes.

        h_act: str
            Hidden layer activations.

        out_act: str
            Output layer activation.

        h_initializer, out_initializer: krs.initializers.Initializer
            Keras initializers.

        batch_norm: bool
            Should batch norm be added?

        Returns
        -------
        krs.Model
            Model network.
        """
        self.net = DuelingNetModel(self.state_dim, self.action_dim, self.h_sizes, h_act, out_act, h_initializer,
                                   out_initializer, batch_norm, self.advantage_norm)

    def state_action_value(self, s: np.ndarray, *args, **kwargs) -> tf.Tensor:
        return self.net(s)

    def advantage(self, s: np.ndarray, *args, **kwargs) -> tf.Tensor:
        return self.net.advantage(s)
