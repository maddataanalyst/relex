import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

from typing import List, Tuple


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
                    model.add(krs.layers.Dense(h_size, kernel_initializer=h_initializer, activation=h_act, input_dim=self.state_dim))
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
    Basic Q-net implementation, that takes s,a and outputs Q(s,a) value
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
        return q_val  # tf.squeeze(q_val)
