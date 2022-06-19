import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

from typing import List, Tuple


class QNet:
    """
    Baseic Q-net implementation
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

    def _build_net(self, a_sizes: List[int], s_sizes: List[int], shared_sizes: List[int],  h_act: str, out_act: str, initializer: krs.initializers.Initializer) -> krs.Model:
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
        return q_val #tf.squeeze(q_val)

