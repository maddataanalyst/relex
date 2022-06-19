import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

from typing import List, Tuple


class ValueNet:
    """
    Basic Value Net implementation.
    """

    def __init__(
            self,
            state_dim: int,
            h_sizes_state: List[int],
            h_act: str = 'relu',
            out_act: str = 'linear',
            initializer: krs.initializers.Initializer = krs.initializers.HeNormal()
    ):
        self.state_dim = state_dim
        self._build_net(h_sizes_state, h_act, out_act, initializer)

    def _build_net(self, h_sizes_state: List[int], h_act: str, out_act: str, initializer: krs.initializers.Initializer) -> krs.Model:
        """
        Prepares an internal network structure.

        Parameters
        ----------
        h_sizes_state: List[int]
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
        s_inp = krs.Input(shape=self.state_dim)
        s_last_out = s_inp
        for h_sz_state in h_sizes_state:
            s_last_out = krs.layers.Dense(units=h_sz_state, activation=h_act, kernel_initializer=initializer)(s_last_out)

        out = krs.layers.Dense(1, out_act, kernel_initializer=initializer)(s_last_out)
        self.net = krs.Model(s_inp, out)

    def state_value(self, s: np.ndarray) -> tf.Tensor:
        """
        Estimates V(s) - state-value function.

        Parameters
        ----------
        s: np.array
            State array (N x dim state)

        Returns
        -------
        tf.Tensor
            State values (N,)
        """
        state_val = self.net(s)
        return tf.squeeze(state_val)

