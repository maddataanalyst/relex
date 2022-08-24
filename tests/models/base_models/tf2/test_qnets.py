import pytest as pt
import gc

import numpy as np
import tensorflow as tf
import src.models.base_models.tf2.q_nets as qnet
import keras as krs


N = 3
STATE_DIM = 3
ACTION_DIM = 2


def test_dueling_qnet():
    # Given
    h_initializer = krs.initializers.HeNormal(seed=123)
    net = qnet.DuelingQNet(STATE_DIM, ACTION_DIM, h_sizes=[16], h_initializer=h_initializer, out_initializer=h_initializer)
    state = tf.convert_to_tensor(np.array([
        [1., 2., 3],
        [100, 300, 200.]
    ]))
    expected_qsa = np.array([
        [-12.221, -2.387],
        [-1236.357, -424.94]
    ])

    # When
    q_sa = net.state_action_value(state).numpy().round(3)

    # Then
    np.testing.assert_almost_equal(expected_qsa, q_sa, decimal=3)