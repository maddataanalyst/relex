import pytest as pt
import gc

import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

import src.models.base_models.tf2.value_nets as value_nets

N = 3
STATE_DIM = 3


@pt.fixture(autouse=True)
def value_net():
    value_net = value_nets.ValueNet(STATE_DIM, [8])
    yield value_net
    krs.backend.clear_session()
    gc.collect()


def test_state_value_shapes(value_net: value_nets.ValueNet):
    # Given
    s = np.linspace(0, 9., num=9).reshape((N, STATE_DIM))

    # When
    vs = value_net.state_value(s)
    vs = vs.numpy()

    # Then
    assert vs.shape == (N,)
