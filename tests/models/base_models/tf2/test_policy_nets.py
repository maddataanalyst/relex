import pytest as pt
import gc

import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

import src.models.base_models.tf2.policy_nets as policynet

N = 3
STATE_DIM = 3
ACTION_DIM = 2


@pt.fixture(autouse=True)
def continuous_policy_net():
    policy_net = policynet.ContinuousPolicyNet(STATE_DIM, ACTION_DIM, [16, 8], 'relu', 'tanh')
    yield policy_net
    krs.backend.clear_session()
    gc.collect()


def test_continuous_policy_output_shapes(continuous_policy_net: policynet.ContinuousPolicyNet):
    # Given
    s = np.linspace(0, 9., num=9).reshape((N, STATE_DIM))

    # When
    action, logprob = continuous_policy_net.policy(s)
    action = action.numpy()
    logprob = logprob.numpy()

    # Then
    assert action.shape == (N, ACTION_DIM)
    assert logprob.shape == (N,)


def test_continuous_policy_salogprob_shapes(continuous_policy_net: policynet.ContinuousPolicyNet):
    # Given
    s = np.linspace(0, 9., num=9).reshape((N, STATE_DIM))
    a = np.random.rand(N, ACTION_DIM)
    # When
    logprob = continuous_policy_net.get_sa_logprob(s, a)
    logprob = logprob.numpy()

    # Then
    assert logprob.shape == (N,)


def test_continuous_policy_entropy_shapes(continuous_policy_net: policynet.ContinuousPolicyNet):
    # Given
    s = np.linspace(0, 9., num=9).reshape((N, STATE_DIM))
    # When
    entropy = continuous_policy_net.get_entropy(s)
    entropy = entropy.numpy()

    # Then
    assert entropy.shape == (N,)
