import pytest as pt
import numpy as np
import numpy.testing as npt

import src.algorithms.advantages as adv


def test_gae():
    # given
    rewards = np.array([0.5] * 5)
    rewards[-1] = 1.

    vs = np.array([1., 2., 3., 4., 5.])
    vs_prime = list(vs[1:]) + [0.]
    dones = np.array([0., 0., 0., 0., 1.])
    gamma = 0.9
    lambda_ = 0.9

    expected_gae = np.array([1.803, 0.621, -0.714, -2.24, -4.])
    expected_returns = np.array([2.803, 2.621, 2.285, 1.76, 1.])

    # when
    actual_gae, actual_returns = adv.GAE.calc_advantage_and_returns(rewards, vs, vs_prime, dones, gamma, lambda_,
                                                                    normalize=False)
    # then
    npt.assert_almost_equal(actual_gae, expected_gae, 3)
    npt.assert_almost_equal(actual_returns, expected_returns, 3)


def test_simple_advantage():
    # given
    rewards = np.array([1., 2., 3., 4., 5.])
    dones = np.array([0., 0., 0., 0., 1.])
    vs = np.array([10., 12., 13., 14., 15.])

    expected_advantages = np.array([1.4265, -0.415, -2.35, -5.5, -10.])
    expected_returns = np.array([11.4265, 11.585, 10.65, 8.5, 5.])

    # when
    advantages, returns = adv.BaselineAdvantage.calc_advantage_and_returns(
        rewards=rewards,
        vs=vs,
        vs_prime=None,
        dones=dones,
        gamma=0.9
    )

    # then
    np.testing.assert_almost_equal(advantages, expected_advantages, decimal=4)
    np.testing.assert_almost_equal(returns, expected_returns, decimal=4)
