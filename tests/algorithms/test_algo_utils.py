import pytest as pt
import numpy as np

import src.algorithms.algo_utils as au


def test_decay_rates():
    # given
    initial_val = 0.9
    minimal_val = 0.1
    decay_ratio = 0.9
    n_episodes = 10

    expected_values = np.array([0.9, 0.55, 0.35, 0.24, 0.17, 0.14, 0.12, 0.11, 0.1, 0.1])

    # when
    actual_values = au.decay_schedule_epsilon(initial_val, minimal_val, decay_ratio, n_episodes).round(2)

    # the
    np.testing.assert_array_equal(expected_values, actual_values)


@pt.mark.parametrize('decay_rate,msg', [
    (0.01, au.TOO_FEW_EPISODES_MSG),
    (0.1, au.NANS_EPISLON)
])
def test_decay_epsilon_too_few_episodes(decay_rate, msg):
    # given
    initial_val = 0.9
    minimal_val = 0.1
    n_episodes = 10

    with pt.raises(ValueError) as exc:
        au.decay_schedule_epsilon(initial_val, minimal_val, decay_rate, n_episodes)

    assert msg in exc.value.args[0]
