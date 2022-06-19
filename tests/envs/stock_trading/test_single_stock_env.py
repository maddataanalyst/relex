import pytest as pt
import pandas as pd
import numpy as np
import datetime as dt

import src.envs.stock_trading.single_stock_env as sse


def test_generate_lagged_data():
    # given
    y = pd.Series(np.arange(1, 6))
    expected = pd.DataFrame({
        'y': [1, 2, 3, 4, 5],
        'y-1': [np.nan, 1, 2, 3, 4],
        'y-2': [np.nan, np.nan, 1, 2, 3]
    })

    # when
    actual = sse.generate_lagged_data(y, 3).reset_index(drop=True)

    # then
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


def test_time_frames_in_bounds():
    # Given
    INITIAL_CASH = 5000
    env = sse.SingleStockEnv(
        'EUNL.DE',
        start_date='2020-01-06',
        end_date='2020-01-15',
        stochastic=False,
        initial_resources=INITIAL_CASH,
        window_size=2)
    actual_cash = []
    actual_pnls = []
    actual_dates = []
    max_steps = 10

    # When
    done = False
    step = 0
    env.reset()
    while not done:
        if step >= max_steps:
            break
        sprime, pnl, done, _ = env.step(sse.ACTION_HOLD)
        actual_cash.append(env.cash)
        actual_pnls.append(pnl)
        if not done:
            actual_dates.append(env.current_date)
        step += 1

    # Then
    assert env.ohlc_data.shape == (5, 13)
    assert env.features.shape == (5, 9)
    assert step == 4
    assert actual_pnls == [0.] * 4
    assert actual_cash == [INITIAL_CASH] * 4
    assert actual_dates == [dt.date(2020, 1, 9), dt.date(2020,  1, 10), dt.date(2020, 1, 13)]


def test_buy_sell():
    # Given
    INITIAL_CASH = 5000
    env = sse.SingleStockEnv(
        'EUNL.DE',
        start_date='2020-01-06',
        end_date='2020-01-15',
        stochastic=False,
        initial_resources=INITIAL_CASH,
        window_size=2)
    actual_cash = []
    actual_pnls = []
    dates = []
    expected_pnls = np.array([0., 31.32, 26.27, 20.36, 41.76])
    expected_cash = np.array([29.17, 29.17, 29.17, 29.17, 5041.76])

    # When
    env.reset()
    actions = [sse.ACTION_BUY, sse.ACTION_HOLD, sse.ACTION_HOLD, sse.ACTION_HOLD, sse.ACTION_SELL]
    for action in actions:
        sprime, pnl, done, _ = env.step(action)
        actual_cash.append(env.cash)
        actual_pnls.append(pnl)
        if not done:
            dates.append(env.current_date)
    actual_pnls = np.array(actual_pnls).round(2)
    actual_cash = np.array(actual_cash).round(2)

    # Then
    assert done
    assert env.ohlc_data.shape == (5, 13)
    assert env.features.shape == (5, 9)
    np.testing.assert_almost_equal(actual_pnls, expected_pnls, 2)
    np.testing.assert_almost_equal(actual_cash, expected_cash, 2)
    assert dates == [dt.date(2020, 1, 9), dt.date(2020,  1, 10), dt.date(2020, 1, 13)]


# TODO: test lstm features