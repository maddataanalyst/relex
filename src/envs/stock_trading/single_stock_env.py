import datetime as dt

import gym
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

from src.algorithms.commons import RLAgent

HIGH_DIFF_ROLLING_STD = 'high_diff_rolling_std'

HIGH_DIFF_ROLLING_MU = 'high_diff_rolling_mu'

LOW_DIFF_ROLLING_STD = 'low_diff_rolling_std'

LOW_DIFF_ROLLING_MU = 'low_diff_rolling_mu'

PRICE_DIFF_ROLLING_STD = 'price_diff_rolling_std'

PRICE_DIFF_ROLLING_MU = 'price_diff_rolling_mu'

PRICE = 'price'
LOW = 'Low'
HIGH = 'High'
CLOSE = 'Close'
ADJ_CLOSE = "Adj Close"

PRICE_DIFF = 'price_diff'
MODE_DIFF = 'diff'
MODE_PCT_CHANGE = 'pct_change'
ALLOWED_MODES = [MODE_DIFF, MODE_PCT_CHANGE]

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ALLOWED_ACTIONS = [ACTION_HOLD, ACTION_SELL, ACTION_BUY]


def generate_lagged_data(y: pd.Series, k_lags: int) -> pd.DataFrame:
    """
    Code taken from a blogpost: https://filip-wojcik.com/en/post/rnns/fifty-shapes-of-time/
    Generates a lagged time series data in a fromat y, y-1, y-2, ..., y-K up to
    K-th lag.

    Parameters
    ----------
    y : pd.Series
        Original time series.
    k_lags: int
        Number of lags to generate.

    Returns
    ----------
    pd.DataFrame
        A lagged dataframe in format y, y-1, y-2, ..., y-K.
    """
    data = pd.DataFrame({'y': y})
    for t in range(k_lags):
        fname = f"y-{t}" if t > 0 else "y"
        lagged = y.shift(t)
        data[fname] = lagged
    return data


class SingleStockEnv(gym.Env):

    def __init__(
            self,
            ticker: str,
            start_date: dt.datetime = None,
            end_date: dt.datetime = None,
            mode: str = MODE_DIFF,
            window_size: int = 64,
            initial_resources: float = 100000,
            stochastic: bool = False,
            lagged_features: bool = False,
            cutoff_period: dt.datetime = None):
        """
        Initializes env with predefined settings: a ticker, prices period and mode of price differencing.
        Parameters
        ----------
        ticker: str
            Financial instrument/ticker to be initialized with. Must be one of the handled by yfinance library.

        start_date: dt.datetime
            Start period to fetch prices. Defaults to None.

        end_date: dt.datetime
            End period to fetch prices. Defaults to None.

        mode: str
            Mode of price preparation: either differencing or pct. changes.

        window_size: int
            Size of the window to be used for daiily price change tracking

        initial_resources: float
            Initial amount of cash to start with.

        stochastic: bool
            Should prices at each day be selected randomly between Open & Close.

        lagged_features: bool
            Should env build a lagged features for LSTM or other RNN type, or just static ts features (like moving
            average, moving std, etc.)?
        """
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid mode in config. Allowed vals: {ALLOWED_MODES}, got: {mode}")
        self.ticker = ticker
        self.lagged_features = lagged_features
        self.stochastic = stochastic
        self._prepare_data(mode, start_date, end_date, window_size)
        self._calculate_features(window_size)
        self.dates = self.features.index
        self.position = 0
        self.start_date = self.prices.index.min()
        self.cash = initial_resources
        self.opening_account_balance = initial_resources
        self.n_stocks = 0
        self.action_space = gym.spaces.Discrete(len(ALLOWED_ACTIONS))

        min_values = np.array(self.features.min().to_list() + [0, 0])
        max_values = np.array(self.features.max().to_list() + [np.inf, np.inf])
        self.observation_space = gym.spaces.Box(min_values, max_values, shape=(self.features.shape[1] + 2,))

    def _prepare_data(self, mode: str, start_date: dt.datetime, end_date: dt.datetime, window_size: int):
        """
        Performs data preparation tasks: downloads financial data, reformats it and makes calculates features.

        Parameters
        ----------
        mode: str
            Mode of price calculations: either a classif diff (y(t+1) - y(t)) or pct change ((y(t+1)-y(t))/(y(t))
        start_date: dt.datetime
            Start date to fetch prices.
        end_date: dt.datetime
            End date to fetch prices
        window_size: int
            Window size for time windows calculation.
        """
        self.ohlc_data = yf.download(self.ticker, start=start_date, end=end_date)
        self.ohlc_data[PRICE_DIFF] = self.ohlc_data[ADJ_CLOSE].diff() if mode == MODE_DIFF else self.ohlc_data[
            ADJ_CLOSE].pct_change()
        self.ohlc_data[PRICE_DIFF_ROLLING_MU] = self.ohlc_data[PRICE_DIFF].rolling(window=window_size).mean()
        self.ohlc_data[PRICE_DIFF_ROLLING_STD] = self.ohlc_data[PRICE_DIFF].rolling(window=window_size).std()
        self.ohlc_data[LOW_DIFF_ROLLING_MU] = self.ohlc_data[LOW].diff().rolling(window=window_size).mean()
        self.ohlc_data[LOW_DIFF_ROLLING_STD] = self.ohlc_data[LOW].diff().rolling(window=window_size).std()
        self.ohlc_data[HIGH_DIFF_ROLLING_MU] = self.ohlc_data[LOW].diff().rolling(window=window_size).mean()
        self.ohlc_data[HIGH_DIFF_ROLLING_STD] = self.ohlc_data[HIGH].diff().rolling(window=window_size).std()
        self.ohlc_data.dropna(inplace=True)

    def _calculate_features(self, window_size: int):
        """
        Depending on the window size, this method generates either features in LSTM format array: (N x t x features)
        or static features, based on rolling time window calculation (e.g. rolling mean, rolling std, etc.)

        Parameters
        ----------
        window_size: int
            Size of the window for lagged features calculation.
        """
        if self.lagged_features:
            self.features = generate_lagged_data(self.ohlc_data[PRICE_DIFF], window_size).dropna()
            self.prices = generate_lagged_data(self.ohlc_data[ADJ_CLOSE], window_size).dropna()
            self.prices_h = generate_lagged_data(self.ohlc_data[HIGH], window_size).dropna()
            self.prices_l = generate_lagged_data(self.ohlc_data[LOW], window_size).dropna()
        else:
            self.features = self.ohlc_data[
                [ADJ_CLOSE, HIGH, LOW, PRICE_DIFF_ROLLING_MU, PRICE_DIFF_ROLLING_STD, LOW_DIFF_ROLLING_MU,
                 LOW_DIFF_ROLLING_STD, HIGH_DIFF_ROLLING_MU,
                 HIGH_DIFF_ROLLING_STD]]
        self.prices = self.ohlc_data[ADJ_CLOSE]
        self.prices_h = self.ohlc_data[LOW]
        self.prices_l = self.ohlc_data[HIGH]

    @property
    def current_date(self):
        return self.dates[self.position]

    def reset(self):
        self.position = 0
        self.date = self.prices.index.min()
        self.cash = self.opening_account_balance
        self.n_stocks = 0
        feature_vec = self.features.loc[self.current_date, :].to_list()
        return np.array(feature_vec + [self.cash, self.n_stocks])

    def step(self, action: int):
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Action not allowed. Possible options: {ALLOWED_ACTIONS}, got: {action}")

        if self.stochastic:
            current_price = np.random.uniform(self.prices_l[self.current_date], self.prices_h[self.current_date])
        else:
            current_price = self.prices.loc[self.current_date]
        self.position += 1
        revenue = 0.

        if action == ACTION_SELL:
            revenue += (self.n_stocks * current_price)
            self.n_stocks = 0.
        elif action == ACTION_BUY:
            nbuy = self.cash // current_price
            revenue = -1 * nbuy * current_price
            self.n_stocks += nbuy
        self.cash = np.max([self.cash + revenue, 0])

        if self.position >= (self.features.shape[0] - 1):
            s = np.zeros(self.observation_space.shape).tolist()
        else:
            s = self.features.loc[self.current_date].to_list()
            s += [self.cash, self.n_stocks]
        done = self.cash <= 0 or self.position >= (self.features.shape[0] - 1)

        total_value = self.cash + (self.n_stocks * current_price)
        profit_or_loss = total_value - self.opening_account_balance
        return s, profit_or_loss, done, {}


def check_agent_trading_env(agent: RLAgent, env: SingleStockEnv, n_epsiodes: int = 100, *args, **kwargs):
    """
    Checks agent on a traing environment. It differs from normal agent checking, where the reward is accumulated.
    In this case, reward == profit & loss, so only last reward matters.

    Parameters
    ----------
    agent: RLAgent
        Any agent, capable of handling this environment.

    env: SingleStockEnv
        Stock trading env.

    n_epsiodes: int
        Number of evaluation iterations

    Returns
    -------

    """
    revs = []
    for _ in tqdm(range(n_epsiodes)):
        s = env.reset()
        done = False
        while not done:
            action, _ = agent.choose_action(s, *args, **kwargs)
            sprime, r, done, _ = env.step(action)
            s = sprime
        revs.append(r)
    return revs
