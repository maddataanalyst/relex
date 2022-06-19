from typing import Tuple
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import yfinance as yf

TOTAL = 'total'
CASH = 'cash'
POSITION_VALUE = 'position_value'
VALUE = 'value'
POSITION = 'position'
LONG_MA = 'long_ma'
SHORT_MA = 'short_ma'
SIGNAL = 'signal'
PRICE = 'price'


class ClassicStrategy:

    def __init__(self, name: str):
        self.name = name

    def implement_strategy(self, initial_capital: float, prices: pd.Series) -> Tuple[float, float]:
        raise NotImplementedError()


class SMACStrategy(ClassicStrategy):
    """Basic momentum-based strategy, that calculates a long-term and short term moving average of
    prices. It sends a buy/sell/hold signal, if short term average crosses the long-term trend.
    Code based and inspired by implementation in this tutorial:
    [1] https://www.freecodecamp.org/news/algorithmic-trading-in-python/
    [2] https://www.datacamp.com/tutorial/finance-python-trading#improving-the-trading-strategy
    """

    def __init__(self, long_window: int = 120, short_window: int = 60, name: str = "SMAC"):
        super().__init__(name)
        self.long_window = long_window
        self.short_window = short_window

    def implement_strategy(self, initial_capital: float, prices: pd.Series) -> Tuple[float, float]:
        """
        Performs a strategy check on a series of prices.

        Parameters
        ----------
        initial_capital: float
            Initial amount of available money.

        prices: pd.Series
            A series of prices (can be randomly drawn from high-low range.

        Returns
        -------
        Tuple[float, float]
            Final total portfolio value, and profit&loss after final period
        """
        buy_sell_signals = pd.DataFrame({PRICE: prices})
        buy_sell_signals[SIGNAL] = 0.0
        buy_sell_signals[SHORT_MA] = buy_sell_signals[PRICE].rolling(window=self.short_window, min_periods=1,
                                                                     center=False).mean()
        buy_sell_signals[LONG_MA] = buy_sell_signals[PRICE].rolling(window=self.long_window, min_periods=1,
                                                                    center=False).mean()
        # Buy when short MA > long MA
        buy_sell_signals[SIGNAL][self.short_window:] = np.where(
            buy_sell_signals[SHORT_MA][self.short_window:] > buy_sell_signals[LONG_MA][self.short_window:], 1., 0.)

        # Position == 1: short-term moved above short-term (signal changed from 0 to 1), therefore action is "buy"
        # Position == -1: short-term moved below long-term (signal changed from 1 to 0), therefore action is "sell"
        # Position == 0 is no change
        buy_sell_signals[POSITION] = buy_sell_signals[SIGNAL].diff()

        portfolio = pd.DataFrame({
            POSITION: np.zeros_like(buy_sell_signals[POSITION]),
            CASH: np.zeros_like(buy_sell_signals[POSITION])
        }, index=buy_sell_signals.index)
        prev_cash = initial_capital
        prev_position = 0
        for date, row in portfolio.iterrows():
            price = buy_sell_signals.loc[date, PRICE]
            signal_pos = buy_sell_signals.loc[date, POSITION]
            if signal_pos == 1:
                amnt_buy = int(prev_cash // price)
                buy_cost = amnt_buy * price
                new_cash = prev_cash - buy_cost
                new_position = prev_position + amnt_buy
                prev_cash = new_cash
                prev_position = new_position
                portfolio.at[date, CASH] = new_cash
                portfolio.at[date, POSITION] = new_position
            elif signal_pos == -1:
                sell_cash = price * prev_position
                new_cash = prev_cash + sell_cash
                portfolio.at[date, POSITION] = 0
                portfolio.at[date, CASH] = new_cash
                prev_cash = new_cash
                prev_position = 0
            else:
                portfolio.at[date, POSITION] = prev_position
                portfolio.at[date, CASH] = prev_cash

        portfolio[VALUE] = portfolio[POSITION] * buy_sell_signals[PRICE]
        portfolio[TOTAL] = portfolio[VALUE] + portfolio[CASH]
        return portfolio.iloc[-1][TOTAL], portfolio.iloc[-1][TOTAL] - initial_capital


def eval_strategy(strategy: ClassicStrategy, ohlc_data: pd.DataFrame, initial_capital: float, n_episodes: int) -> Tuple[np.array, np.array]:
    """
    Tests strategy against stochastic prices (drawn from between high and low) for a
    predefined number of episodes.

    Parameters
    ----------
    strategy: ClassicStrategy
        Selected classic trading strategy.

    ohlc_data: pd.DataFrame
        Data frame with OHLC data.

    initial_capital: float
        Initial investment capital

    n_episodes: int
        Number of evaluation episodes.

    Returns
    -------
    Tuple[np.array, np.array]
        Tuple with two arrays:
         1. portfolio total values per each episode
         2. pnls at the end of period per each episode.
    """
    total_values = []
    pnls = []
    for ep in tqdm(range(n_episodes)):
        random_prices = ohlc_data.apply(lambda row: np.random.uniform(row['Low'], row['High']), axis=1)
        total_value, pnl = strategy.implement_strategy(initial_capital, random_prices)
        total_values.append(total_value)
        pnls.append(pnl)
    return np.array(total_values), np.array(pnls)

