import datetime as dt
import os
from typing import List

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import pingouin as pg
import seaborn as sns
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO, MlpPolicy

import src.algorithms.dummy as dummy
import src.algorithms.sb3.sb3utils as sb3
import src.envs.stock_trading.classic_strategies as smac
import src.envs.stock_trading.single_stock_env as sse
import src.utils as utils

COLUMN_SCORE = 'score'

COLUMN_MODEL = 'model'

EXPERIMENT_NAME = 'ppo sb3 vs smac'
log = utils.prepare_default_log()


def ppo_sb3_vs_smac_run(ticker: str,
                        parallel_envs: int = 1,
                        net_arch: List = [dict(pi=[64, 64, 64], vf=[64, 64, 64])],
                        learning_rate: float = 0.005,
                        n_steps: int = 64,
                        batch_size: int = 32,
                        ent_coef: float = 0.001,
                        total_timesteps: int = 50000,
                        *args,
                        **kwargs):
    """
    Runs complex PPO SB3 vs SMAC vs dummy experiment on a single trading env.
    Parameters
    ----------
    ticker: str
        Ticker from yahoo finnance to run analysis on.
        net_arch: List
        Net architecture as required by Stable Baselines 3.

    parallel_envs: int
        Number of parallel envs to use with PPO.

    learning_rate: float
        Lr for optimizer.

    n_steps: int
        N-steps for TD target calculation.

    batch_size: int
        PPO batch size.

    ent_coef: float
        Entropy coefficient - equivalent of exploration-exploitation ratio.

    total_timesteps: int
        Total number of timesteps to use in training.
    """
    experiment = f"{EXPERIMENT_NAME} {ticker}".replace(".", " ")

    experiment_folder = os.path.join(os.getcwd(), experiment)
    path = Path(experiment_folder)
    path.mkdir(parents=True, exist_ok=True)

    experiment_id = utils.prepare_mlflow(experiment)
    mlflow.autolog()

    with mlflow.start_run(experiment_id=experiment_id):
        start_date = dt.datetime(2016, 1, 1)
        eval_date_from = dt.datetime(2020, 1, 1)

        initial_capital = 100000
        eval_episodes = 100

        # Envs prep
        env_train = sse.SingleStockEnv(ticker, start_date=start_date, end_date=eval_date_from, mode=sse.MODE_PCT_CHANGE,
                                       stochastic=True,
                                       initial_resources=initial_capital, window_size=120)
        env_eval = sse.SingleStockEnv(ticker, start_date=start_date, mode=sse.MODE_PCT_CHANGE, stochastic=True,
                                      initial_resources=initial_capital, window_size=120)

        # Market strategy eval
        smac_eval_pnls, smac_train_pnls = eval_strategy(env_eval, env_train, eval_episodes, initial_capital)

        # Eval RL algorithms
        random_rev_eval, random_rev_train = eval_dummy(env_eval, env_train)
        ppo_rev_after_train, ppo_rev_before, ppo_rev_eval = train_rl(env_eval, env_train, parallel_envs, net_arch,
                                                                     learning_rate,
                                                                     n_steps, batch_size, ent_coef, total_timesteps,
                                                                     *args, **kwargs)

        df_scores = pd.DataFrame({
            'ppo_eval': ppo_rev_eval,
            'ppo_train_before': ppo_rev_before,
            'ppo_train_after': ppo_rev_after_train,
            'random_eval': random_rev_eval,
            'random_train': random_rev_train,
            'strategy_eval': smac_eval_pnls,
            'strategy_train': smac_train_pnls
        })
        scores_train = _melt_scores(df_scores, "train")
        scores_eval = _melt_scores(df_scores, "eval")
        scores_file_path = os.path.join(experiment_folder, "scores.csv")

        plot_scores(experiment_folder, scores_train, "train_scores_boxplot")
        plot_scores(experiment_folder, scores_eval, "eval_scores_boxplot")
        compare_eval_scores(df_scores, experiment_folder, scores_file_path)


def _melt_scores(df_scores: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Melts scores from wide to long form. Returning long df with columns: model, score.
    Parameters
    ----------
    df_scores: pd.DataFrame
        Original scores data frame.

    prefix: str
        Type of scores to get (eval/test).

    Returns
    -------
    pd.DataFrame
        Melted df with scores.

    """
    return df_scores[[c for c in df_scores if prefix in c]].melt(var_name=COLUMN_MODEL, value_name=COLUMN_SCORE)


def compare_eval_scores(df_scores: pd.DataFrame, experiment_folder: str, scores_file_path: str):
    """
    Runs statistical comparisons between model scores. Saves comparison results to the specified folder.

    Parameters
    ----------
    df_scores: pd.DataFrame
        Data frame with scores for different models.
    experiment_folder: str
        Folder to store experiment results.
    scores_file_path: str
        File path to dataframe.
    """
    eval_scores = df_scores[[c for c in df_scores.columns if 'eval' in c]]
    eval_scores_melted = eval_scores.melt(var_name='model', value_name='score')
    kruskal_test_fpath = os.path.join(experiment_folder, 'kruskal.csv')
    pairwise_test_fpath = os.path.join(experiment_folder, 'pairwise.csv')
    kruskal_test = pg.kruskal(eval_scores_melted, dv='score', between='model')
    pariwise_test = pg.pairwise_ttests(eval_scores_melted, dv='score', between='model', parametric=False).round(4)
    dataframes = [
        (df_scores, scores_file_path),
        (kruskal_test, kruskal_test_fpath),
        (pariwise_test, pairwise_test_fpath)
    ]
    for df, pth in dataframes:
        df.to_csv(pth)
        mlflow.log_artifact(pth)


def plot_scores(experiment_folder: str, scores_melted: pd.DataFrame, boxplot_name: str):
    """
    Plots model scores and saves them to specified folder.

    Parameters
    ----------
    experiment_folder: str
        Folde to store experimental results.
    scores_melted: pd.DataFrame
        Melted data frame with experimental results (columns: model, score).
        Used to build boxplot.
    boxplot_name: str
        Name of the boxplot to save.
    """
    scores_boxplot_fpath = os.path.join(experiment_folder, f"{boxplot_name}.png")
    plt.figure(figsize=(10, 10))
    boxplot = sns.boxplot(data=scores_melted, x='model', y='score')
    boxplot.set(title="Scores boxplot")
    boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(scores_boxplot_fpath)
    plt.show()
    mlflow.log_artifact(scores_boxplot_fpath)


def eval_dummy(env_eval, env_train):
    dummy_agent = dummy.DummyAgent(env_train)
    random_rev_train = sse.check_agent_trading_env(dummy_agent, env_train)
    random_rev_eval = sse.check_agent_trading_env(dummy_agent, env_eval)
    return random_rev_eval, random_rev_train


def train_rl(env_eval: sse.SingleStockEnv, env_train: sse.SingleStockEnv,
             parallel_envs: int = 1,
             net_arch: List = [dict(pi=[64, 64, 64], vf=[64, 64, 64])],
             learning_rate: float = 0.005,
             n_steps: int = 64,
             batch_size: int = 32,
             ent_coef: float = 0.001,
             total_timesteps: int = 50000,
             * args,
             **kwargs):
    """
    Trains PPO on a given environments (train & eval).

    Parameters
    ----------
    env_eval: sse.SingleStockEnv
        Evaluation environment.

    env_train: sse.SingleStockEnv
        Training environment.

     parallel_envs: int
        Number of parallel envs to use with PPO.

    net_arch: List
        Net architecture as required by Stable Baselines 3.

    learning_rate: float
        Lr for optimizer.

    n_steps: int
        N-steps for TD target calculation.

    batch_size: int
        PPO batch size.

    ent_coef: float
        Entropy coefficient - equivalent of exploration-exploitation ratio.

    total_timesteps: int
        No. of timesteps in learning.

    Returns
    -------

    """
    # Prepare PPO
    mlflow.log_params({
        'lr': learning_rate,
        'net arch': net_arch,
        'entropy': ent_coef,
        'n steps': n_steps,
        'batch': batch_size,
        'total timesteps': total_timesteps
    })
    policy_kwargs = dict(net_arch=net_arch)
    envs = env_train if parallel_envs == 1 else make_vec_env(lambda: env_train, n_envs=5)
    agent = PPO(MlpPolicy, envs, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=learning_rate, n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef, *args, **kwargs)

    ppo_rev_before = sse.check_agent_trading_env(sb3.BaselinesWrapper(agent), env_train)
    agent.learn(total_timesteps=total_timesteps)
    ppo_rev_after_train = sse.check_agent_trading_env(sb3.BaselinesWrapper(agent), env_train)
    ppo_rev_eval = sse.check_agent_trading_env(sb3.BaselinesWrapper(agent), env_eval)
    return ppo_rev_after_train, ppo_rev_before, ppo_rev_eval


def eval_strategy(env_eval, env_train, eval_episodes, initial_capital):
    # Strategy prep
    strategy = smac.SMACStrategy()
    # Strategy check on train data
    _, smac_train_pnls = smac.eval_strategy(strategy, env_train.ohlc_data, initial_capital, eval_episodes)
    _, smac_eval_pnls = smac.eval_strategy(strategy, env_eval.ohlc_data, initial_capital, eval_episodes)
    return smac_eval_pnls, smac_train_pnls


if __name__ == "__main__":
    ppo_sb3_vs_smac_run("EUNL.DE")
