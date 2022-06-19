import logging
import os
import numpy as np
import gym
import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

from pathlib import Path
from typing import Dict, Callable, Tuple, List
from functools import partial
from pandas import DataFrame

from src.algorithms.commons import RLAgent
import src.consts as cnst
import src.algorithms.algo_utils as autil
import src.utils as utils


def _log_mlflow_params(params: Dict[str, object]):
    for name, param in params.items():
        mlflow.log_param(name, param)


def log_scores(agent_name: str, scores_name: str, agent_scores: np.array, logger: logging.Logger):
    mu = agent_scores.mean().round(3)
    std = agent_scores.std().round(3)
    median = np.median(agent_scores).round(3)
    msg = f"{agent_name} {scores_name}: mean={mu}, std={std}, median={median}"
    logger.info(msg)
    for name, score in [('mu', mu), ('std', std), ('median', median)]:
        mlflow.log_metric(f"{agent_name} {scores_name} {name}", score)


def conduct_experiment(
        agent: RLAgent,
        benchmark_agents: List[RLAgent],
        env: gym.wrappers.TimeLimit,
        experiment_name: str,
        run_name: str,
        agent_learning_params: Dict[str, object],
        agent_hyperparams_to_log: Dict[str, object],
        eval_episodes: int = 100,
        max_ep_steps: int = 200,
        clip_actions: bool = False,
        logger: logging.Logger = utils.prepare_default_log(),
        overall_comparison_test: Callable = pg.kruskal,
        pairwise_comparison_test: Callable = pg.pairwise_ttests,
        pairwise_parametric: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generic function for conducting RL experiments using Relex agents and gym or custom envs.
    Should fit with most agent/env combinations.

    Parameters
    ----------
    agent: RLAgent
        Main agent to be tested.
    benchmark_agents: List[RLAgent]
        List of benchmark agents to be compared with
    env: gym.wrappers.TimeLimit
        Env to be tested in
    experiment_name: str
        Name of the experiment to use in mlflow logging.
    run_name: str
        Specific name of the experiment attempt for logging.
    agent_learning_params: Dict[str, object]
        Parameters for agent training session
    agent_hyperparams_to_log: Dict[str, object]
        Agent hyperparameters to log in mlflow.
    eval_episodes: int
        Number of evaluation episodes
    max_ep_steps: int
        Maximum number of episode steps
    clip_actions: bool
        Should actions be clipped?
    logger: logging.Logger
        Logger for saving the outputs
    overall_comparison_test: Callable
        Ovearall statistical test to conduct. Defaults to pingouin Kruskal test.
    pairwise_comparison_test: Callable
        Pairwise test to condut. Default to pingouin pairwiste ttests.
    pairwise_parametric: bool
        Should pairwise test be parametric? Default to false.
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple consisting of:
        1. A Data frame with agents scores: before training, during training, eval.
        2. Overall statistical difference test results
        3. Pairwise test results
    """
    # TODO: make passing params easier - create common class 'model params': used both to initialize model and for
    #  experiments. One won't have to specify params twice
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.autolog()
    Path(os.path.join(os.curdir, experiment_name, run_name)).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        _log_mlflow_params(agent_hyperparams_to_log)

        agent_scores_before = autil.evaluate_algorithm(agent, env, n_episodes=eval_episodes, max_ep_steps=max_ep_steps,
                                                       clip_action=clip_actions)
        log_scores(agent.name, 'before train', agent_scores_before, logger)

        benchmark_scores_by_name = {}
        for benchmark_agent in benchmark_agents:
            benchmark_agent_score = autil.evaluate_algorithm(benchmark_agent, env, n_episodes=eval_episodes,
                                                             max_ep_steps=max_ep_steps,
                                                             clip_action=clip_actions)
            log_scores(benchmark_agent.name, 'benchmark', benchmark_agent_score, logger)
            benchmark_scores_by_name[benchmark_agent.name] = benchmark_agent_score

        training_scores = agent.train(env, max_steps=max_ep_steps, clip_action=clip_actions, log=logger,
                                      **agent_learning_params)
        log_scores("agent", "training scores", training_scores, logger)

        agent_eval = autil.evaluate_algorithm(agent, env, n_episodes=eval_episodes, max_ep_steps=max_ep_steps,
                                              clip_action=clip_actions)
        log_scores(agent.name, "eval", agent_eval, logger)

        scores_df = pd.DataFrame({
            f"{agent.name} before": agent_scores_before,
            f"{agent.name} eval": agent_eval,
        })
        for benchmark_name, benchmark_score in benchmark_scores_by_name.items():
            scores_df[benchmark_name] = benchmark_score

        save_scores(experiment_name, run_name, scores_df)
        build_boxplot(agent, benchmark_agents, experiment_name, run_name, scores_df)
        build_histograms(agent, agent_eval, agent_scores_before, benchmark_scores_by_name,
                         experiment_name, run_name)
        overall_test, pairwise_test = conduct_statistical_tests(scores_df, experiment_name, run_name,
                                                                overall_comparison_test, pairwise_comparison_test,
                                                                pairwise_parametric)
        return scores_df, overall_test, pairwise_test


def save_scores(experiment_name, run_name, scores_df):
    scores_path = os.path.join(os.curdir, experiment_name, run_name, cnst.SCORES_NAME + '.csv')
    scores_df.to_csv(scores_path)
    mlflow.log_artifact(scores_path)


def conduct_statistical_tests(
        scores: pd.DataFrame,
        experiment_name: str,
        run_name: str,
        overall_comparison_test: Callable = pg.kruskal,
        pairwise_comparison_test: Callable = pg.pairwise_ttests,
        pairwise_parametric: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conducts a statistical comparison test between agents. First performs an overall statistical difference test
    followed by pairwise testing.

    Parameters
    ----------
    scores
    experiment_name: str
        Name of the experiment
    run_name: str
        Name of the run
    overall_comparison_test: Callable
        Pingouin function for overall statistical difference test. Defaults to Kruskal test.
    pairwise_comparison_test: Callable
        Pingouin function for post-hoc tests. Defaults to Baum-Welch test
    pairwise_parametric: bool
        Should pairwise test be parametric? Defaults to False.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Test results - overall test and pairwise test
    """
    scores_melted = scores.melt(value_name=cnst.SCORE_COL, var_name=cnst.MODEL_COL)
    overall_test_result = overall_comparison_test(data=scores_melted, dv=cnst.SCORE_COL, between=cnst.MODEL_COL)
    pairwise_test_result = pairwise_comparison_test(data=scores_melted, dv=cnst.SCORE_COL, between=cnst.MODEL_COL,
                                                    parametric=False)

    overall_test_fname = os.path.join(os.curdir, experiment_name, run_name, cnst.OVERALL_RESULT_NAME + ".csv")
    pairwise_test_fname = os.path.join(os.curdir, experiment_name, run_name, cnst.PAIRWISE_RESULT_NAME + ".csv")
    overall_test_result.to_csv(overall_test_fname)
    pairwise_test_result.to_csv(pairwise_test_fname)

    mlflow.log_artifact(overall_test_fname)
    mlflow.log_artifact(pairwise_test_fname)

    return overall_test_result, pairwise_test_result


def build_histograms(agent: RLAgent, agent_eval: pd.Series, agent_scores_before: pd.Series, benchmark_scores_by_name: dict, experiment_name: str, run_name: str):
    plt.figure()
    palette = sns.color_palette("tab10")
    n_models = 2 + len(benchmark_scores_by_name)
    sns.histplot(agent_scores_before, color=palette[0], label=f"{agent.name} before")
    for idx, (benchmark_name, benchmark_score) in enumerate(benchmark_scores_by_name.items()):
        sns.histplot(benchmark_score, color=palette[idx + 1], label=benchmark_name)
    sns.histplot(agent_eval, color=palette[n_models], label=f"{agent.name} after")
    benchmark_names = ", ".join(list(benchmark_scores_by_name.keys()))
    plt.title(f"{agent.name} vs {benchmark_names}: {experiment_name}")
    plt.legend()
    plot_fname = os.path.join(os.curdir, experiment_name, run_name, cnst.HIST_NAME + ".png")
    plt.savefig(plot_fname)
    mlflow.log_artifact(plot_fname)
    plt.clf()


def build_boxplot(agent: RLAgent, benchmark_agents: List[RLAgent], experiment_name: str, run_name: str, scores_df: pd.DataFrame):
    plt.figure()
    bencharks_names = ", ".join([benchmark_agent.name for benchmark_agent in benchmark_agents])
    sns.boxplot(data=scores_df).set_title(f"{agent.name} vs {bencharks_names}: {experiment_name}")
    plot_fname = os.path.join(os.curdir, experiment_name, run_name, cnst.BOXPLOT_NAME + ".png")
    plt.savefig(plot_fname)
    mlflow.log_artifact(plot_fname)
    plt.clf()
