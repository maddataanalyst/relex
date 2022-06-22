import logging
import sys
import time
import mlflow
from typing import List

import gym
import numpy as np
import tabulate as tbl
from tqdm.auto import tqdm

import src.algorithms.commons as rl_commons

NANS_EPISLON = "NaNs in epsilons check params"

TOO_FEW_EPISODES_MSG = "Cannot allocate decayed espilons. Try to increase number of episodes or increase decay rate"


def normalize_state(s: np.ndarray, scaler: object = None) -> np.array:
    """
    Normalize state if needed (if scaler is present). If not - return state as-is

    Parameters
    ----------
    s: np.narray
        States matrix/array

    scaler: object
        Scaler object. Defaults to None.

    Returns
    -------
    np.array
        Scaled state (DIM_STATE,)
    """
    if scaler:
        return scaler.transform(np.array([s])).squeeze()
    else:
        return s


def log_progress(
        max_train_sec: int,
        average_n_last: int,
        all_rewards: np.array,
        t0: float,
        losses: List[float],
        ep: int,
        max_ep: int,
        log: logging.Logger,
        log_to_mlflow: bool = True):
    """
    Prints pretty formatted log in a readable form

    Parameters
    ----------
    max_train_sec: int
        Maximum train time.

    average_n_last: int
        Number of recent records to average.

    all_rewards: np.array
        All episode rewards

    t0: float
        Start time

    losses: List[float]
        Recorded losses

    ep: int
        Episode number

    max_ep: int
        Max episodes

    log: logging.Logger
        Log to print to

    log_to_mlflow: bool
        Log to MLFlow?
    """
    recent_rewards_avg = np.mean(all_rewards[-average_n_last:])
    recent_loss_avg = np.round(np.mean(losses[-average_n_last:]), decimals=3)
    time_delta = np.round((time.time() - t0) / 60.0, 3)
    txt_to_write = [
        f"Ep: {ep}/{max_ep}",
        f"Rewards avg.: {recent_rewards_avg:.3f}",
        f"Loss avg.: {recent_loss_avg:.3f}",
        f"Time passed: {time_delta:.3f}/{max_train_sec / 60.:.3f}"
    ]
    if log_to_mlflow:
        mlflow.log_metrics({'rewards_avg': recent_rewards_avg, 'loss_avg': recent_loss_avg}, step=ep)
    log.info(tbl.tabulate([txt_to_write], tablefmt="plain"))


def evaluate_algorithm(model: rl_commons.RLAgent, env: gym.Env, n_episodes: int, max_ep_steps: int = 1000,
                       clip_action: bool = True, scaler: object = None) -> np.array:
    """
    Evaluates an agent against an environment and collects scores along the way.

    Parameters
    ----------
    model: rl_commons.RLAgent
        An agent to be evaluated.

    env: gym.Env
        Env to evaluate an agent in.

    n_episodes: int
        Number of episodes to evaluate agent.

    max_ep_steps: int
        Maximal number of steps for agent evaluation.

    clip_action: bool
        Should actions be clipped against the env min/max.

    scaler: object
        A scaler object to be used.

    Returns
    -------
    np.array
        List of episode scores.
    """
    scores = []
    for _ in tqdm(range(n_episodes)):
        done = False
        step = 0
        ep_score = 0.
        s = env.reset()
        while not done:
            if step > max_ep_steps:
                break
            s = np.atleast_1d(normalize_state(s, scaler))
            a, a_logprob = model.choose_action(s)
            a_clip = np.clip(a, env.action_space.low, env.action_space.high) if clip_action else a
            sprime, r, done, _ = env.step(np.atleast_1d(a_clip)) if env.action_space.shape != () else env.step(a_clip)
            ep_score += r
            s = sprime
        scores.append(ep_score)
    return np.array(scores)


def decay_schedule_epsilon(initial_value: float, min_value: float, decay_ratio: float, max_steps: int, log_start: int=-2, log_base: int = 10) -> np.array:
    """
    Builds a decayed epsilon values, according to some decay factor.
    Parameters
    ----------
    initial_value: float
        Initial epsilon value
    min_value: float
        Minimal epsilon value
    decay_ratio: float
        Decay rate
    max_steps: int
        Maximum number of steps for decay. Should be equal to THE NUMBER OF EPIDES.
    log_start: float
        Initial value for the logarithm.
    log_base: float
        Logarithm base

    Returns
    -------
    np.array
        An array of decayed epsilon.
    """
    n_steps = int(max_steps * decay_ratio)
    if n_steps == 0:
        raise ValueError(TOO_FEW_EPISODES_MSG)
    rem_steps = max_steps - n_steps
    values = np.logspace(log_start, 0, n_steps,
                         base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (initial_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    if np.isnan(values).any():
        raise ValueError(NANS_EPISLON)
    return values
