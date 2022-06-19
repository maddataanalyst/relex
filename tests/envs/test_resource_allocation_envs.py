import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy.stats as st

import src.algorithms.algo_utils as autil
import src.envs.resource_allocation_env as rae


def test_simple_projects_env_correct_initialization():
    # given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 2
    size = 1000
    payout_min = -1
    payout_max = 1
    payout_mean = 0.0
    payout_std = 0.1
    env = rae.SimpleProjectsEnv(start_res, start_cash, upkeep_cost, max_payout=max_payout, size=size,
                                min_payout=payout_min,
                                payout_mean=payout_mean, payout_std=payout_std)

    # when
    assert (env.projects.proj_1_alloc < start_res).all()
    assert (env.projects.proj_2_alloc < start_res).all()
    assert (env.projects.proj_1_proba.all() == 1)
    assert (env.projects.proj_1_payouts.min() >= payout_min)
    assert (env.projects.proj_2_payouts.min() <= payout_max)

    payout_95_ci_low,payout_95_ci_high = st.norm.interval(0.95, loc=payout_mean, scale=payout_std)
    for proj in [env.projects.proj_1_payouts, env.projects.proj_2_payouts]:
        perc_within_bounds = ((payout_95_ci_low <= proj) & (proj <= payout_95_ci_high)).sum() / float(proj.shape[0])
        np.testing.assert_almost_equal(perc_within_bounds, 0.95, 2)


def test_simple_projects_env_one_step_allocation():
    # given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [1., 1., 1.],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [1., 1., 1.]
    })

    env = rae.SimpleProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                balance_is_reward=False)

    expected_obs = np.array([10, 1, 1., 5, 3, 1, 100, 397]).astype(int)

    # when
    step_1_alloc = np.array([0.9, 0.1, 0.0]) * 2 - 1.
    obs, reward, done, _ = env.step(step_1_alloc)

    # then
    assert reward == 297
    assert env.current_resources == 100
    assert np.all(obs == expected_obs)


def test_simple_projects_env_one_step_allocation_overbooking():
    # given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [1., 1., 1.],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [1., 1., 1.]
    })

    env = rae.SimpleProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                balance_is_reward=False)

    expected_obs = np.array([10, 1, 1., 5, 3, 1., 100, 364]).astype(int)

    # when
    step_1_alloc = np.array([0.99, 0.01, 0.0]) * 2 - 1.
    obs, reward, done, _ = env.step(step_1_alloc)

    # then
    assert reward == 264
    assert env.current_resources == 100
    assert np.all(obs == expected_obs)


def test_simple_projects_env_one_step_allocation_resource_dismiss():
    # given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [1., 1., 1.],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [1., 1., 1.]
    })

    env = rae.SimpleProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                balance_is_reward=False)

    expected_obs = np.array([10, 1, 1., 5, 3, 1., 91, 370]).astype(int)

    # when
    step_1_alloc = np.array([0.9, 0., 0.1]) * 2 - 1.
    obs, reward, done, _ = env.step(step_1_alloc)

    # then
    assert reward == 270
    assert env.current_resources == 91
    assert np.all(obs == expected_obs)


def test_simple_projects_env_multi_step_allocation():
    # given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [1., 1., 1.],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [1., 1., 1.]
    })

    env = rae.SimpleProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                balance_is_reward=False)

    expected_obs = [
        np.array([10, 1, 1., 5, 3, 1., 100, 397]).astype(int),
        np.array([10, 1, 1., 5, 3, 1., 100, 338]).astype(int),
        np.array([10, 1, 1., 5, 3, 1., 100, 280]).astype(int)
    ]

    # when
    step_alloc = np.array([0.9, 0.1, 0.0]) * 2 - 1.
    observations = []

    done = False
    steps = 0

    while not done:
        obs, reward, done, _ = env.step(step_alloc)
        observations.append(obs)
        steps += 1

    assert steps == 2
    for i in range(steps):
        assert np.all(observations[i] == expected_obs[i])
    assert done


def test_simple_projects_env_multi_step_allocation_stochastic():
    # given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.SimpleProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                balance_is_reward=False, stochastic=True)

    expected_obs = [
        np.array([10, 1, .5, 5, 3, .5, 100, 370]),
        np.array([10, 1, .5, 5, 3, .5, 100, 301])
    ]
    expected_rewards = [270., -69.]

    # when
    step_alloc = np.array([0.9, 0.1, 0.0]) * 2 - 1.
    observations = []

    done = False
    steps = 0
    rewards = []

    while not done:
        obs, reward, done, _ = env.step(step_alloc)
        observations.append(obs)
        steps += 1
        rewards.append(reward)

    assert steps == 2
    assert (np.array(rewards) == np.array(expected_rewards)).all()
    for i in range(steps):
        assert np.all(observations[i] == expected_obs[i])
    assert done


def test_discrete_env_action_proj_1():
    # Given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                  balance_is_reward=False, stochastic=False)

    expected_reward = 260
    expected_balance = 360
    expected_state = np.array([10, 1., .5, 5, 3, .5, 100, 360])

    # When
    obs, reward, done, _ = env.step(np.array(env.A_PROJ_1))

    # Then
    assert reward == expected_reward
    assert env.current_balance == expected_balance
    assert np.all(expected_state == obs)


def test_discrete_env_action_proj_2():
    # Given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                  balance_is_reward=False, stochastic=False)

    expected_reward = 220
    expected_balance = 320
    expected_state = np.array([10, 1., .5, 5, 3, .5, 100, 320])

    # When
    obs, reward, done, _ = env.step(np.array(env.A_PROJ_2))

    # Then
    assert reward == expected_reward
    assert env.current_balance == expected_balance
    assert np.all(expected_state == obs)


def test_discrete_env_action_proj_both():
    # Given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                  balance_is_reward=False, stochastic=False)

    expected_reward = 300
    expected_balance = 400
    expected_state = np.array([10, 1., .5, 5, 3, .5, 100, 400])

    # When
    obs, reward, done, _ = env.step(np.array(env.A_PROJ_BOTH))

    # Then
    assert reward == expected_reward
    assert env.current_balance == expected_balance
    assert np.all(expected_state == obs)


def test_discrete_env_action_wait():
    # Given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                  balance_is_reward=False, stochastic=False)

    expected_reward = -100
    expected_balance = 0
    expected_state = np.array([10, 1., .5, 5, 3, .5, 100, 0])

    # When
    obs, reward, done, _ = env.step(np.array(env.A_WAIT))

    # Then
    assert done
    assert reward == expected_reward
    assert env.current_balance == expected_balance
    assert np.all(expected_state == obs)


def test_discrete_env_action_reduce_25():
    # Given
    start_res = 100
    start_cash = 100
    upkeep_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                  balance_is_reward=False, stochastic=False)

    expected_reward = -75
    expected_balance = 25
    expected_state = np.array([10, 1., .5, 5, 3, .5, 75, 25])

    # When
    obs, reward, done, _ = env.step(np.array(env.A_REDUCE_RES_25PERC))

    # Then
    assert reward == expected_reward
    assert env.current_balance == expected_balance
    assert np.all(expected_state == obs)


def test_discrete_env_action_increase_25():
    # Given
    start_res = 100
    start_cash = 200
    upkeep_cost = -1
    hire_cost = -1
    max_payout = 3
    size = 1000
    projects = pd.DataFrame({
        "proj_1_alloc": [90, 10, 10],
        "proj_1_payouts": [3, 1, 1],
        "proj_1_proba": [0.999, .5, .5],
        "proj_2_alloc": [80, 5, 5],
        "proj_2_payouts": [3, 3, 3],
        "proj_2_proba": [.0001, .5, .5]
    })
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(start_res, start_cash, upkeep_cost, max_payout, size=size, projects=projects,
                                  balance_is_reward=False, stochastic=False, increase_resource_cost=hire_cost)

    expected_reward = -125
    expected_balance = 75
    expected_state = np.array([10, 1., .5, 5, 3, .5, 125, 75])

    # When
    obs, reward, done, _ = env.step(np.array(env.A_INCREASE_RES_25PERC))

    # Then
    assert reward == expected_reward
    assert env.current_balance == expected_balance
    assert np.all(expected_state == obs)


def test_challenging_environment():

    # Given
    np.random.seed(123)
    env = rae.DiscreteProjectsEnv(
        start_resource=100,
        start_cash=100,
        upkeep_cost=-1,
        min_payout=-0.5,
        max_payout=1.5,
        payout_mean=1.,
        payout_std=1.5,
        size=300,
        balance_is_reward=False,
        stochastic=True)

    agent = rae.DiscreteProjectOptimizerAgent(env)

    # When
    results = autil.evaluate_algorithm(agent, env, 100, max_ep_steps=300, clip_action=False)

    # Then
    assert (np.mean(results) <= 0)
