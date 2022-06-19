import pytest
import gym
import tensorflow as tf
import tensorflow.keras as krs
import numpy as np

import tests.testing_utils as tu
import src.algorithms.policy_gradient.tf2.ppo as ppo
import src.models.base_models.tf2.policy_nets as pi_nets
import src.models.base_models.tf2.value_nets as vnets
import src.algorithms.advantages as adv
import src.algorithms.algo_utils as autil

DIM_STATE = 3
DIM_A = 2
N = 10


def test_ppo_optimize_policy_2d_gae_logprob_error():
    # Given
    optimizer = krs.optimizers.Adam()
    a = np.random.random((N, DIM_A))
    s = np.random.random((N, DIM_STATE))
    gae = np.random.random((N, 1))
    old_logprob = np.random.random((N, 1))

    policy_net = pi_nets.ContinuousPolicyNet(DIM_STATE, DIM_A, [32, 32])
    value_net = vnets.ValueNet(DIM_STATE, [16, 16])
    agent = ppo.PPO(policy_net, value_net, optimizer, optimizer, adv.GAE)

    # When
    with pytest.raises(ValueError) as e_info:
        agent.optimize_policy_ppo(gae, s, a, old_logprob)
    assert "Tensor  must have rank in (0, 1).  Received rank 2" in str(e_info.value.args[0])


def test_ppo_optimize_policy_1d_gae_logprob_allowed():
    # Given
    optimizer = krs.optimizers.Adam()
    a = np.random.random((N, DIM_A))
    s = np.random.random((N, DIM_STATE))
    gae = np.random.random((N,))
    old_logprob = np.random.random((N,))

    policy_net = pi_nets.ContinuousPolicyNet(DIM_STATE, DIM_A, [32, 32])
    value_net = vnets.ValueNet(DIM_STATE, [16, 16])
    agent = ppo.PPO(policy_net, value_net, optimizer, optimizer, adv.GAE)

    # When
    loss = agent.optimize_policy_ppo(gae, s, a, old_logprob)

    # Then
    assert loss.numpy().size == 1


def test_ppo_performance_discrete_cartpole():
    # Given
    np.random.seed(123)
    tf.random.set_seed(123)
    env = gym.make('CartPole-v1')

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    policy_net = pi_nets.DiscretePolicyNet(a_dim, [128, 128], 'relu')
    value_net = vnets.ValueNet(s_dim, [128, 128])

    policy_opt = krs.optimizers.Adam(0.0007)
    value_opt = krs.optimizers.Adam(0.005)
    log = tu.get_test_logger()

    agent = ppo.PPO(policy_net, value_net, policy_opt, value_opt, adv.GAE, n_agents=3, clipping_eps=0.05)

    # When
    scores = np.array(
        agent.train(env, 200, max_steps=200, clip_action=False, print_interval=10, batch_size=32, log=log))
    eval_scores = np.array(
        autil.evaluate_algorithm(agent, env, n_episodes=100, max_ep_steps=200, clip_action=False))
    # Then
    polyfit = np.polyfit(np.arange(scores.shape[0]), scores, 1)
    slope = polyfit[0]
    tu.summarize_scores(eval_scores, scores, log)
    assert slope > 1.5
    assert eval_scores.mean() > 150
