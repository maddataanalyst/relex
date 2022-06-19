import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

import src.algorithms.advantages as adv
import src.algorithms.algo_utils as autil
import src.algorithms.policy_gradient.tf2.vpg as vpg
import src.models.base_models.tf2.policy_nets as pi_nets
import src.models.base_models.tf2.value_nets as vnets
import tests.testing_utils as tu

DIM_STATE = 3
DIM_A = 2
N = 10


def test_vpg_performance_discrete_cartpole():
    # Given
    np.random.seed(123)
    tf.random.set_seed(123)
    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    hidden_initializer = krs.initializers.RandomUniform()
    policy_act_f = 'tanh'
    policy_arch = [128, 64]
    policy_net = pi_nets.DiscretePolicyNet(a_dim, policy_arch, h_act=policy_act_f, hidden_initializer=hidden_initializer)

    value_act_f = 'relu'
    value_out_act_f = 'relu'
    value_arch = [128, 64]
    value_net = vnets.ValueNet(s_dim, value_arch, out_act=value_out_act_f, h_act=value_act_f)

    lr = 0.0005
    policy_opt = krs.optimizers.Adam(lr)
    value_opt = krs.optimizers.Adam(lr)

    advantage = adv.GAE
    entropy = 0.001
    agent = vpg.VPG(policy_net, value_net, advantage=advantage, policy_opt=policy_opt, value_opt=value_opt, entropy_coef=entropy)

    log = tu.get_test_logger()

    scores = agent.train(env, 200, max_steps=200, clip_action=False, print_interval=10, log=log, lambda_=0.95, gamma=0.99)
    eval_scores = autil.evaluate_algorithm(agent, env, n_episodes=100, max_ep_steps=300, clip_action=False)

    # Then

    tu.summarize_scores(eval_scores, scores, log)

    assert np.array(eval_scores).mean() >= 140
