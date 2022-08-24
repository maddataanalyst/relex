import os
import gym
import tensorflow.keras as krs
import numpy as np
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler

import src.algorithms.advantages as adv
import src.algorithms.dummy as dummy
import src.algorithms.value_based.ddqn as dqn
import src.consts as consts
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.q_nets as qnets
import src.algorithms.memory_samplers as mbuff
import src.utils as utils
import src.envs.resource_allocation_env as rae

EXPERIMENT_NAME = "DDQN projects env"

ENV_SIZE = 300

if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'ddqn'
    env = rae.DiscreteProjectsEnv(
        start_resource=100,
        start_cash=100,
        upkeep_cost=-1,
        min_payout=-0.5,
        max_payout=1.5,
        payout_mean=1.0,
        payout_std=0.5,
        size=ENV_SIZE,
        pnl_is_reward=False,
        stochastic=True)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    hidden_initializer = krs.initializers.HeNormal()

    lr = 0.005
    optimizer = krs.optimizers.Adam(lr)

    hidden_act = 'relu'
    out_act = 'relu'
    qnet_arch = [64, 64]
    batch_norm = True
    action_net = qnets.QNet(s_dim, a_dim, qnet_arch, h_initializer=hidden_initializer, h_act=hidden_act, out_act=out_act, batch_norm=batch_norm)
    target_net = qnets.QNet(s_dim, a_dim, qnet_arch, h_initializer=hidden_initializer, h_act=hidden_act, out_act=out_act, batch_norm=batch_norm)
    buffer = mbuff.SimpleMemory(10000)
    dummy_agent = dummy.DummyAgent(env)
    optimization_agent = rae.DiscreteProjectOptimizerAgent(env)
    agent = dqn.DDQN(
        target_net,
        action_net,
        buffer,
        0.99,
        0.05,
        0.99,
        net_optimizer=optimizer,
        polyak_tau=0.8,
        target_update_frequency=10, loss_function='huber')

    agent_hyperparams = {
        consts.HIDDEN_ACT_F: hidden_act,
        consts.W_INITIALIZER: hidden_initializer,
        consts.OUTOUT_ACT_F: out_act,
        consts.Q_NET_ARCH: qnet_arch,
        consts.LEARNING_RATE: lr,
        consts.EP_MAX_STEPS: ENV_SIZE,
        consts.BATCH_NORM: batch_norm
    }
    agent_learning_params = {
        consts.NEPISODES: 200,
        consts.PRINT_INTERVAL: 1,
        consts.BATCH_SIZE: 32,
        'warmup_batches': 5
    }

    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent, optimization_agent], env, EXPERIMENT_NAME,
                                                                     'DDQN basic', agent_learning_params,
                                                                     agent_hyperparams, max_ep_steps=ENV_SIZE)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
