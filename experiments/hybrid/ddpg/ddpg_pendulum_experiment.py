import gym
import tensorflow.keras as krs

import src.algorithms.dummy as dummy
import src.algorithms.memory_samplers as mbuff
import src.algorithms.hybrid.tf2.ddpg as ddpg
import src.consts as consts
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.policy_nets as pinets
import src.models.base_models.tf2.q_nets as q_nets
import src.utils as utils

from src.envs.cartpole_continuous import ContinuousCartPoleEnv

EXPERIMENT_NAME = "DDPG pendulum"

if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'ddpg'
    env = gym.make("Pendulum-v1")

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    h = env.action_space.high
    l = env.action_space.low

    policy_act_f = 'relu'
    out_act_f = 'tanh'
    policy_arch = [256, 256]
    h_initializer = krs.initializers.HeNormal()
    out_initializer = krs.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    actor_net = pinets.DeterministicPolicyNet(a_dim, h_sizes=policy_arch, h_act=policy_act_f, out_act=out_act_f,
                                              out_initializer=out_initializer, hidden_initializer=h_initializer,
                                              a_max=h)
    actor_net_target = pinets.DeterministicPolicyNet(a_dim=a_dim, h_sizes=policy_arch, h_act=policy_act_f,
                                                     out_act=out_act_f,
                                                     out_initializer=out_initializer, hidden_initializer=h_initializer,
                                                     a_max=h)

    critic_arch_shared = [256, 256]
    critic_arch_a_sizes = [64]
    critic_arch_s_sizes = [64, 32]
    critic_h_act = 'relu'
    critic_out_act = 'linear'
    critic = q_nets.QSANet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes,
                           h_act=critic_h_act, out_act=critic_out_act)
    critic_target = q_nets.QSANet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes,
                                  h_act=critic_h_act, out_act=critic_out_act)

    lr_policy = 0.0001
    lr_critic = 0.001
    actor_opt = krs.optimizers.Adam(lr_policy)
    critic_opt = krs.optimizers.Adam(lr_critic)

    buffer = mbuff.SimpleMemory(25000)

    agent = ddpg.DDPG(actor_net, actor_net_target, critic, critic_target, buffer, l, h, actor_optimizer=actor_opt,
                      critic_optimizer=critic_opt, target_update_frequency=1,
                      polyak_tau=0.005)
    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"actor{consts.W_INITIALIZER}": h_initializer,
        consts.POLICY_ARCH: policy_arch,
        consts.LEARNING_RATE_POLICY: lr_policy,
        consts.LEARNING_RATE_VALUE: lr_critic,
        consts.CRITIC_ARCH: critic_arch_shared,
        consts.CRITIC_A_SZ: critic_arch_a_sizes,
        consts.CRITIC_S_SZ: critic_arch_s_sizes,
        'target_update_freq': 1
    }
    agent_learning_params = {
        consts.NEPISODES: 150,
        consts.PRINT_INTERVAL: 5,
        consts.GAMMA: 0.99,
        consts.BATCH_SIZE: 64,
        consts.WARMUP_BATCHES: 5,
        consts.EP_MAX_STEPS: 250

    }

    dummy_agent = dummy.DummyAgent(env)
    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'DDPG', agent_learning_params,
                                                                     agent_hyperparams, clip_actions=True)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
