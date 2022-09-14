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

EXPERIMENT_NAME = "DDPG continuous cartpole"


if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'ddpg'
    env = ContinuousCartPoleEnv()

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    h = env.action_space.high
    l = env.action_space.low

    policy_act_f = 'relu'
    out_act_f = None
    policy_arch = [128, 128]
    h_initializer = krs.initializers.GlorotUniform()
    out_initializer = krs.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    actor_net = pinets.DeterministicPolicyNet(a_dim, h_sizes=policy_arch, h_act=policy_act_f, out_act=out_act_f,
                                              out_initializer=out_initializer, hidden_initializer=h_initializer)
    actor_net_target = pinets.DeterministicPolicyNet(a_dim=a_dim, h_sizes=policy_arch, h_act=policy_act_f, out_act=out_act_f,
                                                     out_initializer=out_initializer, hidden_initializer=h_initializer)

    critic_arch_shared = [128, 128]
    critic_arch_a_sizes = [64]
    critic_arch_s_sizes = [64]
    critic_h_act = 'relu'
    critic_out_act = None
    critic = q_nets.QSANet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes, h_act=critic_h_act, out_act=critic_out_act)
    critic_target = q_nets.QSANet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes, h_act=critic_h_act, out_act=critic_out_act)

    lr_policy = 0.001
    opt = krs.optimizers.Adam(lr_policy)

    buffer = mbuff.SimpleMemory(2500)

    agent = ddpg.DDPG(actor_net, actor_net_target, critic, critic_target, buffer, l, h, opt, target_update_frequency=1)
    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"actor{consts.W_INITIALIZER}": h_initializer,
        consts.POLICY_ARCH: policy_arch,
        consts.LEARNING_RATE_POLICY: lr_policy,
        consts.CRITIC_ARCH: critic_arch_shared,
        consts.CRITIC_A_SZ: critic_arch_a_sizes,
        consts.CRITIC_S_SZ: critic_arch_s_sizes,
        'target_update_freq': 1
    }
    agent_learning_params = {
        consts.NEPISODES: 1000,
        consts.PRINT_INTERVAL: 50,
        consts.LAMBDA: 0.95,
        consts.GAMMA: 0.99,
        consts.BATCH_SIZE: 256,
        consts.WARMUP_BATCHES: 5

    }

    dummy_agent = dummy.DummyAgent(env)
    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'DDPG', agent_learning_params,
                                                                     agent_hyperparams, clip_actions=True)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
