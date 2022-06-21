import gym
import tensorflow.keras as krs

import src.algorithms.dummy as dummy
import src.algorithms.memory_samplers as mbuff
import algorithms.hybrid.tf2.sac as sac
import src.consts as consts
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.policy_nets as pinets
import src.models.base_models.tf2.q_nets as q_nets
import src.utils as utils

EXPERIMENT_NAME = "SAC lunar cont"
#TODO: sac is very slow and seems unstable. Debug
if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'sac'
    env = gym.make("LunarLanderContinuous-v2")

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    h = env.action_space.high
    l = env.action_space.low

    policy_act_f = 'relu'
    mean_out_actf = 'linear'
    policy_arch = [128, 64]
    h_initializer = krs.initializers.GlorotUniform()
    out_initializer = krs.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    policy_net = pinets.ContinuousPolicyNetReparam(s_dim, a_dim, policy_arch, policy_act_f, mean_out_act=mean_out_actf,
                                                out_initializer=out_initializer, hidden_initializer=h_initializer)

    critic_arch_shared = [128]
    critic_arch_a_sizes = [128, 64]
    critic_arch_s_sizes = [128, 64]
    critic_h_act = 'relu'
    critic_out_act = 'linear'
    critic1 = q_nets.QNet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes, h_act=critic_h_act, out_act=critic_out_act)
    critic2 = q_nets.QNet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes, h_act=critic_h_act, out_act=critic_out_act)

    target_critic1 = q_nets.QNet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes, h_act=critic_h_act, out_act=critic_out_act)
    target_critic2 = q_nets.QNet(s_dim, a_dim, critic_arch_shared, critic_arch_a_sizes, critic_arch_s_sizes, h_act=critic_h_act, out_act=critic_out_act)

    lr_policy = 0.0001
    lr_value = 0.005
    policy_opt = krs.optimizers.Adam(lr_policy)
    critic_opt = krs.optimizers.Adam(lr_value)

    buffer = mbuff.SimpleMemory(25000)

    agent = sac.SAC(policy_net, critic1, critic2, target_critic1, target_critic2, policy_opt, critic_opt, buffer)
    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"policy_h{consts.W_INITIALIZER}": h_initializer,
        consts.POLICY_ARCH: policy_arch,
        consts.LEARNING_RATE_POLICY: lr_policy,
        consts.LEARNING_RATE_VALUE: lr_value,
        consts.CRITIC_ARCH: critic_arch_shared,
        consts.CRITIC_A_SZ: critic_arch_a_sizes,
        consts.CRITIC_S_SZ: critic_arch_s_sizes
    }
    agent_learning_params = {
        consts.NEPISODES: 300,
        consts.PRINT_INTERVAL: 1,
        consts.LAMBDA: 0.95,
        consts.GAMMA: 0.99,
        consts.BATCH_SIZE: 64,
        consts.WARMUP_BATCHES: 5

    }

    dummy_agent = dummy.DummyAgent(env)
    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'SAC', agent_learning_params,
                                                                     agent_hyperparams, clip_actions=True)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
