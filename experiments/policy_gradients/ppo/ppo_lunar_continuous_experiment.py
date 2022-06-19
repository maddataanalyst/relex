import gym
import tensorflow.keras as krs

import src.algorithms.advantages as adv
import src.algorithms.dummy as dummy
import src.algorithms.policy_gradient.tf2.ppo as ppo
import src.consts as consts
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.policy_nets as policy_nets
import src.models.base_models.tf2.value_nets as vnets
import src.utils as utils

EXPERIMENT_NAME = "PPO lunar cont"

if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'ppo_gae'
    env = gym.make('LunarLanderContinuous-v2')
    batch_size = 64

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    h = env.action_space.high
    l = env.action_space.low

    policy_act_f = 'relu'
    mean_out_actf = 'tanh'
    policy_arch = [64, 64, 64]
    h_initializer = krs.initializers.HeNormal()
    out_initializer = krs.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    policy_net = policy_nets.ContinuousPolicyNet(
        s_dim,
        a_dim,
        policy_arch,
        policy_act_f,
        mean_out_act=mean_out_actf,
        out_initializer=out_initializer,
        hidden_initializer=h_initializer)

    value_act_f = 'relu'
    value_out_act_f = 'linear'
    value_arch = [64, 64, 64]
    value_net = vnets.ValueNet(s_dim, value_arch, h_act=value_act_f, out_act=value_out_act_f)

    lr_policy = 0.0007
    lr_value = 0.005
    policy_opt = krs.optimizers.Adam(lr_policy)
    value_opt = krs.optimizers.Adam(lr_value)

    advantage = adv.GAE
    entropy = 0.001
    clipping_eps = 0.1
    n_agents = 5

    agent = ppo.PPO(policy_net, value_net, policy_opt=policy_opt, value_opt=value_opt, advantage=advantage,
                          entropy_coef=entropy, clipping_eps=clipping_eps, n_agents=n_agents)
    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"policy_h{consts.W_INITIALIZER}": h_initializer,
        consts.VALUE_ACT_F: value_act_f,
        consts.VALUE_OUT_ACT_F: value_out_act_f,
        consts.POLICY_ARCH: policy_arch,
        consts.VALUE_ARCH: value_arch,
        consts.LEARNING_RATE_POLICY: lr_policy,
        consts.LEARNING_RATE_VALUE: lr_value,
        consts.ADVANTAGE_F: advantage.NAME,
        consts.ENTROPY_VAL: entropy,
        consts.CLIPPING_EPS: clipping_eps,
        consts.N_AGENTS: n_agents
    }
    agent_learning_params = {
        consts.NEPISODES: 1500,
        consts.PRINT_INTERVAL: 1,
        consts.LAMBDA: 0.95,
        consts.GAMMA: 0.99,
        consts.BATCH_SIZE: batch_size
    }

    dummy_agent = dummy.DummyAgent(env)
    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'PPO lunar cont', agent_learning_params,
                                                                     agent_hyperparams)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
