import tensorflow.keras as krs

import src.algorithms.advantages as adv
import src.algorithms.dummy as dummy
import src.algorithms.policy_gradient.tf2.vpg as vpg
import src.consts as consts
import src.envs.resource_allocation_env as rae
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.policy_nets as policy_nets
import src.models.base_models.tf2.value_nets as vnets
import src.utils as utils

EXPERIMENT_NAME = "VPG discrete projects optimization"
ENV_SIZE = 300

if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'vpg_gae'
    env = rae.DiscreteProjectsEnv(
        start_resource=100,
        start_cash=100,
        upkeep_cost=-1,
        min_payout=-0.5,
        max_payout=1.5,
        payout_mean=1.0,
        payout_std=0.5,
        size=ENV_SIZE,
        balance_is_reward=False,
        stochastic=True)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    hidden_initializer = krs.initializers.GlorotNormal(seed=789)
    policy_act_f = 'tanh'
    policy_arch = [128, 128]
    policy_net = policy_nets.DiscretePolicyNet(
        a_dim,
        policy_arch,
        h_act=policy_act_f,
        hidden_initializer=hidden_initializer,
        out_initializer=krs.initializers.GlorotNormal(seed=456))

    value_act_f = 'tanh'
    value_out_act_f = 'linear'
    value_arch = [64, 64]
    value_initializer = krs.initializers.GlorotNormal(seed=123)
    value_net = vnets.ValueNet(s_dim, value_arch, out_act=value_out_act_f, h_act=value_act_f,
                               initializer=value_initializer)

    lr_policy = 0.0005
    lr_value = 0.0003
    policy_opt = krs.optimizers.Adam(lr_policy)
    value_opt = krs.optimizers.Adam(lr_value)

    advantage = adv.GAE
    entropy = 0.001

    dummy_agent = dummy.DummyAgent(env)
    optimization_agent = rae.DiscreteProjectOptimizerAgent(env)
    agent = vpg.VPG(policy_net, value_net, policy_opt=policy_opt, value_opt=value_opt, advantage=advantage,
                    entropy_coef=entropy)

    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"policy_h{consts.W_INITIALIZER}": hidden_initializer,
        consts.VALUE_ACT_F: value_act_f,
        consts.VALUE_OUT_ACT_F: value_out_act_f,
        consts.POLICY_ARCH: policy_arch,
        consts.LEARNING_RATE_POLICY: lr_policy,
        consts.LEARNING_RATE_VALUE: lr_value,
        consts.ADVANTAGE_F: advantage.NAME,
        consts.ENTROPY_VAL: entropy,
        consts.VALUE_ARCH: value_arch,
    }
    agent_learning_params = {
        consts.NEPISODES: 500,
        consts.PRINT_INTERVAL: 1,
        consts.LAMBDA: 0.95,
        consts.GAMMA: 0.99,
        consts.BATCH_SIZE: 16
    }

    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent, optimization_agent], env,
                                                                     EXPERIMENT_NAME,
                                                                     'vpg small batch', agent_learning_params,
                                                                     agent_hyperparams, max_ep_steps=ENV_SIZE)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
