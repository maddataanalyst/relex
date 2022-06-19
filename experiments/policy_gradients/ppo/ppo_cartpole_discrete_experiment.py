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

EXPERIMENT_NAME = "PPO cartpole discrete"

if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'ppo_gae'
    env = gym.make('CartPole-v1')

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    hidden_initializer = krs.initializers.GlorotNormal(seed=123)
    policy_act_f = 'tanh'
    policy_arch = [128, 128]
    policy_net = policy_nets.DiscretePolicyNet(a_dim, policy_arch, h_act=policy_act_f,
                                               hidden_initializer=hidden_initializer)

    value_act_f = 'relu'
    value_out_act_f = 'relu'
    value_arch = [128, 128]
    value_initializer = krs.initializers.HeNormal(seed=456)
    value_net = vnets.ValueNet(s_dim, value_arch, out_act=value_out_act_f, h_act=value_act_f,
                               initializer=value_initializer)

    lr = 0.0005
    policy_opt = krs.optimizers.Adam(lr)
    value_opt = krs.optimizers.Adam(lr)

    n_agents = 3
    clipping_eps = 0.05

    advantage = adv.GAE
    entropy = 0.0001
    dummy_agent = dummy.DummyAgent(env)
    agent = ppo.PPO(
        policy_net,
        value_net,
        advantage=advantage,
        n_agents=n_agents,
        clipping_eps=clipping_eps,
        entropy_coef=entropy,
        policy_opt=policy_opt,
        value_opt=value_opt)

    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"policy_h{consts.W_INITIALIZER}": hidden_initializer,
        consts.VALUE_ACT_F: value_act_f,
        consts.VALUE_OUT_ACT_F: value_out_act_f,
        consts.POLICY_ARCH: policy_arch,
        consts.LEARNING_RATE: lr,
        consts.ADVANTAGE_F: advantage.NAME,
        consts.ENTROPY_VAL: entropy,
        consts.N_AGENTS: n_agents,
        consts.CLIPPING_EPS: clipping_eps
    }
    agent_learning_params = {
        consts.NEPISODES: 100,
        consts.PRINT_INTERVAL: 1,
        consts.LAMBDA: 0.95,
        consts.GAMMA: 0.99,
        consts.BATCH_SIZE: 32
    }

    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'PPO gae', agent_learning_params,
                                                                     agent_hyperparams)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
