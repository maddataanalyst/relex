import gym
import tensorflow.keras as krs

import src.algorithms.advantages as adv
import src.algorithms.dummy as dummy
import src.algorithms.policy_gradient.tf2.ac as ac
import src.consts as consts
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.policy_nets as policy_nets
import src.models.base_models.tf2.value_nets as vnets
import src.utils as utils


EXPERIMENT_NAME = "AC lunar discrete"

if __name__ == "__main__":
    log = utils.prepare_default_log()
    n_steps = 50
    model_name_abbrev = f'ac_{n_steps}_gae'
    env = gym.make('LunarLander-v2')
    batch_size = 128
    n_epochs = 3

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    policy_act_f = 'relu'
    policy_arch = [128, 128]
    initializer = krs.initializers.HeNormal()
    policy_net = policy_nets.DiscretePolicyNet(
        a_dim,
        policy_arch,
        policy_act_f,
        hidden_initializer=initializer,
        out_initializer=initializer)

    value_act_f = 'relu'
    value_out_act_f = 'relu'
    value_arch = [128, 128]
    value_net = vnets.ValueNet(s_dim, value_arch, h_act=value_act_f, out_act=value_out_act_f)

    lr_policy = 0.0005
    lr_value = 0.0007
    policy_opt = krs.optimizers.Adam(lr_policy)
    value_opt = krs.optimizers.Adam(lr_value)

    advantage = adv.GAE
    entropy = 0.001

    agent = ac.ActorCrtic(policy_net, value_net, advantage=advantage, actor_opt=policy_opt, critic_opt=value_opt,
                          entropy_coef=entropy)
    agent_hyperparams = {
        consts.POLICY_ACT_F: policy_act_f,
        f"policy_h{consts.W_INITIALIZER}": initializer,
        consts.VALUE_ACT_F: value_act_f,
        consts.VALUE_OUT_ACT_F: value_out_act_f,
        consts.POLICY_ARCH: policy_arch,
        consts.LEARNING_RATE_POLICY: lr_policy,
        consts.LEARNING_RATE_VALUE: lr_value,
        consts.ADVANTAGE_F: advantage.NAME,
        consts.ENTROPY_VAL: entropy,
    }
    agent_learning_params = {
        consts.NEPISODES: 300,
        consts.PRINT_INTERVAL: 1,
        consts.LAMBDA: 0.95,
        consts.GAMMA: 0.99,
        consts.N_STEPS: n_steps
    }

    dummy_agent = dummy.DummyAgent(env)
    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'AC', agent_learning_params,
                                                                     agent_hyperparams)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
