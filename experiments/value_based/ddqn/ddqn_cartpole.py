import gym
import numpy as np
import tensorflow.keras as krs
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import src.algorithms.dummy as dummy
import src.algorithms.memory_samplers as mbuff
import src.algorithms.value_based.ddqn as dqn
import src.consts as consts
import src.experiments.experiment_utils as exu
import src.models.base_models.tf2.q_nets as qnets
import src.utils as utils

EXPERIMENT_NAME = "DDQN cartpole"

if __name__ == "__main__":
    log = utils.prepare_default_log()
    model_name_abbrev = f'ddqn'
    env = gym.make("CartPole-v1")

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    hidden_initializer = krs.initializers.HeNormal()

    lr = 0.005
    optimizer = krs.optimizers.Adam(lr)
    ep_max_steps = 200

    hidden_act = 'relu'
    out_act = 'relu'
    qnet_arch = [128, 128]
    action_net = qnets.QNet(s_dim, a_dim, qnet_arch, h_initializer=hidden_initializer, h_act=hidden_act, out_act=out_act)
    target_net = qnets.QNet(s_dim, a_dim, qnet_arch, h_initializer=hidden_initializer, h_act=hidden_act, out_act=out_act)
    buffer = mbuff.SimpleMemory(10000)
    dummy_agent = dummy.DummyAgent(env)
    agent = dqn.DDQN(
        target_net,
        action_net,
        buffer,
        0.99,
        0.05,
        0.99,
        net_optimizer=optimizer,
        polyak_tau=0.9,
        target_update_frequency=10, loss_function='huber')

    agent_hyperparams = {
        consts.HIDDEN_ACT_F: hidden_act,
        f"policy_h{consts.W_INITIALIZER}": hidden_initializer,
        consts.OUTOUT_ACT_F: out_act,
        consts.Q_NET_ARCH: qnet_arch,
        consts.LEARNING_RATE: lr,
        consts.EP_MAX_STEPS: ep_max_steps
    }
    agent_learning_params = {
        consts.NEPISODES: 200,
        consts.PRINT_INTERVAL: 1,
        consts.BATCH_SIZE: 32,
        'warmup_batches': 5
    }

    samples = []
    n_episodes = 5000
    for ep in tqdm(range(n_episodes)):
        env.reset()
        done = False
        while not done:
            a = np.random.choice(2)
            s_prime, reward, done, info = env.step(a)
            samples.append(s_prime)
            s = s_prime
    ss = StandardScaler()
    ss.fit(np.array(samples))

    scores, overall_result, pairwise_result = exu.conduct_experiment(agent, [dummy_agent], env, EXPERIMENT_NAME,
                                                                     'DDQN basic', agent_learning_params,
                                                                     agent_hyperparams, max_ep_steps=ep_max_steps, scaler=ss)
    log.info(scores.describe())
    log.info(overall_result)
    log.info(pairwise_result)
