from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pingouin as pg
from stable_baselines3.ppo import PPO, MlpPolicy
import scipy.stats as st

import src.algorithms.algo_utils as autil
from src.envs.resource_allocation_env import DiscreteProjectsEnv, DiscreteProjectOptimizerAgent


class BaselinesWrapper:

    def __init__(self, baselines_agent):
        self.agent = baselines_agent

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        return self.agent.predict(s)


if __name__ == "__main__":
    env = DiscreteProjectsEnv(
        start_resource=100,
        start_cash=100,
        upkeep_cost=-1,
        min_payout=0,
        max_payout=1.5,
        size=200,
        balance_is_reward=False,
        stochastic=True)
    batch_size = 128
    n_epochs = 3

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    dummy_agent = DiscreteProjectOptimizerAgent(env)
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    agent = PPO(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

    dummy_scores = autil.evaluate_algorithm(dummy_agent, env, n_episodes=100, max_ep_steps=200, clip_action=False)
    ppo_scores_before = autil.evaluate_algorithm(BaselinesWrapper(agent), env, n_episodes=100, max_ep_steps=200, clip_action=False)

    sns.histplot(ppo_scores_before, color='blue', label='PPO before')
    sns.histplot(dummy_scores, color='yellow', label='Dummy')
    plt.legend()
    plt.show()

    agent.learn(total_timesteps=100000)

    ppo_scores_after = autil.evaluate_algorithm(BaselinesWrapper(agent), env, n_episodes=100, max_ep_steps=500,
                                                clip_action=False)
    print("PPO scores before: ", st.describe(ppo_scores_before))
    print("PPO scores after: ", st.describe(ppo_scores_after))
    print("Dummy scores: ", st.describe(dummy_scores))

    result_df = pd.DataFrame({'PPO before': ppo_scores_before, 'PPO after': ppo_scores_after, 'baseline': dummy_scores}).melt(var_name='model')
    result = pg.welch_anova(result_df, dv='value', between='model')
    print(result)
    print(pg.pairwise_gameshowell(result_df, dv='value', between='model'))

    sns.histplot(ppo_scores_before, color='blue', label='PPO before')
    sns.histplot(dummy_scores, color='yellow', label='Dummy')
    sns.histplot(ppo_scores_after, color='red', label='PPO after')
    plt.legend()
    plt.show()
