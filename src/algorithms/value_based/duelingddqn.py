from typing import Tuple

import numpy as np
import tensorflow.keras as krs

import src.algorithms.commons as rl_commons
import src.algorithms.memory_samplers as mbuff
import src.algorithms.value_based.ddqn as ddqn
import src.models.base_models.tf2.q_nets as qnets
from src import consts


class DuelingDDQN(ddqn.DDQN):

    def __init__(self,
                 online_qnet: qnets.DuelingQNet,
                 target_qnet: qnets.DuelingQNet,
                 buffer: mbuff.SimpleMemory,
                 eps_start: float,
                 eps_min: float,
                 eps_decay_rate: float = 0.1,
                 net_optimizer: krs.optimizers.Optimizer = krs.optimizers.Adam(0.01),
                 loss_function: str = 'mean_squared_error',
                 action_clipper: rl_commons.ActionClipper = rl_commons.DefaultActionClipper(),
                 polyak_tau: float = 0.005,
                 name: str = "DuelingDDQN",
                 target_update_frequency: int = 5):
        super().__init__(online_qnet, target_qnet, buffer, eps_start, eps_min, eps_decay_rate, net_optimizer,
                         loss_function, action_clipper, polyak_tau, name, target_update_frequency)

    def choose_action(self, s: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        epsilon = kwargs.get(consts.EPSILON, 0.0)
        p = np.random.rand()
        if p < epsilon:
            a = np.random.randint(self.online_qnet.action_dim)
            return np.array(a), None
        else:
            advantage = self.online_qnet.advantage(np.atleast_2d(s)).numpy()
            return advantage.argmax(axis=1).squeeze(), None
