import numpy as np
import dataclasses

from collections import deque
from typing import List, Tuple



@dataclasses.dataclass
class MemorySample:

    states: np.array      # dim: batch x dim s
    actions: np.array     # dim: batch x dim a
    rewards: np.array     # dim: batch x 1
    dones: np.array       # dim: batch x 1
    sprimes: np.array     # dim: batch x dim s
    svals: np.array       # dim: batch x dim s
    sprime_vals: np.array # dim: batch x dim s
    logprobs: np.array    # dim: batch x 1

    def __iter__(self):
        return iter(dataclasses.astuple(self))


class SimpleMemory:
    """
    A simple memory collector, that stores states, actions, rewards, s_primes and done signals.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.actual_size = 0

        self.states = []
        self.s_vals = []

        self.actions = []
        self.logprobs = []

        self.rewards = []

        self.sprimes = []
        self.sprime_vals = []

        self.dones = []

    def store_transition(self,
                         s: np.array,
                         a: np.array,
                         r: float,
                         sprime: np.array,
                         vs: np.array,
                         v_sprime: np.array,
                         logprob: np.array,
                         done: bool,
                         *args, **kwargs):
        """
        Stores a single transition from s --> action --> sprime with additional information of v(s), v(sprime) and
        an action logprob.

        Parameters
        ----------
        s: np.array
            Initial state

        a: np.array
            Action array.

        r: float
            Reward info.

        sprime: np.array
            Sprime (state after transition).

        vs: np.array
            V(s) - state value.

        v_sprime: np.array
            V(sprime) - state value.

        logprob: np.array
            Action logprob

        done: bool
            Is episode done?
        """
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.dones.append(done)
        self.sprimes.append(sprime)
        self.s_vals.append(vs)
        self.sprime_vals.append(v_sprime)
        self.logprobs.append(logprob)

        if len(self.states) > self.max_size:
            self.fifo()
        else:
            self.actual_size = len(self.states)

    def fifo(self):
        """
        Removes first stored transitions if buffer is already full.
        """
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.dones.pop(0)
        self.sprimes.pop(0)
        self.s_vals.pop(0)
        self.sprime_vals.pop(0)
        self.logprobs.pop(0)

        self.actual_size = len(self.states)

    def sample(self, batch_size: int, dim_s: int, dim_a: int, preserve_order: bool = True, *args, **kwargs) -> MemorySample:
        if preserve_order:
            return self._sample_preserve_trajectory(batch_size, dim_s, dim_a, *args, **kwargs)
        else:
            return self._sample_random(batch_size, dim_s, dim_a)

    def _sample_preserve_trajectory(self, batch_size: int, dim_s: int, dim_a: int, *args, **kwargs) -> MemorySample:
        """
        Samples transitions given a batch size. Transitions are NOT SAMPLED randomly - order is preserved.
        Whole memory is divided into chunks of size = batch_size, where order of transitions is the same
        as encountered during game. This feature is used for mostly for the on-policy algorithms, where the whole
        trajectory needs to be preserved.

        Parameters
        ----------
        batch_size: int
            Batch size to be sampled.

        dim_s: int
            Dimension of state

        dim_a: int
            Dimension of action

        Returns
        -------
        MemorySample
            Tuple of: [state, action, reward, done, sprime, v(s), v(sprime), logprob(a)]
        """
        batch_start_idx = np.arange(0, self.actual_size, batch_size)
        np.random.shuffle(batch_start_idx)
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_sprimes = []
        batch_s_vals = []
        batch_sprime_vals = []
        batch_logprobs = []

        for start_idx in batch_start_idx:
            end_idx = start_idx + batch_size
            from_to = slice(start_idx, end_idx)
            nobs = len(self.states[from_to])
            batch_states.append(np.array(self.states[from_to]).reshape((nobs, dim_s)))
            batch_actions.append(np.array(self.actions[from_to]).reshape((nobs, dim_a)))
            batch_rewards.append(np.array(self.rewards[from_to]).reshape(nobs,))
            batch_dones.append(np.array(self.dones[from_to]).reshape((nobs, )))
            batch_sprimes.append(np.array(self.sprimes[from_to]).reshape((nobs, dim_s)))
            batch_s_vals.append(np.array(self.s_vals[from_to]).reshape((nobs, )))
            batch_sprime_vals.append(np.array(self.sprime_vals[from_to]).reshape((nobs, )))
            batch_logprobs.append(np.array(self.logprobs[from_to]).reshape((nobs, )))

        return MemorySample(
                np.array(batch_states),
                np.array(batch_actions),
                np.array(batch_rewards),
                np.array(batch_dones),
                np.array(batch_sprimes),
                np.array(batch_s_vals),
                np.array(batch_sprime_vals),
                np.array(batch_logprobs))

    def _sample_random(self, batch_size: int, dim_s: int, dim_a: int) -> MemorySample:
        """
        Samples random batches of elements, without preserving the whole trajectory. Used mostly for off-policy
        algorithms.

        Parameters
        ----------
        batch_size: int
            Batch size

        dim_s: int
            Dimension of state

        dim_a: int
            Dimension of action

        Returns
        -------
        MemorySample
            Tuple of: [state, action, reward, done, sprime, v(s), v(sprime), logprob(a)]
        """
        indices = np.arange(0, self.actual_size)
        sample_idx = np.random.choice(indices, batch_size)
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_sprimes = []
        batch_s_vals = []
        batch_sprime_vals = []
        batch_logprobs = []

        for idx in sample_idx:
            batch_states.append(self.states[idx])
            batch_actions.append(self.actions[idx])
            batch_rewards.append(self.rewards[idx])
            batch_dones.append(np.array(self.dones[idx]))
            batch_sprimes.append(self.sprimes[idx])
            batch_s_vals.append(self.s_vals[idx])
            batch_sprime_vals.append(self.sprime_vals[idx])
            batch_logprobs.append(self.logprobs[idx])

        nobs = len(batch_states)
        return MemorySample(
                np.array(batch_states).reshape((nobs, dim_s)),
                np.array(batch_actions).reshape((nobs, dim_a)),
                np.array(batch_rewards).reshape((nobs, )),
                np.array(batch_dones).reshape((nobs, )),
                np.array(batch_sprimes).reshape((nobs, dim_s)),
                np.array(batch_s_vals).reshape((nobs, )),
                np.array(batch_sprime_vals).reshape((nobs, )),
                np.array(batch_logprobs).reshape((nobs, )))

    def clear_all(self):
        """
        Remove all stored tranistions and reset memory.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.sprimes = []
        self.s_vals = []
        self.sprime_vals = []
        self.logprobs = []
        self.actual_size = 0


class TrajectoryBuffer(SimpleMemory):

    def __init__(self, max_size: int, episode_max_size: int):
        if episode_max_size > max_size:
            raise ValueError("Episode max size must be lower than total buffer size")
        super(TrajectoryBuffer, self).__init__(max_size)
        self.ep_max_size = episode_max_size
        self.episode_memories: List[SimpleMemory] = []
        self.actual_size = 0

    def store_transition(
            self,
            s: np.array,
            a: np.array,
            r: float,
            sprime: np.array,
            vs: np.array,
            v_sprime: np.array,
            logprob: np.array,
            done: bool,
            episode: int = 0,
            *args,
            **kwargs):

        if episode < len(self.episode_memories):
            ep: SimpleMemory = self.episode_memories[episode]
            ep.store_transition(s, a, r, sprime, vs, v_sprime, logprob, done, episode)

        else:
            mem = SimpleMemory(self.ep_max_size)
            mem.store_transition(s, a, r, sprime, vs, v_sprime, logprob, done, episode)
            self.episode_memories.append(mem)
        self._calculate_size()
        if self.actual_size > self.max_size:
            self.fifo()

    def _calculate_size(self):
        self.actual_size = sum([ep_mem.actual_size for ep_mem in self.episode_memories])

    def clear_all(self):
        self.episode_memories: List[SimpleMemory] = []

    def fifo(self):
        if len(self.episode_memories) > 0:
            ep0 = self.episode_memories[0]
            ep0.fifo()
            if ep0.actual_size == 0:
                self.episode_memories.pop(0)
