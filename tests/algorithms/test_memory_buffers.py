import numpy as np
import pytest as pt
import numpy.testing as npt

import src.algorithms.memory_samplers as mbuff


def test_simple_memory_storage():
    # Given
    DIM_STATE = 4
    DIM_ACTION = 2
    N = 18
    BATCH_SZ = 3
    N_BATCH = N // BATCH_SZ

    states = np.arange(0, DIM_STATE * N).reshape((N, DIM_STATE))
    actions = np.arange(0, DIM_ACTION * N).reshape((N, DIM_ACTION))
    rewards = np.arange(0, N)
    v_states = np.arange(0, N)

    buffer = mbuff.SimpleMemory(N)

    for i in range(N - 1):
        s = states[i]
        sprime = states[i + 1]
        a = actions[i]
        vs = v_states[i]
        vsprime = v_states[i + 1]
        logprob = rewards[i]
        r = rewards[i]
        done = 0 if i < (N - 1) else 1
        buffer.store_transition(s, a, r, sprime, vs, vsprime, logprob, bool(done))

    # When
    memory_sample = buffer.sample(BATCH_SZ, DIM_STATE, DIM_ACTION)

    b_s, b_a, b_r, b_d, b_sprime, b_sval, b_sprime_val, b_logprob = memory_sample
    # Then
    assert b_s.shape[0] == \
           b_a.shape[0] == \
           b_r.shape[0] == \
           b_d.shape[0] == \
           b_sprime.shape[0] == \
           b_sprime_val.shape[0] == \
           b_logprob.shape[0] == N_BATCH

    for b_idx in range(N_BATCH):
        assert (0 < b_s[b_idx].shape[0] <= BATCH_SZ) and b_s[b_idx].shape[1] == DIM_STATE
        assert (0 < b_sprime[b_idx].shape[0] <= BATCH_SZ) and b_sprime[b_idx].shape[1] == DIM_STATE
        assert (0 < b_r[b_idx].shape[0] <= BATCH_SZ) and b_r[b_idx].ndim == 1
        assert (0 < b_d[b_idx].shape[0] <= BATCH_SZ) and b_d[b_idx].ndim == 1
        assert (0 < b_sval[b_idx].shape[0] <= BATCH_SZ) and b_sval[b_idx].ndim == 1
        assert (0 < b_sprime_val[b_idx].shape[0] <= BATCH_SZ) and b_sprime_val[b_idx].ndim == 1
        assert (0 < b_logprob[b_idx].shape[0] <= BATCH_SZ) and b_logprob[b_idx].ndim == 1


def test_simple_memory_storage_overflow():
    # Given
    DIM_STATE = 4
    DIM_ACTION = 2
    N = 18
    BATCH_SZ = 3
    N_BATCH = int((N / 2) // BATCH_SZ)

    states = np.arange(0, DIM_STATE * N).reshape((N, DIM_STATE))
    actions = np.arange(0, DIM_ACTION * N).reshape((N, DIM_ACTION))
    rewards = np.arange(0, N)
    v_states = np.arange(0, N)

    buffer = mbuff.SimpleMemory(int(N / 2))

    for i in range(N - 1):
        s = states[i]
        sprime = states[i + 1]
        a = actions[i]
        vs = v_states[i]
        vsprime = v_states[i + 1]
        logprob = rewards[i]
        r = rewards[i]
        done = 0 if i < (N - 1) else 1
        buffer.store_transition(s, a, r, sprime, vs, vsprime, logprob, done)

    # When
    memory_sample = buffer.sample(BATCH_SZ, DIM_STATE, DIM_ACTION)
    b_s, b_a, b_r, b_d, b_sprime, b_sval, b_sprime_val, b_logprob = memory_sample

    # Then
    assert buffer.actual_size == N / 2
    assert b_s.shape[0] == \
           b_a.shape[0] == \
           b_r.shape[0] == \
           b_d.shape[0] == \
           b_sprime.shape[0] == \
           b_sprime_val.shape[0] == \
           b_logprob.shape[0] == N_BATCH

    for b_idx in range(N_BATCH):
        assert (0 < b_s[b_idx].shape[0] <= BATCH_SZ) and b_s[b_idx].shape[1] == DIM_STATE
        assert (0 < b_sprime[b_idx].shape[0] <= BATCH_SZ) and b_sprime[b_idx].shape[1] == DIM_STATE
        assert (0 < b_r[b_idx].shape[0] <= BATCH_SZ) and b_r[b_idx].ndim == 1
        assert (0 < b_d[b_idx].shape[0] <= BATCH_SZ) and b_d[b_idx].ndim == 1
        assert (0 < b_sval[b_idx].shape[0] <= BATCH_SZ) and b_sval[b_idx].ndim == 1
        assert (0 < b_sprime_val[b_idx].shape[0] <= BATCH_SZ) and b_sprime_val[b_idx].ndim == 1
        assert (0 < b_logprob[b_idx].shape[0] <= BATCH_SZ) and b_logprob[b_idx].ndim == 1

def test_episodic_memory_storage():
    # Given
    DIM_STATE = 4
    DIM_ACTION = 2
    N = 20
    BATCH_SZ = 3
    N_BATCH = N // BATCH_SZ
    EP_MAX_SIZE = 10

    states = np.arange(0, DIM_STATE * N).reshape((N, DIM_STATE))
    actions = np.arange(0, DIM_ACTION * N).reshape((N, DIM_ACTION))
    rewards = np.arange(0, N)
    v_states = np.arange(0, N)

    buffer = mbuff.TrajectoryBuffer(N, EP_MAX_SIZE)
    for i in range(N - 1):
        s = states[i]
        sprime = states[i + 1]
        a = actions[i]
        vs = v_states[i]
        vsprime = v_states[i + 1]
        logprob = rewards[i]
        r = rewards[i]
        done = 0 if i < (N - 1) else 1
        ep = i // 10
        buffer.store_transition(s, a, r, sprime, vs, vsprime, logprob, done, ep)

    # then
    assert len(buffer.episode_memories) == 2
    assert buffer.actual_size == N - 1
    for i in range(len(buffer.episode_memories)):
        buffer.episode_memories[i].actual_size == EP_MAX_SIZE


def test_episodic_memory_storage_mem_overflow():
    # Given
    DIM_STATE = 4
    DIM_ACTION = 2
    N = 20
    BATCH_SZ = 3
    N_BATCH = N // BATCH_SZ
    EP_MAX_SIZE = 5

    states = np.arange(0, DIM_STATE * N).reshape((N, DIM_STATE))
    actions = np.arange(0, DIM_ACTION * N).reshape((N, DIM_ACTION))
    rewards = np.arange(0, N)
    v_states = np.arange(0, N)

    buffer = mbuff.TrajectoryBuffer(int(N / 2), EP_MAX_SIZE)
    for i in range(N - 1):
        s = states[i]
        sprime = states[i + 1]
        a = actions[i]
        vs = v_states[i]
        vsprime = v_states[i + 1]
        logprob = rewards[i]
        r = rewards[i]
        done = 0 if i < (N - 1) else 1
        ep = i // 10
        buffer.store_transition(s, a, r, sprime, vs, vsprime, logprob, done, ep)

    # then
    assert len(buffer.episode_memories) == 2
    assert buffer.actual_size == (N/2)
    for i in range(len(buffer.episode_memories)):
        assert buffer.episode_memories[i].actual_size == EP_MAX_SIZE
