#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

import pandas as pd
import numpy as np
from numba import njit

from LibESN.base_functions import *
from LibESN.base_utils import * 

# Base
def states(input, map, A, C, zeta, rho=1, gamma=1, leak=0, init=None):
    # Flatten
    # input = np.atleast_2d(input)

    A_shape = A.shape
    C_shape = C.shape
    zeta_shape = zeta.shape

    # Shape checks
    assert A_shape[0] == A_shape[1], "A matrix is not square"
    assert C_shape[0] == A_shape[0], "A and C matrices are not compatible"
    assert zeta_shape[0] == A_shape[0], "A and zeta are not compatible"
    assert C_shape[1] == input.shape[1]

    # Coefficient checks
    assert rho >= 0
    assert leak >= 0 and leak <= 1

    if init is None:
        init = np.zeros(A_shape[0])
    else:
        init = np.squeeze(init)
        assert len(init) == A_shape[0], "A and init are not compatible"

    if map == np.tanh:
        states = nb_collect_states_tanh(input, A, C, zeta, rho, gamma, leak, init)
    elif map == identity or map == nb_identity:
        states = nb_collect_states_linear(input, A, C, zeta, rho, gamma, leak, init)
    else:
        states = collect_states(input, map, A, C, zeta, rho, gamma, leak, init)

    return states

# NOTE: REFERENCE
#       [@njit version ~6x faster]
def collect_states(input, map, A, C, zeta, rho, gamma, leak, init):
    N = A.shape[0]
    T = input.shape[0]

    X = np.empty((T, N))
    
    X[0,:] = leak * init + (1-leak) * map(rho*(A @ init) + gamma*(C @ input[0,:]) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * map(rho*(A @ X[t-1,:]) + gamma*(C @ input[t,:]) + zeta)

    return X

@njit
def nb_collect_states_tanh(input, A, C, zeta, rho, gamma, leak, init):
    N = A.shape[0]
    T = input.shape[0]

    X = np.empty((T, N))
    
    X[0,:] = leak * init + (1-leak) * np.tanh(rho*(A @ init) + gamma*(C @ input[0,:]) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * np.tanh(rho*(A @ X[t-1,:]) + gamma*(C @ input[t,:]) + zeta)

    return X

@njit
def nb_collect_states_linear(input, A, C, zeta, rho, gamma, leak, init):
    N = A.shape[0]
    T = input.shape[0]

    X = np.empty((T, N))
    
    X[0,:] = leak * init + (1-leak) * (rho*(A @ init) + gamma*(C @ input[0,:]) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * (rho*(A @ X[t-1,:]) + gamma*(C @ input[t,:]) + zeta)

    return X


def iter_state(state, length, map, A, D, zeta, rho=1, gamma=1, leak=0):
    # Flatten
    # state = np.squeeze(state)

    A_shape = A.shape
    D_shape = D.shape
    zeta_shape = zeta.shape

    # Shape checks
    assert A_shape[0] == A_shape[1], "A matrix is not square"
    assert D_shape[0] == A_shape[0], "A and D matrices are not compatible"
    assert zeta_shape[0] == A_shape[0], "A and zeta are not compatible"
    assert A_shape[1] == len(state)

    # Coefficient checks
    assert rho >= 0
    assert leak >= 0 and leak <= 1

    # Length check
    assert type(length) is int and length >= 1, "length must be a positive integer"

    #if input is None:
    #    input = np.zeros(A_shape[0])
    #else:
    #    input = np.squeeze(input)
    #    assert len(input) == A_shape[0], "A matrix and input are not compatible"

    if map == np.tanh:
        states = nb_iterate_state_tanh(state, length, A, D, zeta, rho, gamma, leak)
    elif map == identity or map == nb_identity:
        states = nb_iterate_state_linear(state, length, A, D, zeta, rho, gamma, leak)
    else:
        states = iterate_state(state, length, map, A, D, zeta, rho, gamma, leak)

    return states

# NOTE: REFERENCE 
#       [@njit version ~3x faster]
def iterate_state(state, length, map, A, D, zeta, rho, gamma, leak):
    N = A.shape[0]
    T = length

    X = np.empty((T, N))
    
    # print(state.shape)
    # print(D.shape)
    #v = np.concatenate((np.ones(1), state))

    X[0,:] = leak * state + (1-leak) * map(rho*(A @ state) + gamma*(D @ np.concatenate([np.ones(1), state])) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * map(rho*(A @ X[t-1,:]) + gamma*(D @ np.concatenate([np.ones(1), X[t-1,:]])) + zeta)

    return X

@njit
def nb_iterate_state_tanh(state, length, A, D, zeta, rho, gamma, leak):
    N = A.shape[0]
    T = length

    X = np.empty((T, N))

    # Compacted state matrix
    d0 = D[:,0]
    rAgD = rho * A + gamma * D[:,1:]
    
    X[0,:] = leak * state + (1-leak) * np.tanh(rAgD @ state + d0 + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * np.tanh(rAgD @ X[t-1,:] + d0 + zeta)

    return X

@njit
def nb_iterate_state_linear(state, length, A, D, zeta, rho, gamma, leak):
    N = A.shape[0]
    T = length

    X = np.empty((T, N))

    # Compacted state matrix
    rAgD = rho*A + gamma*D
    
    X[0,:] = leak * state + (1-leak) * (rAgD @ state + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * (rAgD @ X[t-1,:] + zeta)

    return X

def generate(
    states, 
    length,
    W, 
    map, 
    A, 
    C, 
    zeta, 
    rho=1, 
    gamma=1, 
    leak=0, 
    start=0,
    stride=1,
    states_slice=None,
    debug=False,
):
    # Flatten
    # states = np.atleast_2d(states)
    T, N = states.shape

    # Length check
    assert type(length) is int and length >= 1, "length must be a positive integer"
    assert type(start) is int and start >= 0, "length must be a non-negative integer"
    assert type(stride) is int and stride >= 1, "stride must be a positive integer"

    # Print debug
    if debug:
        print(f"[+] LibESN.generate() - - - - - - - - - - -")
        print(f" .  input.shape: {states.shape}")
        print(f" .  start:       {start}")
        print(f" .  step:        {stride}")
        sl_ = [i for i in range(start, T, stride)]
        print(f" -> indices: {sl_[0]}, {sl_[1]}, ..., {sl_[-2]}, {sl_[-1]}")

    # iteration matrix
    D = C @ W.T

    # indices
    if states_slice is None:
        states_slice = range(start, T, stride)
    else:
        assert type(states_slice in [tuple, list, range]), "states_slice is not a tuple, list or range"

    # generate
    Xg = np.empty((len(states_slice), N, length))
    for i, t in enumerate(states_slice):
        # autostates
        its = iter_state(
            state=states[t,:], 
            length=length, 
            map=map, 
            A=A, D=D, zeta=zeta, 
            rho=rho, gamma=gamma, leak=leak,
        )
        #Xg[i,:,:] = np.reshape(its, (N, length))
        Xg[i,:,:] = its.T

    return Xg

