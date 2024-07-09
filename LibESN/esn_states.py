import pandas as pd
import numpy as np
from numba import njit

from libesn.ufuncs import *
from libesn.datautils import * 

"""
Methods to collect and iterate states for ESN models.
"""

# Base
def states(input, map, A, C, zeta, rho=1, gamma=1, leak=0, init=None):
    r"""
    Collects the states of an ESN model given an input and the model's parameters.

    + `input`: np.ndarray, matrix of input data of shape $T \times K$, where $T \geq 1$ is 
        the sample size and $K \geq 1$ is the number of input features/covariates.
    + `map`: function, activation function to apply to the model's states.
    + `A`: np.ndarray, matrix of shape $N \times N$.
    + `C`: np.ndarray, matrix of shape $N \times K$.
    + `zeta`: np.ndarray, vector of shape $N \times 1$.
    + `rho`: float, reservoir (connectivity) matrix spectral radius, $\rho \in [0, \infty)$.
    + `gamma`: float, reservoir input scaling, $\gamma \in (0, \infty)$.
    + `leak`: float, reservoir leak rate, $\alpha \in [0, 1]$.
    + `init`: np.ndarray, initial state. Default is `None`, equivalent to setting `init` 
        equal to a $N$-dimensional zero vector.

    Supposing `input` is a $T \times K$ matrix and that `map` is a valid element-wise function 
    $\sigma : \mathbb{R} \to \mathbb{R}$ (e.g. a `np.ufunc`), state collection is performed by 
    iterating the following equation:

    $$ X_t = \alpha X_{t-1} + (1 - \alpha) \sigma(\rho A X_{t-1} + \gamma C Z_t + \zeta) $$

    where $t \in 1, \ldots, T$ and $X_0 =$`init`. 
    
    This final output is a matrix `states`$= (X_1, \ldots, X_T)$ of shape $T \times N$.

    This function will call a specific implementation based on `map`:	

    + If `map` is `np.tanh`, the states will be collected using JIT-compiled `nb_collect_states_tanh`.
    + If `map` is `libesn.ufuncs.identity` or `libesn.base_functions.nb_identity`, the states 
        will be collected using JIT-compiled `nb_collect_states_linear`.
    + If `map` is another generic function, the states will be collected using `collect_states`.
    """

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
    r"""
    Collects the states of an ESN model given an input and the model's parameters, 
    see `states` function above for details.

    Generic reference implementation of the state collection equation.
    """

    N = A.shape[0]
    T = input.shape[0]

    X = np.empty((T, N))
    
    X[0,:] = leak * init + (1-leak) * map(rho*(A @ init) + gamma*(C @ input[0,:]) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * map(rho*(A @ X[t-1,:]) + gamma*(C @ input[t,:]) + zeta)

    return X

@njit
def nb_collect_states_tanh(input, A, C, zeta, rho, gamma, leak, init):
    r"""
    Collects the states of an ESN model given an input and the model's parameters, 
    see `states` function above for details.
    
    This is a JIT-compiled version of `collect_states` using `np.tanh` as the activation function:

    $$ X_t = \alpha X_{t-1} + (1 - \alpha) \textnormal{tanh}(\rho A X_{t-1} + \gamma C Z_t + \zeta) $$
    """

    N = A.shape[0]
    T = input.shape[0]

    X = np.empty((T, N))
    
    X[0,:] = leak * init + (1-leak) * np.tanh(rho*(A @ init) + gamma*(C @ input[0,:]) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * np.tanh(rho*(A @ X[t-1,:]) + gamma*(C @ input[t,:]) + zeta)

    return X

@njit
def nb_collect_states_linear(input, A, C, zeta, rho, gamma, leak, init):
    r"""
    Collects the states of an ESN model given an input and the model's parameters, 
    see `states` function above for details.
    
    This is a JIT-compiled version of `collect_states` using identity as the activation function:

    $$ X_t = \alpha X_{t-1} + (1 - \alpha) (\rho A X_{t-1} + \gamma C Z_t + \zeta) $$
    """

    N = A.shape[0]
    T = input.shape[0]

    X = np.empty((T, N))
    
    X[0,:] = leak * init + (1-leak) * (rho*(A @ init) + gamma*(C @ input[0,:]) + zeta)
    for t in range(1, T):
        X[t,:] = leak * X[t-1,:] + (1-leak) * (rho*(A @ X[t-1,:]) + gamma*(C @ input[t,:]) + zeta)

    return X

# NOTE: This alternative (parallel) version is SLOWER than simple @njit
#
# @njit
# def nb_collect_states_tanh_alt(input, A, C, zeta, rho, gamma, leak, init):
#     N = A.shape[0]
#     T = input.shape[0]

#     X = np.empty((T, N))
#     X[0,:] = nb_step_state_tanh(init, input[0,:], A, C, zeta, rho, gamma, leak)
#     for t in range(1, T):
#         X[t,:] = nb_step_state_tanh(X[t-1,:], input[t,:], A, C, zeta, rho, gamma, leak)

#     return X
    
# @njit(parallel=True)
# def nb_step_state_tanh(state, input, A, C, zeta, rho, gamma, leak):
#     S1 = np.tanh(rho*(A @ state) + gamma*(C @ input) + zeta)
#     return leak * state + (1-leak) * S1


## FORWARD STATE ITERATION --------------------------------------------

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
    r"""
    Iterates forward the states of an ESN model given a matrix of collected states,
    a matrix of autonomous output coefficients and the model's parameters.

    + `states`: np.ndarray, matrix of input data of shape $T \times N$, where $T \geq 1$ is 
        the state sample size and $N \geq 1$ is the size of the state space.
    + `length`: int, length of the state iteration, $\textnormal{length} \geq 1$.
    + `W`: np.ndarray, matrix of shape $N \times K$ of autonomous output coefficients.
    + `map`: function, activation function to apply to the model's states.
    + `A`: np.ndarray, matrix of shape $N \times N$.
    + `C`: np.ndarray, matrix of shape $N \times K$.
    + `zeta`: np.ndarray, vector of shape $N \times 1$.
    + `rho`: float, reservoir (connectivity) matrix spectral radius, $\rho \in [0, \infty)$.
    + `gamma`: float, reservoir input scaling, $\gamma \in (0, \infty)$.
    + `leak`: float, reservoir leak rate, $\alpha \in [0, 1]$.
    + `start`: int, index to start state iteration from, must be $0 \leq \textnormal{start} < T$. Default is `0`.
    + `stride`: int, stride for state iteration, must be $\textnormal{stride} \geq 1$. Default is `1`.
    + `states_slice`: tuple, list or range of indices to iterate. Default is `None`. 
    
    This method will iterate all states indexed by `states_slice`.
    When argument `states_slice` is `None`, the index will be set to `states_slice = range(start, T, stride)`.

    Input states are supplied in matrix `states`$= X_t$ of shape $T \times N$. For each $t$ in `states_slice`
    the state iteration is performed by iterating the following equation:

    $$ X_{t+h} = \alpha X_{t+h-1} + (1 - \alpha) \sigma(\rho A X_{t+h-1} + \gamma C W' X_{t+h-1} + \zeta) $$

    where $h = 1, \ldots, \textnormal{length}$. This will yield a matrix $X_{t:t+h} := (X_{t+1}, \ldots, X_{t+h})$ 
    of shape $h \times N$ for each $t$. 

    The final output is a 3D tensor of shape `len(states_slice)`$\times N \times \textnormal{length}$, where 
    matrices $X_{t:t+h}$ are stacked along the first dimension.

    This function will call a specific implementation based on `map`:	

    + If `map` is `np.tanh`, the states will be collected using JIT-compiled `nb_iterate_state_tanh`.
    + If `map` is `libesn.ufuncs.identity` or `libesn.ufuncs.nb_identity`, the states 
        will be collected using JIT-compiled `nb_iterate_state_linear`.
    + If `map` is another generic function, the states will be collected using `iterate_state`.
    """
    # Flatten
    # states = np.atleast_2d(states)
    T, N = states.shape

    # Length check
    assert type(length) is int and length >= 1, "length must be a positive integer"
    assert type(start) is int and start >= 0 and start < T, "length must be a non-negative integer"
    assert type(stride) is int and stride >= 1, "stride must be a positive integer"

    # Print debug
    if debug:
        print(f"[+] libesn.generate() - - - - - - - - - - -")
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

