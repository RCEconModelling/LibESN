#
# Standard function library
#

import numpy as np
from numba import njit

# Base

def identity(A):
    return A

def radbas(A):
    return np.exp(-A**2)

def retanh(A):
    return np.tanh(np.maximum(A, 0))

def softplus(A):
    return np.log(1 + np.exp(A))

def vech(V):
    assert type(V) is np.ndarray
    N1, N2 = np.atleast_2d(V).shape
    assert N1 == N2
    v = np.zeros(N1*(N1+1)//2)
    j = 0
    for n in range(N1):
        v[j:(j+N1-n)] = V[n:,n]
        j += N1-n
    return v

def matrh(v, N):
    assert type(v) is np.ndarray
    M = len(v)
    assert N*(N+1)//2 == M
    V = np.zeros((N, N))
    j = 0
    for n in range(N):
        V[n:,n] = v[j:(j+N-n)]
        j += N-n
    return V


# Numba
# NOTE: these JITs time very close to 'base' functions;
#   see 'time_base_functions.py' for comparison

@njit
def nb_identity(A):
    return A

@njit
def nb_radbas(A):
    return np.exp(-A**2)

@njit
def nb_retanh(A):
    return np.tanh(np.maximum(A, 0))

@njit
def nb_softplus(A):
    return np.log(1 + np.exp(A))

@njit
def nb_vech(V):
    # assert type(V) is np.ndarray
    N1, N2 = np.atleast_2d(V).shape
    # assert N1 == N2
    v = np.zeros(N1*(N1+1)//2)
    j = 0
    for n in range(N1):
        v[j:(j+N1-n)] = V[n:,n]
        j += N1-n
    return v

@njit
def nb_matrh(v, N):
    # assert type(v) is np.ndarray
    M = len(v)
    # assert N*(N+1)//2 == M
    V = np.zeros((N, N))
    j = 0
    for n in range(N):
        V[n:,n] = v[j:(j+N-n)]
        j += N-n
    return V
