#
#
#

import sys
import os
import timeit
import numpy as np

libpath = os.path.join(os.path.dirname(__file__), "..//")
print(libpath)
sys.path.append(libpath)

from libesn.matgen import matrixGenerator
from libesn.esn_states import *
from libesn.esn_states import *

INPUT_LENGTH = 500

INPUT_SIZE = 10
RESERVOIR_SIZE = 50

def benchmark_base():
    A = matrixGenerator((RESERVOIR_SIZE, RESERVOIR_SIZE), seed=123)
    C = matrixGenerator((RESERVOIR_SIZE, INPUT_SIZE), seed=1234)
    D = matrixGenerator((RESERVOIR_SIZE, RESERVOIR_SIZE), seed=12345)
    zeta = matrixGenerator((RESERVOIR_SIZE, ), seed=123456)
    rho = 0.5
    gamma = 2
    leak = 0

    input = matrixGenerator((INPUT_LENGTH, INPUT_SIZE), dist="uniform", sparsity=0.2, seed=9876)
    init = np.zeros(RESERVOIR_SIZE)

    collect_states(input, np.tanh, A, C, zeta, rho, gamma, leak, init)
    # iterate_state(init, 500, np.tanh, A, D, zeta, rho, gamma, leak)

def benchmark_nb():
    A = matrixGenerator((RESERVOIR_SIZE, RESERVOIR_SIZE), seed=123)
    C = matrixGenerator((RESERVOIR_SIZE, INPUT_SIZE), seed=1234)
    D = matrixGenerator((RESERVOIR_SIZE, RESERVOIR_SIZE), seed=12345)
    zeta = matrixGenerator((RESERVOIR_SIZE, ), seed=123456)
    rho = 0.5
    gamma = 2
    leak = 0
    
    input = matrixGenerator((INPUT_LENGTH, INPUT_SIZE), dist="uniform", sparsity=0.2, seed=9876)
    init = np.zeros(RESERVOIR_SIZE)

    nb_collect_states_tanh(input, A, C, zeta, rho, gamma, leak, init)
    # nb_iterate_state_tanh(init, 500, A, D, zeta, rho, gamma, leak)

# NOTE: This alternative (parallel) version is SLOWER than simple @njit
# def benchmark_nb_alt():
#     A = matrixGenerator((RESERVOIR_SIZE, RESERVOIR_SIZE), seed=123)
#     C = matrixGenerator((RESERVOIR_SIZE, INPUT_SIZE), seed=1234)
#     D = matrixGenerator((RESERVOIR_SIZE, RESERVOIR_SIZE), seed=12345)
#     zeta = matrixGenerator((RESERVOIR_SIZE, ), seed=123456)
#     rho = 0.5
#     gamma = 2
#     leak = 0
    
#     input = matrixGenerator((INPUT_LENGTH, INPUT_SIZE), dist="sparse_uniform", sparsity=0.2)
#     init = np.zeros(RESERVOIR_SIZE)

#     nb_collect_states_tanh_alt(input, A, C, zeta, rho, gamma, leak, init)

if __name__ == '__main__':
    result_base = timeit.repeat(
        setup='from __main__ import benchmark_base',
        stmt='benchmark_base()',
        repeat = 5,
        number = 100
    )

    print("benchmark_base():  [seconds]")
    print(np.round(result_base, 4))

    result_nb = timeit.repeat(
        setup='from __main__ import benchmark_nb',
        stmt='benchmark_nb()',
        repeat = 5,
        number = 100
    )

    print("benchmark_nb():  [seconds]")
    print(np.round(result_nb, 4))

    # result_nb = timeit.repeat(
    #     setup='from __main__ import benchmark_nb_alt',
    #     stmt='benchmark_nb_alt()',
    #     repeat = 5,
    #     number = 100
    # )

    # print("benchmark_nb_alt():  [seconds]")
    # print(np.round(result_nb, 4))

