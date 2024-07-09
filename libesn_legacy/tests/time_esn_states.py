#
#
#

import sys
import os
import timeit
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from libesn_legacy.esn_states import collect_states, nb_collect_states_tanh
from libesn_legacy.esn_states import iterate_state, nb_iterate_state_tanh
from libesn_legacy.matrix_generator import matrixGenerator

def benchmark_base():
    A = matrixGenerator((50, 50), seed=123)
    C = matrixGenerator((50, 10), seed=1234)
    D = matrixGenerator((50, 50), seed=12345)
    zeta = matrixGenerator((50, ), seed=123456)
    rho = 0.5
    gamma = 2
    leak = 0

    input = matrixGenerator((500, 10), dist="sparse_uniform", sparsity=0.2)
    init = np.zeros(50)

    collect_states(input, np.tanh, A, C, zeta, rho, gamma, leak, init)
    iterate_state(init, 100, np.tanh, A, D, zeta, rho, gamma, leak)

def benchmark_nb():
    A = matrixGenerator((50, 50), seed=123)
    C = matrixGenerator((50, 10), seed=1234)
    D = matrixGenerator((50, 50), seed=12345)
    zeta = matrixGenerator((50, ), seed=123456)
    rho = 0.5
    gamma = 2
    leak = 0
    
    input = matrixGenerator((500,10), dist="sparse_uniform", sparsity=0.2)
    init = np.zeros(50)

    nb_collect_states_tanh(input, A, C, zeta, rho, gamma, leak, init)
    nb_iterate_state_tanh(init, 100, A, D, zeta, rho, gamma, leak)

if __name__ == '__main__':
    result_base = timeit.repeat(
        setup='from __main__ import benchmark_base',
        stmt='benchmark_base()',
        repeat = 5,
        number = 100
    )

    print("benchmark_base():  [seconds]")
    print(np.round(result_base, 6))

    result_nb = timeit.repeat(
        setup='from __main__ import benchmark_nb',
        stmt='benchmark_nb()',
        repeat = 5,
        number = 100
    )

    print("benchmark_nb():  [seconds]")
    print(np.round(result_nb, 6))

