#
#
#

import sys
import os
import timeit
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../LibESN"))

from base_functions import *

def benchmark_base():
    A = 2 * np.ones((200, 200))
    radbas(A)
    #softplus(A)
    #vech(A)
    matrh(np.squeeze(A[1,1:191]), 19)

def benchmark_nb():
    A = 2 * np.ones((200, 200))
    nb_radbas(A)
    #nb_softplus(A)
    #nb_vech(A)
    nb_matrh(np.squeeze(A[1,1:191]), 19)

if __name__ == '__main__':
    result_base = timeit.repeat(
        setup='from __main__ import benchmark_base',
        stmt='benchmark_base()',
        repeat = 5,
        number = 1000
    )

    print("benchmark_base():  [seconds]")
    print(np.round(result_base, 6))

    result_nb = timeit.repeat(
        setup='from __main__ import benchmark_nb',
        stmt='benchmark_nb()',
        repeat = 5,
        number = 1000
    )

    print("benchmark_nb():  [seconds]")
    print(np.round(result_nb, 6))

