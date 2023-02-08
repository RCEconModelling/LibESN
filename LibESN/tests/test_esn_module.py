#
#
#

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from LibESN.matrix_generator import matrixGenerator
from LibESN.esn import stateParameters, ESN
from LibESN.esn_fit import ridgeFit

class TestESN(unittest.TestCase):

    def test_state_parameters(self):
        A = matrixGenerator((50, 50), seed=123)
        C = matrixGenerator((50, 10), seed=12345)
        zeta = matrixGenerator((50, ), seed=123456)
        rho = 0.5
        gamma = 2
        leak = 0

        self.assertIsInstance(stateParameters(
            np.tanh, A, C, zeta, rho, gamma, leak
        ), stateParameters)

    def test_esn_init(self):
        A = matrixGenerator((50, 50), seed=123)
        C = matrixGenerator((50, 10), seed=12345)
        zeta = matrixGenerator((50, ), seed=123456)
        rho = 0.5
        gamma = 2
        leak = 0

        self.assertIsInstance(ESN(
            None,
            smap=np.tanh, 
            A=A, C=C, zeta=zeta, 
            rho=rho, gamma=gamma, leak=leak
        ), ESN)

    def test_esn_fit(self):
        A = matrixGenerator((50, 50), seed=123)
        C = matrixGenerator((50, 2), seed=12345)
        zeta = matrixGenerator((50, ), seed=123456)
        rho = 0.5
        gamma = 2
        leak = 0

        esn = ESN(
            None,
            smap=np.tanh, 
            A=A, C=C, zeta=zeta, 
            rho=rho, gamma=gamma, leak=leak
        )

        # Random data
        data = (
            np.vstack((
                np.sin(np.linspace(0, 10, 100).T / 10),
                np.sin(np.linspace(0, 10, 100) / 3 + 7)
            )).T,
            np.cos(np.linspace(0, 10, 100) / 5)
        )

        W = esn.fit(data, ridgeFit(1))

        self.assertIsNotNone(W)

if __name__ == '__main__':
    unittest.main()