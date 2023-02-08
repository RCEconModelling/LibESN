#
#
#

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from LibESN.matrix_generator import matrixGenerator
from LibESN.esn import ESN
from LibESN.esn_fit import ridgeFit

A = matrixGenerator((50, 50), seed=123)
C = matrixGenerator((50, 2), seed=12345)
zeta = matrixGenerator((50, ), seed=123456)
rho = 0.9
gamma = 0.1
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
        3 + np.sin(np.linspace(0, 10, 100) / 10),
        np.cos(np.linspace(0, 10, 100) / 3 + 7)
    )).T,
    np.vstack((
        3 + np.sin(np.linspace(1, 11, 100) / 10),
        np.cos(np.linspace(1, 11, 100) / 3 + 7)
    )).T
)

fit = esn.fit(data, ridgeFit(0.1))

fitdm = esn.fitDirectMultistep(data, ridgeFit(0.1), steps=5)

fitms = esn.fitMultistep(data, ridgeFit(0.1), steps=5)

# print(fit['residuals'])

