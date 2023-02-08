#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

from typing import Union

import numpy as np
# import pandas as pd
# from numba import njit

from LibESN.base_utils import *
from LibESN.matrix_generator import matrixGenerator
from LibESN.esn_states import states

from LibESN.console import console
from rich.table import Table
from rich import box

class stateParameters:
    def __init__(
        self, 
        smap: np.ufunc, 
        A: np.ndarray, 
        C: np.ndarray, 
        zeta: np.ndarray = None, 
        rho: Union[int, float, np.ndarray] = 0, 
        gamma: Union[int, float, np.ndarray] = 1, 
        leak: Union[int, float, np.ndarray] = 0,
    ) -> None:
        A_shape = A.shape
        C_shape = C.shape

        # preliminary shape checks
        assert A_shape[0] == A_shape[1], "A matrix is not square"
        assert C_shape[0] == A_shape[0], "A and C matrices are not compatible"

        # fix state space size
        self.size = A_shape[0]

        # prehemptive allocations
        if zeta is None:
            zeta = np.zeros(self.size)
        else:
            if len(zeta.shape) == 1:
                assert zeta.shape == (self.size, ), "A and zeta are not compatible"
            elif len(zeta.shape) == 2:
                zeta = np.squeeze(zeta)
                assert zeta.shape == (self.size, ), "A and zeta are not compatible"
            else:
                raise ValueError("zeta must be a 1D or 2D vector")

        # coefficient checks
        assert rho >= 0, "rho is not a nonnegative scalar"
        assert leak >= 0 and leak <= 1, "leak is not a scalar in interval [0,1]"

        # shapes
        self.N = A_shape[0]
        self.K = C_shape[1]
        # state map
        self.smap = smap
        # reservoir (connectivity) matrix 
        self.A = np.copy(A)
        # input mask        
        self.C = np.copy(C)
        # input shift
        self.zeta = np.copy(zeta)
        # reservoir (connectivity) matrix spectral radius
        self.rho = np.copy(rho)
        # input scaling
        self.gamma = np.copy(gamma)
        # leak rate
        self.leak = np.copy(leak)

    def table(self) -> Table:
        table = Table(title="stateParameters", box=box.SIMPLE_HEAD)

        table.add_column("Parameter", justify="left")
        table.add_column("Shape", justify="center")
        table.add_column("Value", justify="center")

        table.add_row("smap", "-", str(self.smap))
        table.add_row("A", str(self.A.shape), "-")
        table.add_row("C", str(self.C.shape), "-")
        table.add_row("zeta", str(self.zeta.shape), "-")
        table.add_row("rho", "scalar", str(np.round(self.rho, 4)))
        table.add_row("gamma", "scalar", str(np.round(self.gamma, 4)))
        table.add_row("leak", "scalar", str(np.round(self.leak, 4)))

        return table
    
    def print(self) -> None:
        console.print(self.table())


class ESN:
    def __init__(self, 
        pars: stateParameters, 
        smap: np.ufunc = None, 
        A: np.ndarray = None, 
        C: np.ndarray = None, 
        zeta: np.ndarray = None, 
        rho: Union[int, float, np.ndarray] = 0, 
        gamma: Union[int, float, np.ndarray] = 1, 
        leak: Union[int, float, np.ndarray] = 0,
    ) -> None:
        if pars is None:
            #assert not smap is None, "pars argument not set, smap is None"
            #assert not A is None, "pars argument not set, A is None"
            #assert not C is None, "pars argument not set, C is None"
            #assert not zeta is None, "pars argument not set, zeta is None"
            #assert not rho is None, "pars argument not set, rho is None"
            #assert not gamma is None, "pars argument not set, gamma is None"
            #assert not leak is None, "pars argument not set, leak is None"

            if not smap is None and not A is None and not C is None:
                pars = stateParameters(smap, A, C, zeta, rho, gamma, leak)
        # init even if empty
        self.pars = pars

        # TODO ? : if state space pars are not None, should update
        #          the 'pars' object 
        self.N = pars.N
        self.K = pars.K

    def setup(
        self, 
        A: dict,
        C: dict,
        zeta: dict = None,
        rho: Union[int, float, np.ndarray] = 0, 
        gamma: Union[int, float, np.ndarray] = 1, 
        leak: Union[int, float, np.ndarray] = 0,
    ) -> None:
        A_mat = matrixGenerator(
            shape=A.shape,
            dist=A.dist,
            sparsity=A.sparsity,
            normalize=A.normalize,
            options=A.options,
            seed=A.seed,
        )
        self.pars.A = A_mat

        C_mat = matrixGenerator(
            shape=C.shape,
            dist=C.dist,
            sparsity=C.sparsity,
            normalize=C.normalize,
            options=C.options,
            seed=C.seed,
        )
        self.pars.C = C_mat

        if not zeta is None:
            zeta_mat = matrixGenerator(
            shape=zeta.shape,
            dist=zeta.dist,
            sparsity=zeta.sparsity,
            normalize=zeta.normalize,
            options=zeta.options,
            seed=zeta.seed,
        )
        self.pars.zeta = zeta_mat

        self.pars.rho = rho
        self.pars.gamma = gamma
        self.pars.leak = leak

    def states(self, input, **kwargs):
        # prepare data
        Z, Z_dates = pd_data_prep(input)

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']

        # collect states
        model_states = states(
            Z, 
            map=self.pars.smap, 
            A=self.pars.A, C=self.pars.C, zeta=self.pars.zeta, 
            rho=self.pars.rho, gamma=self.pars.gamma, leak=self.pars.leak, 
            init=init,
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]

        return X0

    def fit(self, train_data, method, **kwargs):
        fit = method.fit(model=self, train_data=train_data, **kwargs)
        return fit

    def fitMultistep(self, train_data, method, steps=1, **kwargs):
        fit = method.fitMultistep(model=self, train_data=train_data, steps=steps, **kwargs)
        return fit

    def fitDirectMultistep(self, train_data, method, steps=1, **kwargs):
        fit = method.fitDirectMultistep(model=self, train_data=train_data, steps=steps, **kwargs)
        return fit

    def print(self):
        table = self.pars.table()
        table.title = "ESN"
        console.print()