"""
Main ESN model object class.
"""

from typing import Union

import numpy as np
# import pandas as pd
# from numba import njit

from libesn.console import console
from libesn.datautils import * 
from libesn.validation import *
from libesn.esn_states import states
from libesn.matgen import matrixGenerator

from rich.table import Table
from rich import box

class stateParameters:
    r""" 
    Collection of state parameters for an ESN model.
    A `stateParameters` instance contains information regarding: the state map $\sigma$;
    ESN parameter matrices $A$, $C$ and $\zeta$; ESN hyperparameters $\rho$, $\gamma$ and
    leak rate `leak`. 
    """

    size: int
    """Number of ESN models in the collection, fixed to 1."""
    N: int
    """State-space dimension."""
    K: int
    """Input dimension."""
    smap: np.ufunc 
    """State map."""    
    A: np.ndarray 
    r"""Reservoir (connectivity) matrix, must be of shape $N \times N$."""
    C: np.ndarray 
    r"""Reservoir input mask, must be of shape $N \times K$."""
    zeta: np.ndarray
    r"""Reservoir input shift, must be of shape $N \times 1$ or a 1D vector."""
    rho: Union[int, float, np.ndarray]
    r"""Reservoir (connectivity) matrix spectral radius, $\rho \in [0, \infty)$."""  
    gamma: Union[int, float, np.ndarray]
    r"""Reservoir input scaling, $\gamma \in (0, \infty)$."""
    leak: Union[int, float, np.ndarray]
    r"""Reservoir leak rate, $\alpha \in [0, 1]$."""

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
        """ Initialize the `stateParameters` instance. """

        A_shape = A.shape
        C_shape = C.shape

        # preliminary shape checks
        assert A_shape[0] == A_shape[1], "A matrix is not square"
        assert C_shape[0] == A_shape[0], "A and C matrices are not compatible"

        # single-dimensional ESN
        self.size = 1
        self.N = A_shape[0] # state-space dimension
        self.K = C_shape[1] # input dimension

        # prehemptive allocations
        if zeta is None:
            zeta = np.zeros(self.N)
        else:
            if len(zeta.shape) == 1:
                assert zeta.shape == (self.N,), "A and zeta are not compatible"
            elif len(zeta.shape) == 2:
                zeta = np.squeeze(zeta)
                assert zeta.shape == (self.N,), "A and zeta are not compatible"
            else:
                raise ValueError("zeta must be a 1D or 2D vector")

        # coefficient checks
        assert rho >= 0, "rho is not a nonnegative scalar"
        assert leak >= 0 and leak <= 1, "leak is not a scalar in interval [0,1]"

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
        """ Construct a `rich` table of the contents of the `stateParameters` object. """

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
        """ Print `rich` table of contents. """

        console.print(self.table())


class ESN:
    pars: stateParameters
    """State parameters object, either passed as `pars` argument or constructed from other arguments."""
    N: int
    """Short-hand: state-space dimension (automatically extracted from `pars`)."""
    K: int
    """Short-hand: input dimension (automatically extracted from `pars`)."""

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
        """
        Generate ESN parameters from dictionaries describing reservoir matrix shapes,
        normalizations and distributions as well as hyperparameters.

        + `A` and `C` must be dictionaries with keys `shape`, `dist`, `sparsity`, `normalize` and `options`.
        + `zeta` is optional, can be either `None` *or* a dictionary with the same keys 
        as `A` and `C`.
        + `rho`, `gamma` and `leak` are optional. Defaults are 0, 1 and 0 respectively.
        """
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
        """
        Collect ESN states from input data. Optional arguments include:

        + `init` : initial state, defaults to `None` (i.e. zero initial state conditions).
        + `burnin` : number of burn-in periods, defaults to 0 (i.e. no states are discarded).
        """
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
        """
        Fit the ESN model to training data using a specified fitting method. 
        
        This is a short-hand for calling `method.fit(model=self, train_data=train_data, **kwargs)`:

        + `train_data` : training data, can be a `pandas` DataFrame or a `numpy` array.
        + `method` : fitting method, must be an instance of `libesn.esn_fit.esnFitMethod` 
            implementing a `fit()` method.
        """
        fit = method.fit(model=self, train_data=train_data, **kwargs)
        return fit

    def fitMultistep(self, train_data, method, steps=1, **kwargs):
        r"""
        Fit the ESN model to training data using a specified *multistep* fitting method.

        `fitMultistep()` is used to train the ESN to predict multiple steps ahead in **autonomous** mode,
        meaning that for all prediction steps $> 1$ the ESN state equation is run forward in time
        and the output is fed back as input. This requires estimating auxiliary coefficients 
        $W_{\textnormal{ar}}$, following equation
        
        $$
        \begin{aligned}
            X_t &= \alpha X_{t-1} + (1 - \alpha) \sigma(\rho A X_{t-1} + \gamma C Z_t + \zeta) \\\
            Z_{t+1} &= W_{\textnormal{ar}}' X_t + \eta_{t+1}
        \end{aligned}
        $$

        so that the ESN can be run autonomously even when the output data is not a shift of the input data.

        This is a short-hand for calling `method.fit(model=self, train_data=train_data, **kwargs)`:

        + `train_data` : training data, can be a `pandas` DataFrame or a `numpy` array.
        + `method` : fitting method, must be an instance of `libesn.esn_fit.esnFitMethod` 
            implementing a `fitMultistep()` method.
        + `steps` : number of steps ahead to predict when fitting.
        """
        fit = method.fitMultistep(model=self, train_data=train_data, steps=steps, **kwargs)
        return fit

    def fitDirectMultistep(self, train_data, method, steps=1, **kwargs):
        r"""
        Fit the ESN model to training data using a specified *multistep* fitting method. 

        `fitDirectMultistep()` is used to train the ESN to predict multiple steps ahead in **direct** mode,
        meaning that for all prediction steps $\geq 1$ a step (prediction horizon) specific coefficient matrix
        is estimated, following equation
        
        $$
        \begin{aligned}
            X_t &= \alpha X_{t-1} + (1 - \alpha) \sigma(\rho A X_{t-1} + \gamma C Z_t + \zeta) \\\
            Z_{t+s} &= W_{s}' X_t + \eta_{t+s}
        \end{aligned}
        $$

        where $s = 1, \ldots, \textnormal{steps}$ and $\\{ W_{s} \\}_{s=1}^{\textnormal{steps}}$ are the
        step-specific coefficient matrices.

        This is a short-hand for calling `method.fit(model=self, train_data=train_data, **kwargs)`:

        + `train_data` : training data, can be a `pandas` DataFrame or a `numpy` array.
        + `method` : fitting method, must be an instance of `libesn.esn_fit.esnFitMethod` 
            implementing a `fitMultistep()` method.
        + `steps` : number of steps ahead to predict when fitting.
        """
        fit = method.fitDirectMultistep(model=self, train_data=train_data, steps=steps, **kwargs)
        return fit

    def print(self):
        """Print the contents of the `pars` object of the ESN model."""
        table = self.pars.table()
        table.title = "ESN"
        console.print()