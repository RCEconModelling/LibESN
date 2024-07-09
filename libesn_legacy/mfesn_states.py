#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

#from typing import Union

import pandas as pd
import numpy as np
from numpy import linalg as npLA

#from numba import njit

from libesn_legacy.base_functions import *
from libesn_legacy.base_utils import *
from libesn_legacy.base_datetime import closest_past_date
from libesn_legacy.esn_states import states, iter_state

# Base
def mf_dt_states(input, mfmodel, dates=None, **kwargs):
    """
    Take multi-frequency inputs and a MFESN model and collect states
    with optional included dates (Pandas dataframes) as a list
    input: tuple or list of input dataframes
    mfmodel: MFESN model
    dates: optional, add datetime index to states
    """
    # assert isinstance(mfmodel, MFESN), "model is not a MFESN model"

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']
    burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']

    # ensure init is valid 
    if init is None:
        init = [np.zeros(n) for n in mfmodel.N]

    # States
    X = []
    for j, Zj in enumerate(input):
        # Xj = mfmodel.models[j].states(
        #     input=Zj, 
        #     init=init[j], 
        #     burnin=burnin[j],
        # )
        # collect states
        states_j = states(
            Zj, 
            map=mfmodel.models[j].pars.smap, 
            A=mfmodel.models[j].pars.A, 
            C=mfmodel.models[j].pars.C, 
            zeta=mfmodel.models[j].pars.zeta, 
            rho=mfmodel.models[j].pars.rho, 
            gamma=mfmodel.models[j].pars.gamma, 
            leak=mfmodel.models[j].pars.leak, 
            init=init[j],
        )
        # add dates to states
        states_j = pd.DataFrame(
            states_j, 
            index=dates[j][burnin[j]:],
        )
        X.append(states_j)

    return X

def mf_dt_states_to_matrix(states, ref_dates, states_join, states_lags=None):
    """
    Reduce a list of multi-frequency states to a single matrix (Numpy)
    for the purpose of e.g. fitting the model.
    states: tuple or list of states, dated
    ref_dates: reference dates for state stacking
    states_join: stacking method
    states_lags: optional, lags of state to incluse when stacking
    """
    if (states_join == "align") and (not states_lags is None):
        print(f"[+] LibESN.mf_dt_states_to_matrix() - - - - - - - - - - -")
        print(f" !  State joining is 'align', lags in states_lags ignored")

    # multifrequency state matrix
    mf_X = None

    N = [Xj.shape[1] for Xj in states]
    M = sum(N)

    if states_join == "align":
        mf_X = np.full((len(ref_dates), M), np.nan)
        mf_X_dates = []

        # High-frequency states are aligned, i.e. for each frequency
        # only the closest past / contemporary state to the low-freuency 
        # target is used as regressor.
        p = 0
        for j, Xj in enumerate(states):
            kt = 0
            cpdl_j = []
            for t, lf_date_t in enumerate(ref_dates):
                cpd, kt = closest_past_date(Xj.index, lf_date_t, cutoff=kt)
                mf_X[t,p:(p + N[j])] = np.squeeze(Xj.loc[cpd])
                # save slice date
                cpdl_j.append(cpd)
            p += N[j]
            # save alignment dates
            mf_X_dates.append(pd.to_datetime(cpdl_j))

    elif states_join == "lagstack":
        mf_X = np.full((len(ref_dates), int(sum(N * (1 + states_lags)))), np.nan)
        mf_X_dates = []

        # High-frequency states are stacked with lags, i.e. for each frequency
        # the closest past / contemporary state to the low-freuency 
        # target + lagged states are used together as regressors.
        p = 0
        for j, Xj in enumerate(states):
            kt = 0
            cpdl_j = []
            for t, lf_date_t in enumerate(ref_dates):
                cpd, kt = closest_past_date(Xj.index, lf_date_t, cutoff=kt)
                mf_X[t,p:(p + N[j])] = np.squeeze(Xj.loc[cpd])
                # save slice date (only first)
                cpdl_j.append(cpd)
                # lags
                q = p + N[j]
                for l in range(states_lags[j]):
                    cpd_lag = Xj.index[kt-l-1]
                    mf_X[t,q:(q + N[j])] = np.squeeze(Xj.loc[cpd_lag])
                    q += N[j]
            p += q
            # save alignment dates
            mf_X_dates.append(pd.to_datetime(cpdl_j))

    else:
        raise ValueError("Multifrequency state joining method not defined") 

    return mf_X, mf_X_dates

def mf_states(input, mfmodel, ref_dates, states_join, states_lags=None, **kwargs):
    """
    Shorthand for chaining mf_dt_states() and mf_dt_states_to_matrix()
    with dates extraction directly from inputs.
    """
    Z, Z_dates = pd_data_prep(input)

    X = mf_dt_states(Z, mfmodel, dates=Z_dates, **kwargs)
    X = mf_dt_states_to_matrix(X, ref_dates, states_join, states_lags)

    return X
