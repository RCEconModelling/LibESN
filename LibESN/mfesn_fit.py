#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

from typing import Union

import pandas as pd
import numpy as np
# from numpy import linalg as npLA
from numba import njit
from scipy.optimize import minimize as scipy_minimize
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.factory import get_termination

from LibESN.base_utils import * 
from LibESN.base_datetime import *
from LibESN.esn_states import iter_state
from LibESN.esn_fit import ridge, ridge_penalty_check
from LibESN.mfesn import MFESN
from LibESN.mfesn_states import mf_dt_states, mf_dt_states_to_matrix

# NOTE: fit method prototype
#class MFESNfitMethod:
#    def __init__(pars):
#       ...
#
#    def fit(model, input, target, burnin=0):
#       ...
#       return({
#            "states": model_states,
#            "W": W,
#        })
#

class mfRidgeFit:
    def __init__(self, Lambda) -> None:
        # parameter check
        if type(Lambda) in [tuple, list]:
            L = []
            for i, L0 in enumerate(Lambda):
                if type(L0) in [tuple, list]:
                    l = []
                    for j, l0 in enumerate(L0):
                        try:
                            l.append(ridge_penalty_check(l0))
                        except:
                            raise ValueError(f"Lambda failed check at index [{i}][{j}]")
                    L.append(l)
                else:
                    try:
                        L1 = ridge_penalty_check(L0)
                    except:
                        raise ValueError(f"Lambda failed check at index {i}")
                    L.append(L1)
        else:
            try:
                L = ridge_penalty_check(Lambda)
            except:
                raise ValueError("Lambda failed check")
            
        self.Lambda = L

    def fit(
        self, 
        mfmodel: MFESN, 
        train_data: Union[tuple, list], 
        full=True, 
        **kwargs
    ) -> dict:
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']
        for bn in burnin:
            assert bn >= 0, "burnin must be a non-negative integer"
        
        Lambda_models_ar = np.zeros((mfmodel.size)) if not 'Lambda_models_ar' in kwargs.keys() else kwargs['Lambda_models_ar']
        
        # states
        mf_model_states = mf_dt_states(
            input=Z, 
            mfmodel=mfmodel, 
            dates=Z_dates, 
            init=init, 
            burnin=burnin,
        )

        # slice and stack model states plus slicing dates
        mf_X0, mf_X0_dates = mf_dt_states_to_matrix(
            states=mf_model_states, 
            ref_dates=Y_dates, 
            states_join=mfmodel.states_join, 
            states_lags=mfmodel.states_lags,
        )

        # fit target
        W = mfridge(
            X=mf_X0[0:-1,:], 
            Y=Y[1:,:],
            Lambda=self.Lambda,
            N=mfmodel.N,
        )

        # optional
        # fit individual autoregressive models for mfmodel components 
        # (needed e.g. for multi-step forecasting)
        if full:
            W_models_ar = []
            for j, Xj in enumerate(mf_model_states):
                # reduce to ndarray
                Xj_ = Xj.iloc[0:-1,:].to_numpy()
                Yj_ = Z[j][1:,:]
                # fit
                Wj_ = ridge(X=Xj_, Y=Yj_, Lambda=Lambda_models_ar[j])
                W_models_ar.append(Wj_)
        else:
            W_models_ar = None

        # outputs
        T = mf_X0.shape[0]-1
        mf_X = np.hstack((np.ones((T, 1)), mf_X0[0:-1,:]))
        mf_Y = pd.DataFrame(data=Y[1:,:], index=Y_dates[1:])
        mf_Y_fit = pd.DataFrame(data=(mf_X @ W), index=Y_dates[1:])
        residuals = mf_Y - mf_Y_fit
        MSE = np.mean(np.square(residuals.to_numpy()))

        return({
            'model': "MFESN",
            'method': "mfRidgeFit.fit",
            'W': W,
            'states': mf_model_states,
            'mf_X': mf_X0,
            'mf_X_dates': mf_X0_dates,
            'mf_Y': mf_Y,
            'mf_Y_fit': mf_Y_fit,
            'residuals': residuals,
            'MSE': MSE,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'full': full,
            'W_models_ar': W_models_ar,
            'Lambda_models_ar': Lambda_models_ar,
            'init': init,
            'burnin': burnin,
        })

    def fitMultistep(
        self, 
        mfmodel: MFESN, 
        train_data: Union[tuple, list], 
        steps=1,  
        **kwargs
    ) -> dict:
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']
        for bn in burnin:
            assert bn >= 0, "burnin must be a non-negative integer"

        Lambda_models_ar = np.zeros((mfmodel.size)) if not 'Lambda_models_ar' in kwargs.keys() else kwargs['Lambda_models_ar']
        
        # handle steps
        if type(steps) is int:
            steps = range(1, steps+1)
        elif type(steps) in [tuple, list]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple/list of integers"
        else:
            raise ValueError("steps must be an integer or a tuple/list of integers")

        # states
        mf_model_states = mf_dt_states(
            input=Z, 
            mfmodel=mfmodel, 
            dates=Z_dates, 
            init=init, 
            burnin=burnin,
        )

        # TODO: currently, due to complexity, only state alignment option 'align'
        #       is implemented below
        assert mfmodel.states_join == "align", "fitMultistep() currently only supports states_join='align'"

        # slice and stack model states plus slicing dates
        mf_X0, mf_X0_dates = mf_dt_states_to_matrix(
            states=mf_model_states, 
            ref_dates=Y_dates, 
            states_join=mfmodel.states_join, 
            states_lags=mfmodel.states_lags,
        )

        # autoregressive models for mfmodel components 
        W_models_ar = []
        for j, Xj in enumerate(mf_model_states):
            X0_j = Xj.to_numpy()
            # fit
            W_j = ridge(
                X=X0_j[0:-1,:], 
                Y=Z[j][1:,:], 
                Lambda=Lambda_models_ar[j],
            )
            W_models_ar.append(W_j)

        fits = []
        for i, s in enumerate(steps):
            # construct low-frequency index
            ref_dates_s = Y_dates[0:(-s)]

            # states
            if s == 1:
                mf_X0_g = mf_X0[0:-1,:]
            else:
                mf_X0_g = np.full((len(ref_dates_s), mfmodel.M), np.nan)
                p = 0
                for j, x_j in enumerate(mf_model_states):
                    # iteration matrix
                    D_j = mfmodel.models[j].pars.C @ W_models_ar[j].T

                    ks = 0
                    for i, d in enumerate(ref_dates_s):
                        cpd, ks = closest_past_date(x_j.index, d, cutoff=ks)
                        # target date
                        tgd = Y_dates[i]
                        # iterate states
                        l_j = len(x_j.loc[cpd:tgd,])
                        state_j = np.squeeze(x_j.loc[cpd,].to_numpy())
                        Xg_j = iter_state(
                            state=state_j,
                            length=l_j,
                            map=mfmodel.models[j].pars.smap,
                            A=mfmodel.models[j].pars.A, 
                            D=D_j, 
                            zeta=mfmodel.models[j].pars.zeta, 
                            rho=mfmodel.models[j].pars.rho, 
                            gamma=mfmodel.models[j].pars.gamma, 
                            leak=mfmodel.models[j].pars.leak,
                        )
                        mf_X0_g[i,p:(p+mfmodel.N[j])] = np.squeeze(Xg_j[-1,])
                    p += mfmodel.N[j]

            # step lambda
            Lambda_s = self.Lambda[i] if type(self.Lambda) in [tuple, list] else self.Lambda
            
            # fit target
            W_s = mfridge(
                X=mf_X0_g, 
                Y=Y[s:,:], 
                Lambda=Lambda_s,
                N=mfmodel.N,
            )

            # outputs
            T = mf_X0_g.shape[0]
            mf_X = np.hstack((np.ones((T, 1)), mf_X0_g))
            mf_Y = pd.DataFrame(data=Y[s:,:], index=Y_dates[s:])
            mf_Y_fit = pd.DataFrame(data=(mf_X @ W_s), index=Y_dates[s:])
            residuals = mf_Y - mf_Y_fit
            MSE = np.mean(np.square(residuals.to_numpy()))

            # outputs
            fits.append({
                's': s,
                'W': W_s,
                'mf_X': mf_X,
                'mf_Y': mf_Y,
                'mf_Y_fit': mf_Y_fit,
                'residuals': residuals,
                'MSE': MSE,
            })

        return({
            'model': "MFESN",
            'method': "mfRidgeFit.fitMultistep",
            'fits': fits,
            'states': mf_model_states,
            'mf_X': mf_X0,
            'mf_X_dates': mf_X0_dates,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'steps': steps,
            'init': init,
            'burnin': burnin,
        })

    def fitDirectMultistep(
        self, 
        mfmodel: MFESN, 
        train_data: Union[tuple, list], 
        steps=1,  
        **kwargs
    ) -> dict:
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']
        for bn in burnin:
            assert bn >= 0, "burnin must be a non-negative integer"

        # handle steps
        if type(steps) is int:
            steps = range(1, steps+1)
        elif type(steps) in [tuple, list]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple/list of integers"
        else:
            raise ValueError("steps must be an integer or a tuple/list of integers")
        
        # states
        mf_model_states = mf_dt_states(
            input=Z, 
            mfmodel=mfmodel, 
            dates=Z_dates, 
            init=init, 
            burnin=burnin,
        )

        # slice and stack model states plus slicing dates
        mf_X0, mf_X0_dates = mf_dt_states_to_matrix(
            states=mf_model_states, 
            ref_dates=Y_dates, 
            states_join=mfmodel.states_join, 
            states_lags=mfmodel.states_lags,
        )

        # fit model
        fits = []
        for i, s in enumerate(steps):
            # step lambda
            Lambda_s = self.Lambda[i] if type(self.Lambda) in [tuple, list] else self.Lambda

            # fit target
            W_s = mfridge(
                X=mf_X0[0:(-s),:], 
                Y=Y[s:,:],
                Lambda=Lambda_s,
                N=mfmodel.N,
            )

            # outputs
            T = mf_X0.shape[0]-s
            mf_X = np.hstack((np.ones((T, 1)), mf_X0[0:(-s),:]))
            mf_Y = pd.DataFrame(data=Y[s:,:], index=Y_dates[s:])
            mf_Y_fit = pd.DataFrame(data=(mf_X @ W_s), index=Y_dates[s:])
            residuals = mf_Y - mf_Y_fit
            MSE = np.mean(np.square(residuals.to_numpy()))

            # outputs
            fits.append({
                's': s,
                'W': W_s,
                'mf_X': mf_X,
                'mf_Y': mf_Y,
                'mf_Y_fit': mf_Y_fit,
                'residuals': residuals,
                'MSE': MSE,
            })

        return({
            'model': "MFESN",
            'method': "mfRidgeFit.fitDirectMultistep",
            'fits': fits,
            'states': mf_model_states,
            'mf_X': mf_X0,
            'mf_X_dates': mf_X0_dates,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'steps': steps,
            'init': init,
            'burnin': burnin,
        })

    def fitDirectHighFreq(
        self, 
        mfmodel: MFESN, 
        train_data: Union[tuple, list], 
        freqratio, 
        steps=0, 
        **kwargs
    ) -> dict:
        """
        Fit MFESN using direct regressions at fixed high-frequency steps over
        each low-frequency interval. This is explicitly designed for nowcasting
        as the targets used are always the closest LF future observations (w.r.t
        current HF time).
        mfmodel: MFESN model.
        train_data: fitting dataset.
        freqratio: fixed integer ratio of HF to LF observations. 
        steps: number of fit steps. Default is s=0 because unline in the low-freq.
               setup, HF regressors can be observed at distance of less than 1
               LF interval from the target c.f. nowcasting.
        """
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # handle freqratio
        assert freqratio >= 1, "freqratio must be a positive integer"

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']
        for bn in burnin:
            assert bn >= 0, "burnin must be a non-negative integer"

        # handle steps
        if type(steps) is int:
            # NOTE: here we start the range at 0, see def of steps above
            steps = range(0, steps+1)
        elif type(steps) in [tuple, list]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple/list of integers"
        else:
            raise ValueError("steps must be an integer or a tuple/list of integers")

        # states
        mf_model_states = mf_dt_states(
            input=Z, 
            mfmodel=mfmodel, 
            dates=Z_dates, 
            init=init, 
            burnin=burnin,
        )

        # find high-frequency date slices
        max_freq_dates = pd_max_freq_dates(Z_dates)
        max_freq_dslices = []
        for k in range(freqratio):
            dslice_k = []
            l = None
            if k == 0:
                # start with closest future dates to Y_dates
                for d in Y_dates[:-1]:
                    cfd, l = closest_future_date(max_freq_dates, d, cutoff=l, strict=True)
                    dslice_k.append(cfd)
                    # adjust cutoff
                    l += 2*freqratio
            else:
                # iterate, closest future date to previous slice
                for d in dslice_0:
                    cfd, l = closest_future_date(max_freq_dates, d, cutoff=l, strict=True)
                    dslice_k.append(cfd)
                    # adjust cutoff
                    l += 2*freqratio
            # append
            dslice_0 = dslice_k
            max_freq_dslices.append(pd.to_datetime(dslice_k))

        # fit model
        fits = []
        for i, s in enumerate(steps):
            # step lambda
            # NOTE: must be able to hande nested lists e.g. [[1,2],]
            #       where the outer level list enumerates over steps,
            #       and the inner level list enumerates over freq. ratio
            Lambda_s = None
            if type(self.Lambda) in [tuple, list]:
                if len(self.Lambda) == len(steps):
                    Lambda_s = self.Lambda[i]
                elif len(self.Lambda) == 1 and type(self.Lambda[0]) in [tuple, list]:
                    Lambda_s = self.Lambda[0]
            else:
                Lambda_s = self.Lambda

            freq_fits_s = []
            for k in range(freqratio):
                # sub-step lambda
                Lambda_k = Lambda_s[k] if type(Lambda_s) in [tuple, list] else Lambda_s

                # slice and stack model states plus slicing dates
                mf_X0_k, mf_X0_dates_k = mf_dt_states_to_matrix(
                    states=mf_model_states, 
                    ref_dates=max_freq_dslices[k], 
                    states_join=mfmodel.states_join, 
                    states_lags=mfmodel.states_lags,
                )

                # print(f"+ {s} - {k}")

                # fit target
                # NOTE: for regressors, need to index with 0:(-s)
                #       because the 1st step in HF means that we are
                #       targeting the next available LF observation,
                #       which is less than 1 LF interval away c.f. nowcasting
                mf_X0_k_s = mf_X0_k if s == 0 else mf_X0_k[0:(-s),:]
                Y_k_s = Y[(1+s):,:]

                W_k = mfridge(
                    X=mf_X0_k_s, 
                    Y=Y_k_s,
                    Lambda=Lambda_k,
                    N=mfmodel.N,
                )

                # outputs
                T = mf_X0_k_s.shape[0]
                mf_X_k = np.hstack((np.ones((T, 1)), mf_X0_k_s))
                mf_Y_k = pd.DataFrame(data=Y[(1+s):,:], index=Y_dates[(1+s):])
                mf_Y_fit = pd.DataFrame(data=(mf_X_k @ W_k), index=Y_dates[(1+s):])
                residuals = mf_Y_k - mf_Y_fit
                MSE = np.mean(np.square(residuals.to_numpy()))

                # outputs
                freq_fits_s.append({
                    'k': k,
                    'hf_lag': (freqratio-k-1),
                    'W': W_k,
                    'mf_X': mf_X_k,
                    'mf_X_dates': mf_X0_dates_k,
                    'mf_Y': mf_Y_k,
                    'mf_Y_fit': mf_Y_fit,
                    'residuals': residuals,
                    'MSE': MSE,
                    # additional info
                    'Lambda': Lambda_k,
                })

            # steps output
            fits.append({
                's': s,
                'freq_fits': freq_fits_s
            })

        return({
            'model': "MFESN",
            'method': "mfRidgeFit.fitDirectHighFreq",
            'fits': fits,
            'states': mf_model_states,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'steps': steps,
            'freqratio': freqratio,
            'max_freq_slices': max_freq_dslices,
            'init': init,
            'burnin': burnin,
        })


class mfRidgeCV:
    def __init__(self) -> None:
        pass

    def cv(
        self,
        mfmodel: MFESN, 
        train_data: Union[tuple, list], 
        step=1,
        **kwargs
    ) -> dict:
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']
        for bn in burnin:
            assert bn >= 0, "burnin must be a non-negative integer"
        
        test_size = 1 if not 'test_size' in kwargs.keys() else kwargs['test_size']
        min_train_size = 1 if not 'min_train_size' in kwargs.keys() else kwargs['min_train_size']
        max_train_size = None if not 'max_train_size' in kwargs.keys() else kwargs['max_train_size']
        overlap = False if not 'overlap' in kwargs.keys() else kwargs['overlap']
        
        optim = 'scipy' if not 'optim' in kwargs.keys() else kwargs['optim']
        
        verbose = False if not 'verbose' in kwargs.keys() else kwargs['verbose']

        # handle steps
        if not type(step) is int:
            raise ValueError("steps must be an integer or a tuple, list or range of integers")
        assert step >= 0, "step must be a non-negative integer"
        
        # states
        mf_model_states = mf_dt_states(
            input=Z, 
            mfmodel=mfmodel, 
            dates=Z_dates, 
            init=init, 
            burnin=burnin,
        )

        # slice and stack model states plus slicing dates
        mf_X0, _ = mf_dt_states_to_matrix(
            states=mf_model_states, 
            ref_dates=Y_dates, 
            states_join=mfmodel.states_join, 
            states_lags=mfmodel.states_lags,
        )

        # slice step
        mf_X0_cv = mf_X0[0:(-step),:]
        mf_Y_cv = Y[step:,:]

        # cv splits
        tscv = ShiftTimeSeriesSplit(
            length=mf_X0_cv.shape[0],
            test_size=test_size,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            overlap=overlap,
        )
        iter_tscv = iter(tscv)

        # objective function
        Nmods = mfmodel.N

        def cv_obj(cv_lambda):
            # Lambda = np.exp(cv_lambda)
            Lambda = cv_lambda

            cv_RSS = 0
            for train_idx, test_idx in iter_tscv:
                W = jit_kron_mfridge(
                    X=mf_X0_cv[train_idx,:], 
                    Y=mf_Y_cv[train_idx,:], 
                    Lambda=Lambda,
                    N=Nmods,
                )
                residuals = mf_Y_cv[test_idx] - np.hstack([np.ones((len(test_idx), 1)), mf_X0_cv[test_idx,:]]) @ W
                cv_RSS += np.sum(np.square(residuals))

            return cv_RSS

        # print(cv_obj([0.1, 0.1]))

        # optimization
        if optim == "scipy":
            lb = 1e-2*np.ones(len(Nmods))
            ub = 1e9*np.ones(len(Nmods))

            result = scipy_minimize(
                fun=cv_obj,
                x0=1e-2*np.ones(len(Nmods)),
                bounds=tuple(zip(lb, ub)),
                method='L-BFGS-B',
                options={'disp': verbose},
            )

            # output
            cvLambda = result.x

        elif optim == "pymoo":
            problem = FunctionalProblem(
                len(Nmods),
                cv_obj,
                x0=1e-2*np.ones(len(Nmods)),
                xl=1e-9*np.ones(len(Nmods)),
                xu=1e5*np.ones(len(Nmods)),
            )

            result = minimize(
                problem, 
                PatternSearch(n_sample_points=50), 
                #get_termination("n_eval", 500),
                #get_termination("time", "00:01:00"),
                get_termination("ftol", 1e-8),
                verbose=verbose, 
                seed=1203477
            )

            # output
            cvLambda = result.X

        else:
            raise ValueError("Unknown optimization method selected")

        return({
            'model': "MFESN",
            'method': "mfRidgeFit.cv",
            'cvLambda': cvLambda,
            'result': result,
            # additional info
            'optim': optim,
            'init': init,
            'burnin': burnin,
        })

    def cvHighFreq(
        self,
        mfmodel: MFESN, 
        train_data: Union[tuple, list], 
        freqratio,
        steps=0,
        **kwargs
    ) -> dict:
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # handle freqratio
        assert freqratio >= 1, "freqratio must be a positive integer"

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = [0 for _ in range(mfmodel.size)] if not 'burnin' in kwargs.keys() else kwargs['burnin']
        for bn in burnin:
            assert bn >= 0, "burnin must be a non-negative integer"
        
        test_size = 1 if not 'test_size' in kwargs.keys() else kwargs['test_size']
        min_train_size = 1 if not 'min_train_size' in kwargs.keys() else kwargs['min_train_size']
        max_train_size = None if not 'max_train_size' in kwargs.keys() else kwargs['max_train_size']
        overlap = False if not 'overlap' in kwargs.keys() else kwargs['overlap']
        
        optim = 'scipy' if not 'optim' in kwargs.keys() else kwargs['optim']
        
        verbose = False if not 'verbose' in kwargs.keys() else kwargs['verbose']

        # handle steps
        if type(steps) is int:
            # NOTE: here we start the range at 0, see def of steps above
            steps = range(0, steps+1)
        elif type(steps) in [tuple, list]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple/list of integers"
        else:
            raise ValueError("steps must be an integer or a tuple/list of integers")

        # states
        mf_model_states = mf_dt_states(
            input=Z, 
            mfmodel=mfmodel, 
            dates=Z_dates, 
            init=init, 
            burnin=burnin,
        )

        # find high-frequency date slices
        max_freq_dates = pd_max_freq_dates(Z_dates)
        max_freq_dslices = []
        for k in range(freqratio):
            dslice_k = []
            l = None
            if k == 0:
                # start with closest future dates to Y_dates
                for d in Y_dates[:-1]:
                    cfd, l = closest_future_date(max_freq_dates, d, cutoff=l, strict=True)
                    dslice_k.append(cfd)
                    # adjust cutoff
                    l += 2*freqratio
            else:
                # iterate, closest future date to previous slice
                for d in dslice_0:
                    cfd, l = closest_future_date(max_freq_dates, d, cutoff=l, strict=True)
                    dslice_k.append(cfd)
                    # adjust cutoff
                    l += 2*freqratio
            # append
            dslice_0 = dslice_k
            max_freq_dslices.append(pd.to_datetime(dslice_k))


        Nmods = mfmodel.N

        cvLambda = []
        result = []
        for s in steps:
            cvLambda_s = []
            result_s = []

            for k in range(freqratio):
                # slice and stack model states plus slicing dates
                mf_X0_k_cv, _ = mf_dt_states_to_matrix(
                    states=mf_model_states, 
                    ref_dates=max_freq_dslices[k], 
                    states_join=mfmodel.states_join, 
                    states_lags=mfmodel.states_lags,
                )

                mf_X0_k_cv = mf_X0_k_cv if s == 0 else mf_X0_k_cv[0:(-s),:]
                mf_Y_cv = Y[(1+s):,:]

                # cv splits
                tscv = ShiftTimeSeriesSplit(
                    length=mf_X0_k_cv.shape[0],
                    test_size=test_size,
                    min_train_size=min_train_size,
                    max_train_size=max_train_size,
                    overlap=overlap,
                )
                iter_tscv = iter(tscv)

                # objective function
                def cv_obj_k(cv_lambda):
                    # Lambda = np.exp(cv_lambda)
                    Lambda = cv_lambda

                    cv_RSS = 0
                    for train_idx, test_idx in iter_tscv:
                        W = jit_kron_mfridge(
                            X=mf_X0_k_cv[train_idx,:], 
                            Y=mf_Y_cv[train_idx,:], 
                            Lambda=Lambda,
                            N=Nmods,
                        )
                        residuals = mf_Y_cv[test_idx] - np.hstack([np.ones((len(test_idx), 1)), mf_X0_k_cv[test_idx,:]]) @ W
                        cv_RSS += np.sum(np.square(residuals))

                    return cv_RSS

                # print(cv_obj([0.1, 0.1]))

                if verbose: print(f"[+] CV HF step: {s}.{k+1} of {max(steps)}.{freqratio}")

                # optimization
                if optim == "scipy":
                    lb = 1e-2*np.ones(len(Nmods))
                    ub = 1e9*np.ones(len(Nmods))

                    result_k = scipy_minimize(
                        fun=cv_obj_k,
                        x0=1e-2*np.ones(len(Nmods)),
                        bounds=tuple(zip(lb, ub)),
                        method='L-BFGS-B',
                        # options={'disp': 10 if verbose else -1},
                    )

                    # output
                    cvLambda_s.append(result_k.x)
                    result_s.append(result_k)

                elif optim == "pymoo":
                    problem_k = FunctionalProblem(
                        len(Nmods),
                        cv_obj_k,
                        x0=1e-2*np.ones(len(Nmods)),
                        xl=1e-9*np.ones(len(Nmods)),
                        xu=1e5*np.ones(len(Nmods)),
                    )

                    result_k = minimize(
                        problem_k, 
                        PatternSearch(n_sample_points=50), 
                        #get_termination("n_eval", 500),
                        #get_termination("time", "00:01:00"),
                        get_termination("ftol", 1e-8),
                        verbose=verbose, 
                        seed=1203477
                    )

                    # output
                    cvLambda_s.append(result_k.X)
                    result_s.append(result_k)

                else:
                    raise ValueError("Unknown optimization method selected")

            # steps output
            cvLambda.append({
                's': s,
                'L': cvLambda_s,
            })
            result.append({
                's': s,
                'result': result_s, 
            })

        return({
            'model': "MFESN",
            'method': "mfRidgeFit.cvHighFreq",
            'cvLambda': cvLambda,
            'result': result,
            # additional info
            'init': init,
            'burnin': burnin,
        })



def mfridge(X: np.ndarray, Y: np.ndarray, Lambda=None, N=None):
    """
    Ridge regression for a multi-frequency setup i.e. if N is a 
    list of dimensions, and Lambda and N have the same length, then
    the final penalty is given by diag(kron(Lambda, rep(1, N)))
    """
    # sanity checks
    Tx, Kx = X.shape
    Ty, _  = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    if Lambda is None:
        print("[-!-] mfridge() called with Lambda set to None!")
        L = np.array(0)
    else:
        L = np.array(Lambda).squeeze()

    if L.shape == ():
        # Lambda is a scalar 
        if L < 1e-12:
            V = np.hstack((np.ones((Tx, 1)), X))
            res = np.linalg.lstsq(V, Y, rcond=None)
            W = res[0]
        else:
            W = np.linalg.solve(((X.T @ X / Tx) + L * np.eye(Kx)), (X.T @ Y / Ty))
            a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
            W = np.vstack((a, W))
    elif not N is None:
        N = np.array(N).astype(int)
        assert len(Lambda) == len(N), "Lambda and N should have the same length"
        assert np.min(N) > 0, "N should contain only positive integers"
        assert np.sum(N) == Kx, "Elements of N should sum up to the state dimension of X"
        l = np.concatenate([L[i] * np.ones(N[i]) for i in range(len(N))])
        W = np.linalg.solve(((X.T @ X / Tx) + np.diag(l)), (X.T @ Y / Ty))
        a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
        W = np.vstack((a, W))
    else:
        raise ValueError("Could not run ridge regression")

    return W

# @njit
def jit_kron_mfridge(X: np.ndarray, Y: np.ndarray, Lambda, N):
    Tx = X.shape[0]
    Ty = Y.shape[0]
    L = np.array(Lambda)
    N = np.array(N).astype(int)
    l = np.concatenate([L[i] * np.ones(N[i]) for i in range(len(N))])
    W = np.linalg.solve(((X.T @ X / Tx) + np.diag(l)), (X.T @ Y / Ty))
    a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
    W = np.vstack((a, W))
    
    return W