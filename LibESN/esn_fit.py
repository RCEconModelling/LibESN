#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

from typing import Union

import numpy as np
from numpy import linalg as npLA
# from numba import njit
from cvxopt import matrix as cvx_matrix
from cvxopt import solvers as cvx_solvers
#from cvxopt import matrix, solvers
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.factory import get_termination

from LibESN.base_utils import * 
from LibESN.esn import ESN
from LibESN.esn_states import states, generate

# non-negativity check function
def ridge_penalty_check(L) -> np.ndarray:
    L = np.squeeze(np.copy(L))

    if L.shape == ():
        assert L >= 0, "L must be a nonnegative scalar"
    elif len(L.shape) == 1:
        assert np.min(L) >= 0, "L must be a nonnegative vector"
    else:
        assert np.min(npLA.eigvals(L)) >= 0, "L must be nonnegative definite"

    return L

# PROTOTYPE
class esnFitMethod:
    def fit(self) -> None:
        pass

    def fitMultistep(self) -> None:
        pass

    def fitDirectMultistep(self) -> None:
        pass

# RIDGE
class ridgeFit(esnFitMethod):
    def __init__(self, Lambda) -> None:
        # parameter check
        if type(Lambda) in [tuple, list]:
            L = []
            for i, L0 in enumerate(Lambda):
                try:
                    L.append(ridge_penalty_check(L0))
                except:
                    raise ValueError(f"Lambda failed check at index {i}")
        else:
            try:
                L = ridge_penalty_check(Lambda)
            except:
                raise ValueError("Lambda failed check")
            
        self.Lambda = L
        
    def fit(
        self, 
        model: ESN, 
        train_data: Union[tuple, list], 
        step=1,
        **kwargs
    ) -> dict:
        # unwrap
        inputs, targets = train_data

        # prepare data
        Z, Z_dates = pd_data_prep(inputs)
        Y, Y_dates = pd_data_prep(targets)

        # check penalty
        assert not type(self.Lambda) in [tuple, list], "Lambda can not be a tuple or list in method fit()"

        # handle kwargs
        init = None if not 'init' in kwargs.keys() else kwargs['init']
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']
        assert burnin >= 0, "burnin must be a non-negative integer"

        # handle steps
        if not type(step) is int:
            raise ValueError("steps must be an integer or a tuple, list or range of integers")
        assert step >= 0, "step must be a non-negative integer"

        # collect states
        model_states = states(
            Z, 
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            init=init
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]
        Y = Y[burnin:,]

        # fit
        W = ridge(X=X0[0:(-step),:], Y=Y[step:,:], Lambda=self.Lambda)

        # outputs
        T = X0.shape[0]-step
        X = np.hstack((np.ones((T, 1)), X0[0:(-step),:]))
        Y = pd.DataFrame(data=Y[step:,:], index=Y_dates[step:])
        Y_fit = pd.DataFrame(data=(X @ W), index=Y_dates[step:])
        residuals = Y - Y_fit
        MSE = np.mean(np.square(residuals.to_numpy()))

        return({
            'model': "ESN",
            'method': "ridgeFit",
            'W': W,
            'states': model_states,
            'X': X,
            'Y': Y,
            'Y_fit': Y_fit,
            'residuals': residuals,
            'MSE': MSE,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'init': init,
            'burnin': burnin,
        })

    def fitMultistep(
        self, 
        model: ESN, 
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
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']
        assert burnin >= 0, "burnin must be a non-negative integer"

        Lambda_ar = None if not 'Lambda_ar' in kwargs.keys() else np.array(kwargs['Lambda_ar'])

        # handle steps
        if type(steps) is int:
            steps = range(1, steps+1)
        elif type(steps) in [tuple, list, range]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple, list or range of integers"
        else:
            raise ValueError("steps must be an integer or a tuple, list or range of integers")

        # collect states
        model_states = states(
            Z, 
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            init=init
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]
        Y = Y[burnin:,]

        # first-step fit
        if Lambda_ar is None:
            Lambda_ar = self.Lambda[0] if type(self.Lambda) in [tuple, list] else self.Lambda
        W_ar = ridge(X=X0[0:-1,:], Y=Z[(burnin+1):,:], Lambda=Lambda_ar)

        if int(max(steps)) > 1:
            # generate states 
            Xg = generate(
                states=X0,
                length=int(max(steps)),
                W=W_ar,
                map=model.pars.smap, 
                A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
                rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            )

            # stack states (row-wise)
            X0g = np.dstack([X0[:,:,None], Xg])
        else:
            X0g = X0[:,:,None]

        fits = []
        for i, s in enumerate(steps):
            #X0g_s = X0g[0:(-s),:,i]
            if s == 1:
                X0g_s = X0[0:-1,:]
            else:
                X0g_s = Xg[0:(-s),:,i-1]

            # step lambda
            Lambda_s = self.Lambda[i] if type(self.Lambda) in [tuple, list] else self.Lambda
            
            # fit 
            W_s = ridge(
                X=X0g_s, 
                Y=Y[s:,:], 
                Lambda=Lambda_s
            )

            # outputs
            T = X0g_s.shape[0]
            X_s = np.hstack((np.ones((T, 1)), X0g_s))
            Y_s = pd.DataFrame(data=Y[s:,:], index=Y_dates[s:])
            Y_fit_s = pd.DataFrame(data=(X_s @ W_s), index=Y_dates[s:])
            residuals = Y_s - Y_fit_s
            MSE = np.mean(np.square(residuals.to_numpy()))

            # outputs
            fits.append({
                's': s,
                'W': W_s,
                'X': X_s,
                'Y': Y_s,
                'Y_fit': Y_fit_s,
                'residuals': residuals,
                'MSE': MSE,
            })

        return({
            'model': "ESN",
            'method': "ridgeFit.fitMultistep",
            'fits': fits,
            'states': model_states,
            'X': X0g,
            #'X_dates': X0_dates,
            'W_ar': W_ar,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'steps': steps,
            'init': init,
            'burnin': burnin,
        })

    def fitDirectMultistep(
        self, 
        model: ESN, 
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
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']
        assert burnin >= 0, "burnin must be a non-negative integer"

        # handle steps
        if type(steps) is int:
            steps = range(1, steps+1)
        elif type(steps) in [tuple, list, range]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple, list or range of integers"
        else:
            raise ValueError("steps must be an integer or a tuple, list or range of integers")

        # collect states
        model_states = states(
            Z, 
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            init=init
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]
        Y = Y[burnin:,]

        fits = []
        for i, s in enumerate(steps):
            # step lambda
            Lambda_s = self.Lambda[i] if type(self.Lambda) in [tuple, list] else self.Lambda

            # fit target
            W_s = ridge(
                X=X0[0:(-s),:], 
                Y=Y[s:,:],
                Lambda=Lambda_s
            )

            # outputs
            T = X0.shape[0]-s
            X_s = np.hstack((np.ones((T, 1)), X0[0:(-s),:]))
            Y_s = pd.DataFrame(data=Y[s:,:], index=Y_dates[s:])
            Y_fit_s = pd.DataFrame(data=(X_s @ W_s), index=Y_dates[s:])
            residuals = Y_s - Y_fit_s
            MSE = np.mean(np.square(residuals.to_numpy()))

            # outputs
            fits.append({
                's': s,
                'W': W_s,
                'X': X_s,
                'Y': Y_s,
                'Y_fit': Y_fit_s,
                'residuals': residuals,
                'MSE': MSE,
            })

        return({
            'model': "ESN",
            'method': "ridgeFit.fitDirectMultistep",
            'fits': fits,
            'states': model_states,
            'X': X0,
            #'X_dates': X0_dates,
            # method info
            'Lambda': self.Lambda,
            # additional info
            'steps': steps,
            'init': init,
            'burnin': burnin,
        })
    
def ridge(X: np.ndarray, Y: np.ndarray, Lambda=None):
    # sanity checks
    Tx, Kx = X.shape
    Ty, _  = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    L = np.array(0) if Lambda is None else Lambda

    # Regression matrices
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
    else:
        if len(L.shape) == 1:
            assert len(L) == Kx, f"Lambda is not scalar, vector or 2D matrix with Gram shape ({Kx},{Kx}), found shape {Lambda.shape}"
            L = np.diag(L)
        else:
            assert L.shape == (Kx, Kx), f"Lambda is not scalar, vector or 2D matrix with Gram shape ({Kx},{Kx}), found shape {Lambda.shape}"
        W = np.linalg.solve(((X.T @ X / Tx) + L), (X.T @ Y / Ty))
        a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
        W = np.vstack((a, W))

    return W

# @njit
def jit_ridge(X: np.ndarray, Y: np.ndarray, L: np.ndarray):
    T, K = X.shape
    W = np.linalg.solve(((X.T @ X / T) + L * np.eye(K)), (X.T @ Y / T))
    a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
    W = np.vstack((a, W))

    return W

# @njit
# def jit_tikhonov(X: np.ndarray, Y: np.ndarray, L: np.ndarray):
#     T = X.shape[0]
#     W = np.linalg.solve(((X.T @ X / T) + L), (X.T @ Y / T))
#     a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
#     W = np.vstack((a, W))

#     return W

class ridgeCV:
    def __init__(self) -> None:
        pass

    def cv(
        self,
        model: ESN, 
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
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']
        assert burnin >= 0, "burnin must be a non-negative integer"

        test_size = 1 if not 'test_size' in kwargs.keys() else kwargs['test_size']
        min_train_size = 1 if not 'min_train_size' in kwargs.keys() else kwargs['min_train_size']
        max_train_size = None if not 'max_train_size' in kwargs.keys() else kwargs['max_train_size']
        overlap = False if not 'overlap' in kwargs.keys() else kwargs['overlap']

        # handle steps
        if not type(step) is int:
            raise ValueError("steps must be an integer or a tuple, list or range of integers")
        assert step >= 0, "step must be a non-negative integer"

        # collect states
        model_states = states(
            Z, 
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            init=init
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]
        Y = Y[burnin:,]

        # slice step
        X0_cv = X0[0:(-step),:]
        Y_cv = Y[step:,:]

        # cv splits
        tscv = ShiftTimeSeriesSplit(
            length=X0_cv.shape[0],
            test_size=test_size,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            overlap=overlap,
        )
        iter_tscv = iter(tscv)

        # objective function
        def cv_obj(cv_lambda):
            # Lambda = np.exp(cv_lambda)
            Lambda = cv_lambda

            cv_RSS = 0
            for train_idx, test_idx in iter_tscv:
                W = jit_ridge(X=X0_cv[train_idx,:], Y=Y_cv[train_idx,:], L=Lambda)
                residuals = Y_cv[test_idx] - np.hstack([np.ones((len(test_idx), 1)), X0_cv[test_idx,:]]) @ W
                cv_RSS += np.sum(np.square(residuals))

            return cv_RSS

        # optimization
        problem = FunctionalProblem(
            1,
            cv_obj,
            x0=np.array([1e-2]),
            xl=np.array([1e-9]),
            xu=np.array([1e5])
        )

        result = minimize(
            problem, 
            PatternSearch(n_sample_points=50), 
            #get_termination("n_eval", 500),
            #get_termination("time", "00:01:00"),
            get_termination("ftol", 1e-8),
            verbose=False, 
            seed=1203477
        )

        # output
        cvLambda = result.X[0]

        return({
            'model': "ESN",
            'method': "ridgeFit.cv",
            'cvLambda': cvLambda,
            'result': result,
            # additional info
            'init': init,
            'burnin': burnin,
        })

    def cvDirectMultistep(
        self,
        model: ESN, 
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
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']
        assert burnin >= 0, "burnin must be a non-negative integer"

        test_size = 1 if not 'test_size' in kwargs.keys() else kwargs['test_size']
        min_train_size = 1 if not 'min_train_size' in kwargs.keys() else kwargs['min_train_size']
        max_train_size = None if not 'max_train_size' in kwargs.keys() else kwargs['max_train_size']
        overlap = False if not 'overlap' in kwargs.keys() else kwargs['overlap']

        # handle steps
        if type(steps) is int:
            steps = range(1, steps+1)
        elif type(steps) in [tuple, list, range]:
            for s in steps:
                assert type(s) is int, "steps must be a tuple, list or range of integers"
        else:
            raise ValueError("steps must be an integer or a tuple, list or range of integers")

        # collect states
        model_states = states(
            Z, 
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            init=init
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]
        Y = Y[burnin:,]

        cv_lambdas = []
        cv_results = []
        for s in steps:
            # slice step
            X0_cv = X0[0:(-s),:]
            Y_cv = Y[s:,:]

            # cv splits
            tscv = ShiftTimeSeriesSplit(
                length=X0_cv.shape[0],
                test_size=test_size,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                overlap=overlap,
            )
            iter_tscv = iter(tscv)

            # objective function
            def cv_obj(cv_lambda):
                # Lambda = np.exp(cv_lambda)
                Lambda = cv_lambda

                cv_RSS = 0
                for train_idx, test_idx in iter_tscv:
                    W = jit_ridge(X=X0_cv[train_idx,:], Y=Y_cv[train_idx,:], L=Lambda)
                    residuals = Y_cv[test_idx] - np.hstack([np.ones((len(test_idx), 1)), X0_cv[test_idx,:]]) @ W
                    cv_RSS += np.sum(np.square(residuals))

                return cv_RSS

            # optimization
            problem = FunctionalProblem(
                1,
                cv_obj,
                x0=np.array([1e-2]),
                xl=np.array([1e-9]),
                xu=np.array([1e5])
            )

            result = minimize(
                problem, 
                PatternSearch(n_sample_points=50), 
                #get_termination("n_eval", 500),
                #get_termination("time", "00:01:00"),
                get_termination("ftol", 1e-8),
                verbose=False, 
                seed=1203477
            )

            # output
            cv_lambdas.append(result.X[0])
            cv_results.append(result)

        return({
            'model': "ESN",
            'method': "ridgeFit.cvDirectMultistep",
            'cvLambda': cv_lambdas,
            'result': cv_results,
            # additional info
            'steps': steps,
            'init': init,
            'burnin': burnin,
        })


# QUADRATIC PROGRAMMING
class qpFit:
    def __init__(self, **kwargs) -> None:
        # optional QP constraints
        # NOTE: using the CVXOPT convention, see e.g.
        # https://cvxopt.org/userguide/coneprog.html#quadratic-programming

        G = None if not 'G' in kwargs.keys() else kwargs['G']
        h = None if not 'h' in kwargs.keys() else kwargs['h']
        A = None if not 'A' in kwargs.keys() else kwargs['A']
        b = None if not 'b' in kwargs.keys() else kwargs['b']

        # parameter check
        assert not ((not G is None and h is None) or (G is None and not h is None)), "Must provide both G and h for inequality constraints"
        assert not ((not A is None and b is None) or (A is None and not b is None)), "Must provide both A and b for equality constraints"

        self.G = G
        self.h = h
        self.A = A
        self.b = b

    def fit(
        self, 
        model: ESN, 
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
        burnin = 0 if not 'burnin' in kwargs.keys() else kwargs['burnin']
        assert burnin >= 0, "burnin must be a non-negative integer"

        # collect states
        model_states = states(
            Z, 
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
            inti=init
        )

        # slice burn-in periods
        X0 = model_states[burnin:,]
        Y = Y[burnin:,]

        # fit
        W = qp(X=X0, Y=Y, G=self.G, h=self.h, A=self.A, b=self.b)

        # outputs
        T = X0.shape[0]
        X = np.hstack((np.ones((T, 1)), X0))
        Y_fit = W @ X
        residuals = Y - Y_fit
        MSE = np.mean(np.square(residuals))

        return({
            'model': "ESN",
            'method': "qpFit",
            'W': W,
            'states': model_states,
            'X': X,
            'Y': Y,
            'Y_fit': Y_fit,
            'residuals': residuals,
            'MSE': MSE,
            # method info
            'G': self.G,
            'h': self.h,
            'A': self.A,
            'b': self.b,
            # additional info
            'init': init,
            'burnin': burnin,
        })

def qp(X: np.ndarray, Y: np.ndarray, G=None, h=None, A=None, b=None):
    # sanity checks
    Tx = X.shape[0]
    Ty = Y.shape[0]
    assert Tx == Ty, "Shapes of X and Y non compatible"

    V = np.hstack((np.ones((Tx, 1)), X))
    P = cvx_matrix(V.T @ V)
    q = cvx_matrix(V.T @ Y)

    if G is None and A is None:
        W = cvx_solvers.qp(P, q)
    elif not G is None and A is None:
        G = cvx_matrix(G) 
        h = cvx_matrix(h) 
        W = cvx_solvers.qp(P, q, G=G, h=h)
    elif G is None and not A is None:
        A = cvx_matrix(A) 
        b = cvx_matrix(b)
        W = cvx_solvers.qp(P, q, A=A, b=b)
    else:
        G = cvx_matrix(G) 
        h = cvx_matrix(h) 
        A = cvx_matrix(A) 
        b = cvx_matrix(b)
        W = cvx_solvers.qp(P, q, G=G, h=h, A=A, b=b)

    return W