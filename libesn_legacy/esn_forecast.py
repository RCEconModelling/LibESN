#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

from typing import Union

import numpy as np
# from numba import njit

from libesn_legacy.base_utils import * 
from libesn_legacy.esn import ESN
from libesn_legacy.esn_states import states, generate

def forecast(
    model: ESN, 
    forecast_data, 
    fit: dict,
    **kwargs
) -> dict:
    # prepare data
    Zf, Zf_dates = pd_data_prep(forecast_data)

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']
    if init is None:
        init = fit['states'][-1,:]

    # Shape checks
    assert fit['W'].shape[0] == model.pars.N+1, "fit['W'] and ESN state dimensions are not compatible"
    W = fit['W']

    # forecast states
    forecast_states = states(
        Zf, 
        map=model.pars.smap, 
        A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
        rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
        init=init
    )

    # stack init
    X0f = np.vstack([init, forecast_states])

    # Forecast
    T = X0f.shape[0]
    Xf = np.hstack((np.ones((T, 1)), X0f))
    forecast = Xf @ W

    return({
        'model': "ESN",
        'method': "forecast",
        'states': Xf,
        'forecast': forecast,
        # additional info
        'init': init,
    })

def autonomousForecast(
    model: ESN,
    fit: dict,
    steps: int,
    **kwargs
) -> dict:
    
    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']
    if init is None:
        init = fit['states'][[-1],:]
    
    # Shape checks
    assert fit['W'].shape[0] == model.pars.N+1, "fit['W'] and ESN state dimensions are not compatible"
    W = fit['W']

    # forecast
    forecast = []
    for_idx = []

    if steps > 0:
        # generate forecasting states 
        Xfg = generate(
            states=init,
            length=int(steps),
            W=W,
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
        )

        Xf = np.hstack([
            np.ones((steps, 1)), 
            np.squeeze(Xfg).T
        ])
        forecast = Xf @ W
        for_idx = list(range(1, steps+1))
    
    else:
        raise ValueError("Autonomous forecasting steps should be a positive integer number.")
    
    return({
        'model': "ESN",
        'method': "autonomousForecast",
        'states': Xf,
        'forecast': forecast,
        'forecast_index': for_idx,
        # additional info
        'init': init,
    })

def forecastMultistep(
    model: ESN, 
    forecast_data, 
    fit: dict,
    **kwargs
) -> dict:
    # prepare data
    Zf, Zf_dates = pd_data_prep(forecast_data)

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']
    if init is None:
        init = fit['states'][-1,:]

    # forecast states
    forecast_states = states(
        Zf, 
        map=model.pars.smap, 
        A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
        rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
        init=init
    )

    # stack init
    X0f = np.vstack([init, forecast_states])

    # steps
    steps = fit['steps']
    
    # forecast
    forecast = []
    for_idx = []
    if int(max(steps)) > 1:
        # generate forecasting states 
        Xfg = generate(
            states=X0f,
            length=int(max(steps)),
            W=fit['W_ar'],
            map=model.pars.smap, 
            A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
            rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
        )

        # stack states (row-wise)
        X0fg = np.dstack([X0f[:,:,None], Xfg])

        # for i, s in enumerate(range(int(max(steps)))):
        #     X0fg_t = np.squeeze(X0fg[:,:,i])

        #     W_s = fit['fits'][i]['W']

        #     T_s = X0fg_t.shape[0]
        #     Xf_s = np.hstack((np.ones((T_s, 1)), X0fg_t))
            
        #     forecast.append(Xf_s @ W_s)
        #     for_idx.append(np.arange(s, T_s+s))

        for t in range(X0fg.shape[0]):
            X0fg_t = np.squeeze(X0fg[t,:,:]).T
            Xf_t = np.hstack([np.ones((X0fg_t.shape[0], 1)), X0fg_t])

            forecast_t = np.zeros([len(steps), fit['fits'][0]['W'].shape[1]])
            for_idx_t = t + np.array(steps) - 1
            for i in range(int(max(steps))):
                # Shape checks
                assert fit['fits'][i]['W'].shape[0] == model.pars.N+1, "fit['W'] and ESN state dimensions are not compatible"
                W_s = fit['fits'][i]['W']

                forecast_t[i,:] = Xf_t[[i],:] @ W_s

            forecast.append(forecast_t)
            for_idx.append(for_idx_t)

        Xf = X0fg
    else:
        # Shape checks
        assert fit['fits'][0]['W'].shape[0] == model.pars.N+1, "fit['W'] and ESN state dimensions are not compatible"
        W = fit['fits'][0]['W']

        T = X0f.shape[0]
        Xf = np.hstack((np.ones((T, 1)), X0f))
        
        forecast.append(Xf @ W)
        for_idx.append(np.arange(1, T))

    return({
        'model': "ESN",
        'method': "forecastMultistep",
        'states': Xf,
        'forecast': forecast,
        'forecast_index': for_idx,
        # additional info
        'init': init,
    })

def forecastDirectMultistep(
    model: ESN, 
    forecast_data, 
    fit: dict,
    **kwargs
) -> dict:
    # prepare data
    Zf, Zf_dates = pd_data_prep(forecast_data)

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']
    if init is None:
        init = fit['states'][-1,:]

    # forecast states
    forecast_states = states(
        Zf, 
        map=model.pars.smap, 
        A=model.pars.A, C=model.pars.C, zeta=model.pars.zeta, 
        rho=model.pars.rho, gamma=model.pars.gamma, leak=model.pars.leak, 
        init=init
    )

    # stack init
    X0f = np.vstack([init, forecast_states])

    # steps
    steps = fit['steps']
    
    # forecast
    T = X0f.shape[0]
    Xf = np.hstack((np.ones((T, 1)), X0f))

    forecast = []
    for_idx = []
    if int(max(steps)) > 1:
        for t in range(X0f.shape[0]):
            forecast_t = np.zeros([len(steps), fit['fits'][0]['W'].shape[1]])
            for_idx_t = t + np.array(steps) - 1
            for i in range(int(max(steps))):
                # Shape checks
                assert fit['fits'][i]['W'].shape[0] == model.pars.N+1, "fit['W'] and ESN state dimensions are not compatible"
                W_s = fit['fits'][i]['W']

                forecast_t[i,:] = Xf[[t],:] @ W_s

            forecast.append(forecast_t)
            for_idx.append(for_idx_t)
    else:
        # Shape checks
        assert fit['fits'][0]['W'].shape[0] == model.pars.N+1, "fit['W'] and ESN state dimensions are not compatible"
        W = fit['fits'][0]['W']
        
        forecast.append(Xf @ W)
        for_idx.append(np.arange(1, T))

    return({
        'model': "ESN",
        'method': "forecastDirectMultistep",
        'states': Xf,
        'forecast': forecast,
        'forecast_index': for_idx,
        # additional info
        'init': init,
    })