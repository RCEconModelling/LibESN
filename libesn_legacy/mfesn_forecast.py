#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

import pandas as pd
import numpy as np
# from numpy import linalg as npLA
# from numba import njit

from libesn_legacy.base_utils import * 
from libesn_legacy.base_datetime import *
from libesn_legacy.esn_states import states, iter_state
from libesn_legacy.mfesn import MFESN
from libesn_legacy.mfesn_states import mf_dt_states, mf_dt_states_to_matrix

def mfForecast(
    mfmodel: MFESN, 
    forecast_data, 
    fit: dict,
    ref_dates=None,
    stack_init=True,
    **kwargs
) -> dict:
    """
    Forecast using MFESN model.
    mfmodel: MFESN class model to forecast with.
    forecast_data: dataset of regressors to use for forecasting.
    fit: MFESN fit that contains parameter matrix W.
    ref_dates: slicing dates for states collected with forecast_data. 
               If not set, the function will try to slice align states 
               according to lowest-frequency input data in forecast_data.
    stack_init: optionally stack init values when making forecast.
    """
    # prepare data
    Zf, Zf_dates = pd_data_prep(forecast_data)

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']

    # load init states and dates
    init_list = []
    init_dates = []
    if init is None:
        # init = [np.zeros(n) for n in mfmodel.N]
        # slice fit states as init using dates
        for j, Xj in enumerate(fit['states']):
            init_list.append(Xj.iloc[-1,:].to_numpy().squeeze())
            cpd = Xj.index[-1]
            init_dates.append(cpd)
    else:
        for j, initj in enumerate(init):
            d = Zf_dates[j]
            cpd, _ = closest_past_date(Xj.index, d, strict=True)
            init_list.append(initj.loc[cpd].to_numpy().squeeze())
            init_dates.append(cpd)

    # Shape checks
    assert fit['W'].shape[0] == mfmodel.M+1, "fit['W'] and MFESN state dimensions are not compatible"
    W = fit['W']

    # forecast states
    mf_forecast_states = mf_dt_states(
        input=Zf, 
        mfmodel=mfmodel, 
        dates=Zf_dates, 
        init=init_list, 
    )

    # find minimum frequency
    state_ref_dates = None
    if ref_dates is None:
        state_ref_dates = pd_min_freq_dates(Zf_dates)
    else:
        assert isinstance(ref_dates, pd.DatetimeIndex), "Supplied ref_dates is not a pd.DatetimeIndex"
        state_ref_dates = ref_dates

    # slice and stack model states plus slicing dates
    mf_X0_f, mf_X0_f_dates = mf_dt_states_to_matrix(
        states=mf_forecast_states, 
        ref_dates=state_ref_dates, 
        states_join=mfmodel.states_join, 
        states_lags=mfmodel.states_lags,
    )

    # stack init
    if stack_init:
        mf_X_f = np.vstack([
            fit['mf_X'][[-1],:], mf_X0_f,
        ])
        forecast_dates = pd.DatetimeIndex([]).union([pd_min_freq_dates(fit['mf_X_dates'])[-1]]).union(state_ref_dates)
    else:
        mf_X_f = mf_X0_f
        forecast_dates = state_ref_dates

    # Forecast
    T = mf_X_f.shape[0]
    mf_X_n = np.hstack((np.ones((T, 1)), mf_X_f))
    forecast = pd.DataFrame(data=(mf_X_n @ W), index=forecast_dates)

    return({
        'model': "MFESN",
        'method': "mfForecast",
        'states': mf_forecast_states,
        'mf_X0_f': mf_X0_f,
        'mf_X0_f_dates': mf_X0_f_dates,
        'mf_X_f': mf_X_f,
        'forecast': forecast,
        # additional info
        'init': init,
    })

def mfDirectHighFreqForecast(
    mfmodel: MFESN, 
    forecast_data, 
    fit: dict,
    freqs=None,
    terminal=True,
    **kwargs
) -> dict:
    """
    Forecast using MFESN model at high frequency.
    mfmodel: MFESN class model to forecast with.
    forecast_data: dataset of regressors to use for forecasting.
    fit: MFESN fit that contains parameter matrix W.
    ref_dates: slicing dates for states collected with forecast_data. 
               If not set, the function will try to slice align states 
               according to lowest-frequency input data in forecast_data.
    stack_init: optionally stack init values when making forecast.
    """
    # prepare data
    Zf, Zf_dates = pd_data_prep(forecast_data)

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']

    # load init states and dates
    init_list = []
    init_dates = []
    if init is None:
        # init = [np.zeros(n) for n in mfmodel.N]
        # slice fit states as init using dates
        for j, Xj in enumerate(fit['states']):
            init_list.append(Xj.iloc[-1,:].to_numpy().squeeze())
            cpd = Xj.index[-1]
            init_dates.append(cpd)
    else:
        for j, initj in enumerate(init):
            d = Zf_dates[j]
            cpd, _ = closest_past_date(Xj.index, d, strict=True)
            init_list.append(initj.loc[cpd].to_numpy().squeeze())
            init_dates.append(cpd)

    # TODO: currently, due to complexity, only state alignment option 'align'
    #       is implemented below
    assert mfmodel.states_join == "align", "mfDirectHighFreqForecast() currently only supports states_join='align'"

    # forecast steps
    steps_idx = [i for i in range(len(fit['fits'])) if fit['fits'][i]['s'] > 0]
    steps = [fit['fits'][i]['s'] for i in steps_idx]
    
    # Shape checks
    if fit['method'] == 'mfRidgeFit.fit':
        assert fit['W'].shape[0] == mfmodel.M+1, "fit['W'] and MFESN state dimensions are not compatible"
        W = fit['W']

    elif fit['method'] == 'mfRidgeFit.fitDirectHighFreq':
        # check for all 
        W_list = []
        for i in steps_idx:
            W_list_s = []
            for j, fitj in enumerate(fit['fits'][i]['freq_fits']):
                assert fitj['W'].shape[0] == mfmodel.M+1, f"fit['fits'][{j}]['W'] and MFESN state dimensions are not compatible"
                W_list_s.append(fitj['W'])
            # NOTE: cycle last element to correspond to
            #       modulus == 0 c.f. below
            W_list_s = [W_list_s[-1], ] + W_list_s[:-1]
            W_list.append(W_list_s)
        W = None

    # forecast states
    # NOTE: these are constructed directly here because one
    #       must handle cases where some data is missing due
    #       to HF forecasting setup
    Xfhff = []
    for j, Zfj in enumerate(Zf):
        if len(Zfj) == 0:
            # HF forecasting data for jth component is not available
            # just yield last collected state when fitted
            fstates_j = pd.DataFrame(
                data=init_list[j][None,:], 
                index=[init_dates[j]],
            )
        else:
            # collect states
            fstates_j = states(
                Zfj, 
                map=mfmodel.models[j].pars.smap, 
                A=mfmodel.models[j].pars.A, 
                C=mfmodel.models[j].pars.C, 
                zeta=mfmodel.models[j].pars.zeta, 
                rho=mfmodel.models[j].pars.rho, 
                gamma=mfmodel.models[j].pars.gamma, 
                leak=mfmodel.models[j].pars.leak, 
                init=init_list[j],
            )
            # add dates to states
            fstates_dates_j = pd.DatetimeIndex([]).union([init_dates[j]]).union(Zf_dates[j])
            fstates_j = pd.DataFrame(
                data=np.vstack([init_list[j], fstates_j]), 
                index=fstates_dates_j,
            )
        # append
        Xfhff.append(fstates_j)

    # find high-frequency dates
    if freqs is None:
        # no suggestion, need to infer from data
        max_freq_dates = pd_max_freq_dates(Zf_dates)
    else:
        for f in freqs:
            assert type(f) is str, "All frequencies in 'freqs' must be strings"
        max_freq_j = pd_max_freq(freqs, return_index=True)
        max_freq_dates = Zf_dates[max_freq_j]

    # high-freq forecast
    hffor = []
    if terminal:
        # only forecast using the terminal state
        T = 1
        hff_X = np.full((1, mfmodel.M), np.nan)
        hff_X_dates = []

        p = 0
        for j, Xfhffj in enumerate(Xfhff):
            cpd, _ = closest_past_date(Xfhffj.index, max_freq_dates[-1])
            hff_X[0,p:(p + mfmodel.N[j])] = np.squeeze(Xfhffj.loc[cpd])
            # save slice date
            hff_X_dates.append(cpd)
            p += mfmodel.N[j]

        hfforecast_dates = [max_freq_dates[-1]]

        for i, s in enumerate(steps):
            if fit['method'] == 'mfRidgeFit.fitDirectHighFreq':
                # must choose the right fit to nowcast
                # NOTE: this is a bit hacky, but we recycle the last index date
                #       from the first fit as a reference and count the division
                #       residual to terminal date
                fit_lf_last_date = fit['fits'][i]['freq_fits'][0]['mf_Y'].index[-1]
                h = len(max_freq_dates[max_freq_dates >= fit_lf_last_date]) % fit['freqratio'] 
                # select step W
                W_s = W_list[i][h]

            mf_X_hff = np.hstack((np.ones((1, 1)), hff_X))
            hfforecast = pd.DataFrame(data=(mf_X_hff @ W_s), index=hfforecast_dates)

            # output
            hffor.append({
                's': s,
                'forecast': hfforecast
            })

    else:
        # nowcast using all collected states
        T = len(max_freq_dates)
        hff_X = np.full((T, mfmodel.M), np.nan)
        hff_X_dates = []

        p = 0
        for j, Xfnj in enumerate(Xfhff):
            kt = 0
            cpdl_j = []
            for t, hf_date_t in enumerate(max_freq_dates):
                cpd, kt = closest_past_date(Xfnj.index, hf_date_t, cutoff=kt)
                hff_X[t,p:(p + mfmodel.N[j])] = np.squeeze(Xfnj.loc[cpd])
                # save slice date
                cpdl_j.append(cpd)
            p += mfmodel.N[j]
            # save alignment dates
            hff_X_dates.append(pd.to_datetime(cpdl_j))

        hfforecast_dates = max_freq_dates

        for i, s in enumerate(steps):
            if fit['method'] == 'mfRidgeFit.fitDirectHighFreq':
                # must choose the right fit to *initiate* nowcast
                # NOTE: see above
                fit_lf_last_date = fit['fits'][0]['freq_fits'][0]['mf_Y'].index[-1]
                h = len(max_freq_dates[max_freq_dates >= fit_lf_last_date]) % fit['freqratio'] 

                T = hff_X.shape[0]
                K = W_list[i][h].shape[1]
                mf_X_hff = np.hstack((np.ones((T, 1)), np.full((T, mfmodel.M), np.nan)))

                # iterate over dates
                hfforecast = pd.DataFrame(data=np.full((T, K), np.nan), index=hfforecast_dates)
                for j, nd in enumerate(hfforecast_dates):
                    mf_X_hff[j,1:] = hff_X[j,:]
                    
                    W_j = W_list[i][((j + h) % fit['freqratio'])]
                    hfforecast.loc[nd] = np.hstack([1, hff_X[j,:]]) @ W_j

            else:
                T = hff_X.shape[0]
                mf_X_hff = np.hstack((np.ones((T, 1)), hff_X))
                hfforecast = pd.DataFrame(data=(mf_X_hff @ W), index=hfforecast_dates)

            # output
            hffor.append({
                's': s,
                'forecast': hfforecast
            })
    

    return({
        'model': "MFESN",
        'method': "mfNowcast",
        'states': Xfhff,
        'mf_X_n': mf_X_hff,
        'forecast': hffor,
        # additional info
        'hff_X_dates': hff_X_dates, 
        'init': init,
    })

def mfNowcast(
    mfmodel: MFESN, 
    nowcast_data, 
    fit: dict,
    freqs=None,
    terminal=True,
    **kwargs
) -> dict:
    """
    Nowcast using MFESN model.
    mfmodel: MFESN class model to forecast with.
    nowcast_data: dataset of regressors to use for nowcasting.
    fit: MFESN fit that contains parameter matrix W.
         If fit was produced by fitDirectHighFreq(), correctly
         use the high-frequency W's fits based on frequency ratio
    freqs: pandas string for frequency of observations of regressors
           in nowcast_data
    terminal: if True, produce only a nowcast for the terminal state
              collected from nowcast_data, useful in e.g. online nowcasting.
    """
    # prepare data
    Zf, Zf_dates = pd_data_prep(nowcast_data)

    # handle kwargs
    init = None if not 'init' in kwargs.keys() else kwargs['init']
    
    # load init states and dates
    init_list = []
    init_dates = []
    if init is None:
        # init = [np.zeros(n) for n in mfmodel.N]
        # slice fit states as init using dates
        for j, Xj in enumerate(fit['states']):
            init_list.append(Xj.iloc[-1,:].to_numpy().squeeze())
            cpd = Xj.index[-1]
            init_dates.append(cpd)
    else:
        for j, initj in enumerate(init):
            d = Zf_dates[j]
            cpd, _ = closest_past_date(Xj.index, d, strict=True)
            init_list.append(initj.loc[cpd].to_numpy().squeeze())
            init_dates.append(cpd)

    # TODO: currently, due to complexity, only state alignment option 'align'
    #       is implemented below
    assert mfmodel.states_join == "align", "mfNowcast() currently only supports states_join='align'"

    # Shape checks
    if fit['method'] == 'mfRidgeFit.fit':
        assert fit['W'].shape[0] == mfmodel.M+1, "fit['W'] and MFESN state dimensions are not compatible"
        W = fit['W']
    
    elif fit['method'] == 'mfRidgeFit.fitDirectHighFreq':
        # check steps of fits
        if len(fit['fits']) > 1:
            print(f"Length of fit['fits'] > 1, fitDirectHighFreq() will only use the first (s == 0) fit!")
        assert fit['fits'][0]['s'] == 0, "Step 0 (nowcasting) fit missing!"
        
        # check for all 
        W_list = []
        for j, fitj in enumerate(fit['fits'][0]['freq_fits']):
            assert fitj['W'].shape[0] == mfmodel.M+1, f"fit['fits'][{j}]['W'] and MFESN state dimensions are not compatible"
            W_list.append(fitj['W'])
        # NOTE: cycle last element to correspond to
        #       modulus == 0 c.f. below
        W_list = [W_list[-1], ] + W_list[:-1]
        W = None

    # nowcast states
    # NOTE: these are constructed directly here because one
    #       must handle cases where some data is missing due
    #       to nowcasting setup
    Xfn = []
    for j, Zfj in enumerate(Zf):
        if len(Zfj) == 0:
            # nowcasting data for jth component is not available
            # just yield last collected state when fitted
            fstates_j = pd.DataFrame(
                data=init_list[j][None,:], 
                index=[init_dates[j]],
            )
        else:
            # collect states
            fstates_j = states(
                Zfj, 
                map=mfmodel.models[j].pars.smap, 
                A=mfmodel.models[j].pars.A, 
                C=mfmodel.models[j].pars.C, 
                zeta=mfmodel.models[j].pars.zeta, 
                rho=mfmodel.models[j].pars.rho, 
                gamma=mfmodel.models[j].pars.gamma, 
                leak=mfmodel.models[j].pars.leak, 
                init=init_list[j],
            )
            # add dates to states
            fstates_dates_j = pd.DatetimeIndex([]).union([init_dates[j]]).union(Zf_dates[j])
            fstates_j = pd.DataFrame(
                data=np.vstack([init_list[j], fstates_j]), 
                index=fstates_dates_j,
            )
            # fstates_j = pd.DataFrame(
            #     data=fstates_j, 
            #     index=Zf_dates[j],
            # )
        # append
        Xfn.append(fstates_j)

    # find high-frequency dates
    if freqs is None:
        # no suggestion, need to infer from data
        max_freq_dates = pd_max_freq_dates(Zf_dates)
    else:
        for f in freqs:
            assert type(f) is str, "All frequencies in 'freqs' must be strings"
        max_freq_j = pd_max_freq(freqs, return_index=True)
        max_freq_dates = Zf_dates[max_freq_j]
    
    # Nowcast
    if terminal:
        # only nowcast using the terminal state
        T = 1
        n_X = np.full((1, mfmodel.M), np.nan)
        n_X_dates = []

        p = 0
        for j, Xfnj in enumerate(Xfn):
            cpd, _ = closest_past_date(Xfnj.index, max_freq_dates[-1])
            n_X[0,p:(p + mfmodel.N[j])] = np.squeeze(Xfnj.loc[cpd])
            # save slice date
            n_X_dates.append(cpd)
            p += mfmodel.N[j]

        nowcast_dates = [max_freq_dates[-1]]

        if fit['method'] == 'mfRidgeFit.fitDirectHighFreq':
            # must choose the right fit to nowcast
            # NOTE: this is a bit hacky, but we recycle the last index date
            #       from the first fit as a reference and count the division
            #       residual to terminal date
            fit_lf_last_date = fit['fits'][0]['freq_fits'][0]['mf_Y'].index[-1]
            h = len(max_freq_dates[max_freq_dates >= fit_lf_last_date]) % fit['freqratio'] 
            W = W_list[h]

        mf_X_n = np.hstack((np.ones((1, 1)), n_X))
        nowcast = pd.DataFrame(data=(mf_X_n @ W), index=nowcast_dates)

    else:
        # nowcast using all collected states
        T = len(max_freq_dates)
        n_X = np.full((T, mfmodel.M), np.nan)
        n_X_dates = []

        p = 0
        for j, Xfnj in enumerate(Xfn):
            kt = 0
            cpdl_j = []
            for t, hf_date_t in enumerate(max_freq_dates):
                cpd, kt = closest_past_date(Xfnj.index, hf_date_t, cutoff=kt)
                n_X[t,p:(p + mfmodel.N[j])] = np.squeeze(Xfnj.loc[cpd])
                # save slice date
                cpdl_j.append(cpd)
            p += mfmodel.N[j]
            # save alignment dates
            n_X_dates.append(pd.to_datetime(cpdl_j))

        nowcast_dates = max_freq_dates

        if fit['method'] == 'mfRidgeFit.fitDirectHighFreq':
            # must choose the right fit to *initiate* nowcast
            # NOTE: see above
            fit_lf_last_date = fit['fits'][0]['freq_fits'][0]['mf_Y'].index[-1]
            h = len(max_freq_dates[max_freq_dates >= fit_lf_last_date]) % fit['freqratio'] 

            T = n_X.shape[0]
            K = W_list[h].shape[1]
            mf_X_n = np.hstack((np.ones((T, 1)), np.full((T, mfmodel.M), np.nan)))

            # iterate over dates
            nowcast = pd.DataFrame(data=np.full((T, K), np.nan), index=nowcast_dates)
            for j, nd in enumerate(nowcast_dates):
                mf_X_n[j,1:] = n_X[j,:]
                
                W_j = W_list[((j + h) % fit['freqratio'])]
                nowcast.loc[nd] = np.hstack([1, n_X[j,:]]) @ W_j

        else:
            T = n_X.shape[0]
            mf_X_n = np.hstack((np.ones((T, 1)), n_X))
            nowcast = pd.DataFrame(data=(mf_X_n @ W), index=nowcast_dates)

    return({
        'model': "MFESN",
        'method': "mfNowcast",
        'states': Xfn,
        'mf_X_n': mf_X_n,
        'nowcast': nowcast,
        # additional info
        'n_X_dates': n_X_dates, 
        'init': init,
    })
