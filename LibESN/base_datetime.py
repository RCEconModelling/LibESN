#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

import pandas as pd

def closest_past_date(list_date, base_date, cutoff=0, strict=False):
    """
    list_date: collections of dates to compare.
    base_date: reference date to compate to for closest match.
    cutoff: cutoff index to reduce search array.
    """
    #return min([i for i in list_date if i <= base_date], key=lambda x: abs(x - base_date)), 0
    assert type(strict) is bool, "strict must be a boolean"
    try:
        if strict:
            d_min = min([i for i in list_date[cutoff:] if i < base_date], key=lambda x: abs(x - base_date))
        else:
            d_min = min([i for i in list_date[cutoff:] if i <= base_date], key=lambda x: abs(x - base_date))
    except ValueError:
        print(f"Error at: {base_date} with cutoff: {cutoff}")
        raise ValueError
    return d_min, list_date.get_loc(d_min)
    

def closest_future_date(list_date, base_date, cutoff=None, strict=False):
    """
    list_date: collections of dates to compare.
    base_date: reference date to compate to for closest match.
    cutoff: cutoff index to reduce search array.
    strict: strict inequality in definition.
    """
    #return min([i for i in list_date if i >= base_date], key=lambda x: abs(x - base_date)), 0
    assert type(strict) is bool, "strict must be a boolean"
    try:
        if strict:
            d_min = min([i for i in list_date[:cutoff] if i > base_date], key=lambda x: abs(x - base_date))
        else:
            d_min = min([i for i in list_date[:cutoff] if i >= base_date], key=lambda x: abs(x - base_date))
    except ValueError:
        print(f"Error at: {base_date} with cutoff: {cutoff}")
        raise ValueError
    return d_min, list_date.get_loc(d_min)

def pd_infer_periods(freq, periods=10**3, scale=100):
    """
    freq : str pandas frequency alias.
    periods : numeric, given freq, should create many years. 
    scale: scale of years to group by (century = 100).
    """
    # while True:
    #     try:
    #         s = pd.Series(data=pd.date_range('1970-01-01', freq=freq, periods=periods))
    #         break
    #     # reduce periods if too large
    #     except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError): 
    #         periods = periods / 10
    # return s.groupby(s.dt.year // scale * scale).size().value_counts().index[0]
    # NOTE: the above is too convoluted, use diffs 
    while True:
        try:
            test_series = pd.Series(data=pd.date_range('1970-01-01', freq=freq, periods=periods))
            break
        # reduce periods if too large
        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError): 
            periods = periods / 10
    return test_series.diff().max()

def pd_max_2freq(f1, f2):
    """
    Find the maximum frequency between two pandas freq strings.
    f1 : frequency 1, str pandas frequency alias.
    f2 : frequency 2, str pandas frequency alias.
    """
    p1 = pd_infer_periods(f1)
    p2 = pd_infer_periods(f2)
    return (f1 if p1 > p2 else f2)

def pd_max_freq(*freqs, **kwargs):
    """
    Find the maximum frequency in either a variable number of pandas frequency
    strings or a list/tuple of freq. strings.
    *freqs: arguments or list/tuple
    """
    return_index = False if not 'return_index' in kwargs.keys() else kwargs['return_index']
    assert type(return_index) is bool, "return_index must be a boolean"
    
    if len(freqs) == 1:
        freqs = freqs[0]
        assert len(freqs) > 0

    pmax = pd.Timedelta(days=1e5)
    fmax = None
    jmax = 0
    for j, f in enumerate(freqs):
        if not f is None:
            p = pd_infer_periods(f)
            if p < pmax:
                pmax = p
                fmax = f
                jmax = j

    if return_index:
        return jmax
    return fmax

def pd_max_freq_dates(*dates, **kwargs):
    """
    Find the dates (index) associated to the maximum frequency in either a 
    variable number of pandas frequency strings or a list/tuple of freq. strings.
    *dates: arguments or list/tuple
    """
    return_index = False if not 'return_index' in kwargs.keys() else kwargs['return_index']
    assert type(return_index) is bool, "return_index must be a boolean"
    
    if len(dates) == 1:
        dates = dates[0]
        assert len(dates) > 0

    pmax = pd.Timedelta(days=1e5)
    # fmax = None
    dmax = None
    jmax = 0
    for j, d in enumerate(dates):
        if not d is None:
            f = pd.infer_freq(d)
            p = pd_infer_periods(f)
            if p < pmax:
                pmax = p
                # fmax = f
                dmax = d
                jmax = j

    if return_index:
        return jmax
    return dmax

def pd_min_freq_dates(*dates, **kwargs):
    """
    Find the dates (index) associated to the minimum frequency in either a 
    variable number of pandas frequency strings or a list/tuple of freq. strings.
    *dates: arguments or list/tuple
    """
    return_index = False if not 'return_index' in kwargs.keys() else kwargs['return_index']
    assert type(return_index) is bool, "return_index must be a boolean"
    
    if len(dates) == 1:
        dates = dates[0]
        assert len(dates) > 0

    pmin = pd.Timedelta(microseconds=1)
    # fmin = None
    dmin = None
    jmin = 0
    for j, d in enumerate(dates):
        if not d is None:
            f = pd.infer_freq(d)
            p = pd_infer_periods(f)
            if p > pmin:
                pmin = p
                # fmin = f
                dmin = d
                jmin = j
                
    if return_index:
        return jmin
    return dmin