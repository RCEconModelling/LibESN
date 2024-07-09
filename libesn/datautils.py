import pandas as pd
import numpy as np

# Data
def pd_to_np_array(data):
    if data is None:
        # return empty array
        return np.array([])
    
    if (type(data) is pd.DataFrame):
        v = data.to_numpy(copy=True)
    elif (type(data) is pd.Series):
        v = data.to_numpy(copy=True)[:,None]
    elif type(data) is np.ndarray:
        v = np.copy(data)
    
    if v.ndim == 1:
        v = v[:, None] # Reshape 1D vector to 2D column vector
    
    #print(v.shape)

    return np.ascontiguousarray(v) # Make contiguous for Numba

def pd_data_prep(V):
    if type(V) is tuple or type(V) is list:
        V_dates = []
        V_data = [] 
        #assert len(Z) > 0
        if len(V) > 0:
            for j, Vj in enumerate(V):
                try:
                    V_j_dates = Vj.index
                    # assert isinstance(V_j_dates, pd.DatetimeIndex)
                except:
                    if not Vj is None:
                        V_j_dates = np.arange(start=0, stop=Vj.shape[0])
                    else:
                        V_j_dates = None
                V_j_data = pd_to_np_array(Vj)
                if V_j_data is None:
                    raise TypeError(f"Type of data at index {j} not recognized, need pandas.DataFrame or numpy.ndarray")

                V_dates.append(V_j_dates)
                V_data.append(V_j_data)
        # Make immutable
        V_dates = tuple(V_dates)
        V_data = tuple(V_data)
    else:
        try:
            V_dates = V.index
            # assert isinstance(V_dates, pd.DatetimeIndex)
        except:
            if not V is None:
                V_dates = np.arange(start=0, stop=V.shape[0])
            else:
                V_dates = None
        V_data = pd_to_np_array(V)
        if V_data is None:
            raise TypeError("Type of data not recognized, need pandas.DataFrame or numpy.ndarray")

    return V_data, V_dates