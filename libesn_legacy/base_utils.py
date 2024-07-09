#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

import pandas as pd
import numpy as np

# Data
def pd_to_np_array(data):
    v = None
    
    if data is None:
        # return empty array
        # return []
        return np.array([])

    elif (type(data) is pd.DataFrame):
        v = data.to_numpy(copy=True)
    elif (type(data) is pd.Series):
        v = data.to_numpy(copy=True)[:,None]
    elif type(data) is np.ndarray:
        v = np.copy(data)
    
    if (not v is None) and (v.ndim == 1):
        # Mutate 1D vector to 2D column vector
        v = np.atleast_2d(v).reshape((-1,1))
    #print(v.shape)

    # return v as contiguous array 
    # (needed for Numba in case data was transposed)
    return np.ascontiguousarray(v)

def pd_data_prep(V):
    if type(V) is tuple or type(V) is list:
        V_dates = []
        V_ = [] 
        #assert len(Z) > 0
        if len(V) > 0:
            for j, Vj in enumerate(V):
                try:
                    Vj_dates = Vj.index
                    # assert isinstance(Vj_dates, pd.DatetimeIndex)
                except:
                    if not Vj is None:
                        Vj_dates = np.arange(start=0, stop=Vj.shape[0])
                    else:
                        Vj_dates = None
                Vj_ = pd_to_np_array(Vj)
                if Vj_ is None:
                    raise TypeError(f"Type of data at index {j} not recognized, need pandas.DataFrame or numpy.ndarray")

                V_dates.append(Vj_dates)
                V_.append(Vj_)
        # Make immutable
        V_dates = tuple(V_dates)
        V_ = tuple(V_)
    else:
        try:
            V_dates = V.index
            # assert isinstance(V_dates, pd.DatetimeIndex)
        except:
            if not V is None:
                V_dates = np.arange(start=0, stop=V.shape[0])
            else:
                V_dates = None
        V_ = pd_to_np_array(V)
        if V_ is None:
            raise TypeError("Type of data not recognized, need pandas.DataFrame or numpy.ndarray")

    return V_, V_dates

class ShiftTimeSeriesSplit:
    def __init__(
        self, 
        length, 
        test_size=1, 
        min_train_size=1, 
        max_train_size=np.inf,
        overlap=False,
    ):
        assert length >= 1, "Sample length must be a positive integer"
        assert test_size >= 1, "Test set size must be a positive integer"
        
        assert min_train_size > 0
        if not max_train_size is None:
            assert max_train_size > 0, "Maximum train set size must be a positive integer"
            assert max_train_size >= min_train_size, "Maximum train set size must be greater or equal to minimum split size"

        assert overlap is True or overlap is False, "Overlap must be a boolean"

        self.length = length

        self.test_size = test_size
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.overlap = overlap
        
    def split(self):
        flag_mss = False
        if not self.max_train_size is None:
            flag_mss = True

        T = self.length
        # Compute number of splits
        if self.overlap:
            n_splits = T - self.min_train_size - self.test_size + 1
        else:
            m = T - self.min_train_size
            n_splits = np.floor(m / self.test_size).astype(int)
            if self.test_size == 1:
                self.n_splits -= 1

        train_idxs = []
        test_idxs  = []
        t_i = self.min_train_size
        for _ in range(n_splits):
            v_i = t_i if self.max_train_size is None else self.max_train_size
            start_idx = max(0, t_i - v_i) if flag_mss else 0

            train_idxs.append(list(range(start_idx, t_i)))
            test_idxs.append(list(range(t_i, t_i + self.test_size)))

            if self.overlap:
                t_i += 1
            else:
                t_i += self.test_size

        return tuple(zip(train_idxs, test_idxs))

    def __iter__(self):
        self.flag_mss = False if not self.max_train_size is None else True

        T = self.length
        # Compute number of splits
        if self.overlap:
            self.n_splits = T - self.min_train_size - self.test_size + 1
        else:
            m = T - self.min_train_size
            self.n_splits = np.floor(m / self.test_size).astype(int)
            if self.test_size == 1:
                self.n_splits -= 1

        # iterator variables
        self.i = 0
        self.minss = self.min_train_size
        self.maxss = self.max_train_size

        return self

    def __next__(self):
        if self.i < self.n_splits:
            v_i = self.minss if self.max_train_size is None else self.max_train_size
            start_idx = max(0, self.minss - v_i) if self.flag_mss else 0

            train_idx = list(range(start_idx, self.minss))
            test_idxs = list(range(self.minss, self.minss + self.test_size))
            
            self.i += 1
            if self.overlap:
                self.minss += 1
            else:
                self.minss += self.test_size

            return (train_idx, test_idxs)
        else:
            raise StopIteration