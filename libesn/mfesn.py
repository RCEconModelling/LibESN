from typing import Union

import numpy as np

from libesn.esn import ESN
from libesn.console import console

# from rich.table import Table
# from rich import box

class MFESN:
    def __init__(
        self,
        models: tuple,
        states_join: str = "align",
        states_lags: Union[tuple, list, np.ndarray] = None,
    ) -> None:
        assert type(models) is tuple, "models must be a tuple of ESN models"
        for m in models:
            assert isinstance(m, ESN), "All models in a MFESN must be ESN models"

        assert states_join in ("align", "lagstack"),  "State joining specification must be one of 'align', 'lagstack'"
        if (states_join == "align") and (not states_lags is None):
            states_lags = None

            print(f"[+] libesn.MFESN - - - - - - - - - - -")
            print(f" !  State joining is 'align', lags in states_lags ignored")
        if not states_lags is None:
            assert len(states_lags) == len(models), "State lags specification must be of same length as models"
            for l in states_lags:
                assert type(l) is int and l >= 0, "State lags must be non-negative integers"

        # models
        self.models = models
        # state joining 
        self.states_join = states_join
        # state lags when joining
        self.states_lags = states_lags

        # Inherited properties
        self.size = len(self.models)
        # shapes
        self.N = [m.pars.N for m in self.models]
        self.K = [m.pars.K for m in self.models]
        self.M = sum(self.N)
        # joined states
        if not self.states_lags is None:
            self.states_N = [int(n * (1 + l)) for n, l in zip(self.N, self.states_lags)]
        else:
            self.states_N = self.N

    def fit(self, train_data, method, **kwargs):
        fit = method.fit(mfmodel=self, train_data=train_data, **kwargs)
        return fit

    def fitMultistep(self, train_data, method, **kwargs):
        fit = method.fitMultistep(mfmodel=self, train_data=train_data, **kwargs)
        return fit

    def fitDirectMultistep(self, train_data, method, **kwargs):
        fit = method.fitDirectMultistep(mfmodel=self, train_data=train_data, **kwargs)
        return fit

    def fitDirectHighFreq(self, train_data, method, **kwargs):
        fit = method.fitDirectHighFreq(mfmodel=self, train_data=train_data, **kwargs)
        return fit

    def print(self):
        for i, m in enumerate(self.models):
            table = m.pars.table()
            table.title = f"MFESN:  Component ESN #{i+1}"
            console.print(table)

