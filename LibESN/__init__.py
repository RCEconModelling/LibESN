r"""
# What is LibESN ?

An Echo State Network library for single and multi-frequency ESN (MFESN) forecasting.

Currently we implement:

+ ESN (single reservoir, single frequency):
  + `ESN` class
  + Ridge CV and fitting, with direct and iterated multistep fit methods
  + Simple forecasting, with direct and iterated multistep forecasting methods
+ MFESN (multiple reservoirs, multi-frequency):
  + `MFESN` class
  + Ridge CV and fitting, with high-frequency (nowcasting) fit methods
  + Simple MF forecasting, with high-frequency nowcasting and forecasting methods
+ Reservoir matrix generation utilities

For more info, see the paper *[Reservoir Computing for Macroeconomic Forecasting with Mixed Frequency Data](https://doi.org/10.1016/j.ijforecast.2023.10.009)*, 
its [repository](https://github.com/RCEconModelling/Reservoir-Computing-for-Macroeconomic-Modelling).

**NOTE:** LibESN works best with *data in `pandas.DataFrame` format and "datetime" indexing*, especially when dealing with MFESN models.
Our initial goal for LibESN was to make it easy to handle economic data sampled at multiple (yearly, quarterly, monthly, etc) frequencies.

# Example Notebooks

To jump right into working code or to have an idea of how LibESN can be used in practice, we provide a few practical examples of the core functionality in the `examples` folder:

+ Autonomous prediction of the Lorenz attractor ([Jupyter Notebook](https://github.com/RCEconModelling/LibESN/blob/main/examples/example_lorenz.ipynb))

# Basic Usage

LibESN provides two main classes for working with ESN models, `ESN` and `MFESN`.

- `ESN` constructs a simple (nonlinear) ESN model
- `MFESN` constructs a multi-frequency, multi-reservoir MFESN model, see the references for more information.

LibESN is designed with a focus on implementing `MFESN` models, which are able to integrate data of multiple frequencies/releases. These type of mixed-frequency models are commonly and widely used for economic forecasting and nowcasting.

## Creating an ESN model

```python
from libesn.matgen import matrixGenerator
from libesn.esn import ESN

N = 50
K = 3

esn = ESN(
    None,
    smap=np.tanh, 
    A=matrixGenerator((N, N), dist='uniform', sparsity=0.2, seed=1234), 
    C=matrixGenerator((N, K), dist='uniform', sparsity=0.2, seed=12345), 
    zeta=np.zeros((N, 1)), 
    rho=0.8, 
    gamma=1, 
    leak=0.5,
)
```
ESN models consist of a **state equation** and a **readout**/**regression equation**, formally

$$
\begin{aligned}
    X_t &= \alpha X_{t-1} + (1 - \alpha) \sigma(\rho A X_{t-1} + \gamma C Z_t + \zeta) \\\
    Y_{t} &= W' X_t + \eta_{t}
\end{aligned}
$$


The `matrixGenerator()` function allows us to draw random matrices from a set of commonly used (sparse) random and non-random distributions.

## Composing a Multi-Frequency ESN model

MFESNs can be easily composed starting from individual `ESN` model objects:

```python
from LibESN.mfesn import MFESN

mfesn = MFESN((esn_A, esn_B))
```

## Fitting

Fitting is easily done with both `ESN` and `MFESN` models. For example, we can use ridge regression with cross-validation to fit the MFESN model above:

```python
from LibESN.mfesn_fit import mfRidgeFit, mfRidgeCV

mfesn_simple = MFESN((esn_A, ))

mfcv = mfRidgeCV().cvHighFreq(
    mfmodel=mfesn_simple,
    train_data=((data_A_train, ), data_Target_train,),
    steps=[0,1], # CV at HF for both nowcast and 1-step-ahead
    freqratio=3,
    min_train_size=100,
    test_size=5,
    verbose=True,
)

Lambda = [
    mfcv['cvLambda'][0]['L'],
    mfcv['cvLambda'][1]['L']
]

mfesn_fit = mfesn_simple.fitDirectHighFreq(
    train_data=((data_A_train, ), data_Target_train,), 
    method=mfRidgeFit(Lambda_hf),
    steps=[0,1],
    freqratio=3,
)
```

Note:

- We assume above that the data `data_A_train` is available in higher frequency compared to target in `data_Target_train`
- The frequency ratio of inputs to targets must be set with option `freqratio`, and above we chose `freqratio=3`: this is case for examples when inputs are monthly series and targets are quarterly series
- `cvHighFreq()` implements "high frequency" cross-validation, that is, it will cross-validate a specific ridge penalty (scalar) for each of high-frequency steps in the set $1, 2, ..., r$ where here $r =$`freqratio`
- Option `steps=[0,1]` is used to validate and fit coefficients for both *nowcasting* (step 0) and 1-step-ahead *forecasting* (step 1)
- Function `fitDirectHighFreq()` handles fitting specific coefficient matrices for both the nowcasting and forecasting (1 step) setup. This means output `dhf_fit` will contain `len(steps)*freqratio` sets of coefficients

## Forecasting

```python
from LibESN.mfesn_forecast import mfDirectHighFreqForecast

forecast = mfDirectHighFreqForecast(
    mfmodel=mfesn_simple,
    forecast_data=(data_A_test,),
    fit=mfesn_fit,
    freqs=['MS',], # monthly data
    terminal=True, # only return forecast with most recent data
)
```

## Nowcasting

```python
from LibESN.mfesn_forecast import mfNowcast

nowcast = mfNowcast(
    mfmodel=mfesn_simple,
    nowcast_data=(data_A_test,),
    fit=mfesn_fit,
    freqs=['MS',], # monthly data
    terminal=True, # only return forecast with most recent data
)
```

"""

from libesn.ufuncs import *
from libesn.datautils import *
from libesn.matgen import matrixGenerator