# LibESN

![version badge](https://img.shields.io/badge/version-0.2-blue)

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
+ Reservoir matrix generations utility

**IMPORTATN NOTICE:** as of 07/2024, LibESN is being consistently refactored to improve peformance,
code readibility and functionality. See below for breaking changes!

## How to...?

You can **[read the full documentation here.](https://rceconmodelling.github.io/LibESN/)**.

<sub>(The documentation is automatically built from the latest main commit using [pdoc](https://github.com/mitmproxy/pdoc))</sub>

## News

+ 09/07/2024 - First commit of `v0.2` refactoring. The following are breaking changes:
  + Library name is now `libesn`, not `LibESN`, to better conform to Python conventions
  + `base_datetime` submodule has been changed to `datetime`
  + `base_utils` submodule has been split:
    + Data utility functions are now in the `datautils` submodule
    + `ShiftTimeSeriesSplit` cross-validation class is now in the `validation` submodule
  + `base_functions` submodule has been changed to `ufuncs`
  + `matrix_generator` submodule has been changed to `matgen`
    + `matrixGenerator()` has been signifcantly changed. In particular, `dist`
        doest not accept `sparse_` options - the `sparsity` optional argument
        automatically handles sparseness of entry-wise distributions (see docs)
  + [Documentation](https://rceconmodelling.github.io/LibESN/) pages are officially available, but *very early stage*

## References

LibESN is based on the Python codebased originally developed for the paper "*Reservoir Computing for Macroeconomic Forecasting with Mixed Frequency Data*" (UKRI funded project, Ref: ES/V006347/1), available at the following links:

+ [International Journal of Forecasting](https://doi.org/10.1111/jtsa.12737) (Open Access)

+ [ArXiv](https://arxiv.org/abs/2211.00363)

+ [ResearchGate](https://www.researchgate.net/publication/364957371_Reservoir_Computing_for_Macroeconomic_Forecasting_with_Mixed_Frequency_Data)
