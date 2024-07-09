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

For more info, see the paper [Reservoir Computing for Macroeconomic Forecasting with Mixed Frequency Data](https://arxiv.org/abs/2211.00363), its [repository](https://github.com/RCEconModelling/Reservoir-Computing-for-Macroeconomic-Modelling) and references below.

**NOTE:** LibESN works best with *data in `pandas.DataFrame` format and "datetime" indexing*, especially when dealing with MFESN models.
Our initial goal for LibESN was to make it easy to handle economic data sampled at multiple (yearly, quarterly, monthly, etc) frequencies.
"""