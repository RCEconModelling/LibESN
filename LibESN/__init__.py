r"""
# What is LibESN ?

An Echo State Network library.
"""

import warnings
import torch

from libesn.ufuncs import *
from libesn.matgen import *

# Set device at first call
PREFER_CPU = True

if PREFER_CPU and torch.cpu.is_available():
    warnings.warn(f"[  LibESN / torch backend set to 'cpu' (preferred)  ]")
    torch.set_default_device("cpu")
else:
    backend = torch.get_default_device()
    warnings.warn(f"[  LibESN / torch backend set to '{backend}' ]")