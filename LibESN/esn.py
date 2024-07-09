"""
Main ESN model object class.
"""

import warnings
from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from libesn.matgen import MatrixGenerator
from libesn.datetime import *

def __prepare_state_params(
        dim0: int, 
        dim1: int, 
        x: Union[torch.tensor, dict], 
        name: str
    ) -> torch.tensor:
    assert type(x) in [dict, Type(None), torch.tensor], (
        f"{name} must be None, a dictionary defining a valid MatrixGenerator() spec or a torch.tensor"
    )
    if x is None:
        warnings.warn(f"{name} not set, using random uniform initialization")
        return torch.rand(dim0, dim1)
    elif type(x) is dict:
        if 'shape' in x:
            assert x['shape'][0] == dim0
            assert x['shape'][1] == dim1
            return MatrixGenerator()(shape=(dim0, dim1), **x)
        else:
            return MatrixGenerator()(**x)
    else:
        return x

class ESN(nn.Module):
    def __init__(self, input_size: int, state_size: int, output_size: int, **kwargs) -> None:
        super(ESN, self).__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        A = kwargs.get('A', None)
        C = kwargs.get('C', None)
        zeta = kwargs.get('zeta', None)
        rho = kwargs.get('rho', None)
        gamma = kwargs.get('gamma', None)
        leak = kwargs.get('leak', None)

        # Load state tensors
        self.A = __prepare_state_params(self.state_size, self.state_size, A, 'A')
        self.C = __prepare_state_params(self.state_size, self.input_size, C, 'C')
        self.zeta = __prepare_state_params(self.state_size, 1, zeta, 'zeta')

        # Load hyperparameters
        if rho is None:
            warnings.warn("rho not set, using default value of 0")
            self.rho = 0
        if gamma is None:
            warnings.warn("gamma not set, using default value of 1")
            self.gamma = 1
        if leak is None:
            warnings.warn("leak not set, using default value of 0")
            self.leak = 0

        # Layers
        self.in2state = nn.Linear(input_size, state_size, bias=False)
        self.state2state = nn.Linear(state_size, state_size)
        self.state2out = nn.Linear(state_size, output_size)
            
        # Initialize weights
        with torch.no_grad():
            self.in2state.weight.copy_(self.C * self.gamma)
            self.state2state.weight.copy_(self.A * self.rho)
            self.state2out.bias.copy_(self.zeta)

    def forward(self, input, state):
        with torch.no_grad():
            state = self.leak * state + (1 - self.leak) * F.tanh(self.state2state(state) + self.in2state(input))
        output = self.state2out(state)
        return output, state

    def initState(self):
        return torch.zeros(1, self.state_size)