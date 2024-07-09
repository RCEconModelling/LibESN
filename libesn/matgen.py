import warnings
from typing import Type, Union

import torch

def __cast_float_or_none(x: Union[float, Type[None]]) -> Union[float, Type[None]]:
    if x is None:
        return None
    else:
        try:
            return float(x)
        except:
            raise ValueError(f"Could not cast {type(x)} to float")
        
def __torch_rand_to_sym_uniform(x: torch.tensor) -> torch.tensor:
    return (x * 2. - 1.)

class MatrixGenerator():
    def __init__(self, seed: Union[int, Type[None]]=None) -> None:
        self.Generator = torch.Generator()
        if not seed is None:
            self.Generator = self.Generator.manual_seed(seed)

    def __call__(
            self, 
            shape: tuple,
            measure: str, 
            **kwargs
        ) -> torch.tensor:
        # Check measure
        assert type(shape) is tuple or type(shape) is list, "shape must be a tuple or list"
        assert measure in ('normal', 'uniform', 'ortho'),  "measure must be one of 'normal', 'uniform', 'ortho'"

        # Load kwargs
        sparsity = kwargs.get('sparsity', None)
        normalize = kwargs.get('normalize', None)
        options = kwargs.get('options', None)
        seed = kwargs.get('seed', None)

        # Seed `self.Generator`
        if not seed is None:
            self.Generator.manual_seed(seed)
        else:
            self.Generator.seed()
            warnings.warn("Generator seed not explicitly set, using semi-random seed")
        
        # Generate
        rndmat = torch.empty(shape)
        if measure is 'normal':
            rndmat = torch.randn(shape, generator=self.Generator)
        elif measure is 'uniform':
            rndmat = __torch_rand_to_sym_uniform(torch.rand(shape, generator=self.Generator))
        elif measure is 'ortho':
            rndmat = torch.qr(torch.randn(shape, generator=self.Generator)).Q
        elif measure is 'takens':
            N1, N2 = shape
            M = 1
            if not options is None:
                M = options.get("M")
            if N1 == N2:
                rndmat = torch.eye((N1-1))
                rndmat = torch.hstack((rndmat, torch.zeros((N1-1, 1))))
                rndmat = torch.vstack((torch.zeros((1, N1)), rndmat))
                rndmat = torch.kron(torch.eye(M), rndmat)
            elif N1 > N2:
                rndmat = torch.hstack((1, torch.zeros(N1-1)))
                rndmat = torch.atleast_2d(rndmat).T
                rndmat = torch.kron(torch.eye(M), rndmat)
            else:
                raise ValueError("shape not comformable to Takens matrix form")
        elif measure is 'exp_takens':
            # TODO
            if True:
                raise ValueError("!!! TODO !!!")
            else:
                raise ValueError("shape not comformable to Takens matrix form")
        elif measure is 'augment_takens':
            # TODO
            if True:
                raise ValueError("!!! TODO !!!")
            else:
                raise ValueError("shape not comformable to Takens matrix form")
        
        # Sparsify
        assert type(sparsity) is float or type(sparsity) is type(None), "sparsity must be a float or None"
        sparsity = __cast_float_or_none(sparsity)
        
        if (sparsity < 0) or (sparsity > 1):
            raise ValueError("sparsity degree is not within [0,1)")
        if not sparsity is None and measure in ('ortho', 'takens', 'exp_takens', 'augment_takens'):
            raise ValueError(f"measure '{measure}' does not support sparsity")
        
        if not sparsity is None:
            sparsemask = torch.bernoulli(torch.full(shape, sparsity))
            rndmat = rndmat * sparsemask
        
        # Normalize
        if not normalize is None:
            if normalize is 'eigen':
                maxeigval = torch.max(torch.linalg.eigvals(rndmat)[0])
                rndmat = rndmat / maxeigval
            elif normalize is 'singular':
                maxsv = torch.max(torch.linalg.svdvals(rndmat))
                rndmat = rndmat / maxsv
            elif normalize is 'frobenius':
                frobnorm = torch.linalg.norm(rndmat, ord='fro')
                rndmat = rndmat / frobnorm
            elif normalize is 'norm2':
                norm2 = torch.linalg.norm(rndmat, ord=2)
                rndmat = rndmat / norm2
            elif normalize is 'max':
                maxval = torch.max(rndmat)
                rndmat = rndmat / maxval
            else:
                raise ValueError(f"normalize '{normalize}' not recognized")
            
        return rndmat
