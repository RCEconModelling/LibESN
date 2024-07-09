import warnings
from typing import Type, Union

import numpy as np
from scipy.linalg import block_diag

def __cast_float_or_none(x: Union[float, Type[None]]) -> Union[float, Type[None]]:
    if x is None:
        return None
    else:
        try:
            return float(x)
        except:
            raise ValueError(f"Could not cast {type(x)} to float")

def matrixGenerator(
        shape, 
        dist='normal',
        **kwargs
    ) -> np.ndarray:
    """
    Function to generate (random) matrices (e.g. state matrix A, input mask C)
    according to commonly used entry-wise or matrix-wise distributions.

    The required arguments are:

    + `shape` : tuple, dimensions of the matrix to generate. Currently, this is checked
                to be a 2-entry `list` or `tuple`, meaning `matrixGenerator()` will return
                a 2D matrix.
    + `dist`: type of matrix to generate. Currently we implement entry-wise `'normal'` (Gaussian) and
                `'uniform'` distributions, as well as `'orthogonal'`, `'takens'`, `'takens_exp'` (exponential
                Takens) and `'takens_augment'` (augmented Takens) matrix forms.

    Optional `kwargs`:

    + `sparsity` : degree of sparsity of the generated matrix. **Note that** `sparsity` is implemented
                    using a sparsity mask generated with `rng.binomial(n=1, p=sparsity, size=shape)`:
                    therefore, `sparsity = 1.0` implies a generally *fully dense* output. 
                    Ignored if `dist` is not an entry-wise matrix distribution
    + `normalize` : normalization to apply to the matrix:
                    + `'eigen'` / `'eig'` :            maximum absolute eigenvalue
                    + `'sv'` :                        maximum singular value
                    + `'norm2'` :                     spectral (L2) norm
                    + `'max'` / `'normS'` :            infinity (sup) norm  
                    + `'fro'` :                       Frobenius norm       
                    
    Arguments `sparsity` and `normalize` are ignored if `dist` is not an entry-wise matrix distribution.
    """

    # Check shape
    assert type(shape) is tuple or type(shape) is list, "shape must be a tuple or list"
    if (len(shape) > 2):
        raise ValueError("Shape tuple is larger than 2D")
    
    # Check distribution
    assert type(dist) is str, f"{dist} is not a valid string"

    # Load kwargs
    sparsity = kwargs.get('sparsity', None)
    normalize = kwargs.get('normalize', None)
    options = kwargs.get('options', None)
    seed = kwargs.get('seed', None)

    # Set seed
    if not seed is None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(None)
        warnings.warn("np.random.Generator seed not explicitly set, using semi-random seed")
    
    # Generate
    rndmat = np.empty(shape)
    if dist == 'normal':
        rndmat = rng.standard_normal(size=shape)
    elif dist == 'uniform':
        rndmat = rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'orthogonal':
        rndmat = np.linalg.svd(rng.standard_normal(size=shape))[0]
        normalize = None # force no normalization
    elif dist  == 'takens':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            rndmat = np.eye((N1-1))
            rndmat = np.hstack((rndmat, np.zeros((N1-1, 1))))
            rndmat = np.vstack((np.zeros((1, N1)), rndmat))
            rndmat = np.kron(np.eye(M), rndmat)
        elif N1 > N2:
            rndmat = np.hstack((1, np.zeros(N1-1)))
            rndmat = np.atleast_2d(rndmat).T
            id_matrix = np.eye(M)
            rndmat = np.kron(id_matrix, rndmat)
        else:
            raise ValueError("Shape not comformable to Takens form")
        normalize = None # force no normalization
    elif dist  == 'takens_exp':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            for j in range(1,M + 1):
                rndmat = np.eye((N1-1))
                np.fill_diagonal(rndmat, np.exp(-np.random.uniform(low=1, high=M, size=(1,1))*np.array(range(1,N1))))
                rndmat = np.hstack((rndmat, np.zeros((N1-1, 1))))
                rndmat = np.vstack((np.zeros((1, N1)), rndmat))
                if j == 1:
                    H_res = rndmat
                else:
                    H_res = block_diag(H_res, rndmat)
            rndmat = H_res
        elif N1 > N2:
            rndmat = np.hstack((1, np.zeros(N1-1)))
            rndmat = np.atleast_2d(rndmat).T
            id_matrix = np.eye(M)
            np.fill_diagonal(id_matrix, np.random.uniform(low=0, high=1, size=(1,M)))
            rndmat = np.kron(id_matrix, rndmat)
        else:
            raise ValueError("Shape not comformable to Takens form")
        normalize = None # force no normalization
    elif dist == 'takens_augment':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            rndmat = np.zeros((M*N1, M*N1))
            for m in range(M):
                def underdiag_f(j):
                    #return np.random.uniform(low=0, high=0.5, size=(j))
                    #return (M-j)/(M+1) * np.ones((j))
                    #return np.exp(-(N1-1-j)) * np.ones((j))
                    return np.exp(-np.arange(N1-1, N1-1-j, -1)**.5)
                    #return np.exp(-np.arange(j)/N1)
                # Progressive fill under-diagonals
                H_m = np.atleast_2d(underdiag_f(1))
                for j in range(1, N1):
                    Q_j = np.zeros((j, j))
                    Q_j[1:,:-1] = H_m
                    np.fill_diagonal(Q_j, underdiag_f(j))
                    H_m = Q_j
                # Largest under-diagonal is just 1s (normalization)
                # Q_j = np.zeros((N1-1, N1-1))
                # Q_j[1:,:-1] = H_m
                # np.fill_diagonal(Q_j, np.ones(N1-1))
                H_m = Q_j
                H_m = np.hstack((H_m, np.zeros((N1-1, 1))))
                H_m = np.vstack((np.zeros((1, N1)), H_m))
                rndmat[m*N1:(m+1)*N1,m*N1:(m+1)*N1] = H_m
        else:
            raise ValueError("Shape not comformable to Takens form")
        normalize = None # force no normalization
    else:
        raise ValueError(f"Matrix distribution {dist} is invalid")
    
    # Sparsify
    sparsity = __cast_float_or_none(sparsity)

    if not sparsity is None and dist in ('ortho', 'takens', 'takens_exp', 'takens_augment'):
        raise ValueError(f"measure '{dist}' does not support sparsity")
    
    if not sparsity is None:
        if (sparsity < 0) or (sparsity > 1):
            raise ValueError("sparsity degree is not within [0,1)")
        sparsemask = rng.binomial(n=1, p=sparsity, size=shape)
        rndmat = rndmat * sparsemask

    # Normalize
    if not normalize is None:
        if normalize in ['eigen', 'eig']:
            rndmat /= np.max(np.abs(np.linalg.eigvals(rndmat)))
        elif normalize == 'sv':
            rndmat /= np.max(np.linalg.svd(rndmat)[1])
        elif normalize == 'norm2':
            rndmat /= np.linalg.norm(rndmat, ord=2)
        elif normalize in ['max', 'normS']:
            rndmat /= np.linalg.norm(rndmat, ord=np.inf)
        elif normalize == 'frobenius':
            rndmat /= np.linalg.norm(rndmat, ord='fro')
        else:
            raise ValueError("Unknown normalization")

    return rndmat