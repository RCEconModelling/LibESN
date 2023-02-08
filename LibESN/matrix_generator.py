#
# LibESN
# A better ESN library
#
# Current version: ?
# ================================================================

import re
import numpy as np
from scipy.linalg import block_diag

def matrixGenerator(shape, dist='normal', sparsity=None, normalize=None, options=None, seed=None):
    # matrixGenerator
    #   Function to generate (random) matrices (e.g. state matrix A, input mask C)
    #   according to commonly used entry-wise or matrix-wise distributions.
    #
    #       shape       tuple, dimensions of the matrix to generate
    #       dist        type of matrix to generate
    #       sparsity    degree of sparsity (~ proportion of 0 elements)
    #                   of the generated matrix. Ignored if 'type' does
    #                   not have a 'sparse_' prefix
    #       normalize   normalization to apply to the matrix:
    #                       'eig'       maximum absolute eigenvalue
    #                       'sv'        maximum singular value
    #                       'norm2'     spectral norm
    #                       'normS'     infinity (sup) norm         
    #  

    # Set seed
    if not seed is None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(12345)

    if (shape is None) or (not type(shape) is tuple):
        raise ValueError("No shape selected")
    if (len(shape) > 2):
        raise ValueError("Shape tuple is larger than 2D")
    if dist is None:
        dist = 'normal'
    if sparsity is None:
        if re.findall('^sparse', dist):
            raise ValueError('Sparse distributions require choosing sparsity degree')
        sparsity = 1.
    else:
        if (sparsity < 0) or (sparsity >= 1):
            raise ValueError("Chosen sparsity degree is not within [0,1)")
    
    # Generate
    H = np.empty(shape)
    if dist == 'normal':
        H = rng.standard_normal(size=shape)
    elif dist == 'uniform':
        H = rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'sparse_normal':
        B = rng.binomial(n=1, p=sparsity, size=shape)
        H = B * rng.standard_normal(size=shape)
    elif dist == 'sparse_uniform':
        B = rng.binomial(n=1, p=sparsity, size=shape)
        H = B * rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'orthogonal':
        H = np.linalg.svd(rng.standard_normal(size=shape))[0]
        # Ignore normalization
        normalize = None
    elif dist  == 'takens':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            H = np.eye((N1-1))
            H = np.hstack((H, np.zeros((N1-1, 1))))
            H = np.vstack((np.zeros((1, N1)), H))
            H = np.kron(np.eye(M), H)
        elif N1 > N2:
            H = np.hstack((1, np.zeros(N1-1)))
            H = np.atleast_2d(H).T
            id_matrix = np.eye(M)
            H = np.kron(id_matrix, H)
        else:
            raise ValueError("Shape not comformable to Takens form")
    elif dist  == 'takens_exp':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            for j in range(1,M + 1):
                H = np.eye((N1-1))
                np.fill_diagonal(H, np.exp(-np.random.uniform(low=1, high=M, size=(1,1))*np.array(range(1,N1))))
                H = np.hstack((H, np.zeros((N1-1, 1))))
                H = np.vstack((np.zeros((1, N1)), H))
                if j == 1:
                    H_res = H
                else:
                    H_res = block_diag(H_res, H)
            H = H_res
        elif N1 > N2:
            H = np.hstack((1, np.zeros(N1-1)))
            H = np.atleast_2d(H).T
            id_matrix = np.eye(M)
            np.fill_diagonal(id_matrix, np.random.uniform(low=0, high=1, size=(1,M)))
            H = np.kron(id_matrix, H)
        else:
            raise ValueError("Shape not comformable to Takens form")
    elif dist == 'takens_augment':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            H = np.zeros((M*N1, M*N1))
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
                H[m*N1:(m+1)*N1,m*N1:(m+1)*N1] = H_m
        else:
            raise ValueError("Shape not comformable to Takens form")
    else:
        raise ValueError("Unknown matrix distribution or type")

    # Normalize
    if not normalize is None:
        if normalize == 'eig':
            H /= np.max(np.abs(np.linalg.eigvals(H)))
        elif normalize == 'sv':
            H /= np.max(np.linalg.svd(H)[1])
        elif normalize == 'norm2':
            H /= np.linalg.norm(H, ord=2)
        elif normalize == 'normS':
            H /= np.linalg.norm(H, ord=np.inf)
        else:
            raise ValueError("Unknown normalization")

    return H