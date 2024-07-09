import torch

def identity(A):
    return A

def radbas(A):
    return torch.exp(-A**2)

def retanh(A):
    return torch.tanh(torch.maximum(A, 0))

def softplus(A):
    return torch.log(1 + torch.exp(A))