import torch
from torch import tensor


def allclose(a, b):
    if not torch.is_tensor(a):
        a = tensor(a, dtype=b.dtype)
    if not torch.is_tensor(b):
        b = tensor(b, dtype=a.dtype)
    if a.dtype != b.dtype:
        a = a.astype(b.dtype)
    if a.device != b.device:
        a = a.to('cpu')
        b = b.to('cpu')
    a = a.to_dense()
    b = b.to_dense()
    return torch.allclose(a, b)
