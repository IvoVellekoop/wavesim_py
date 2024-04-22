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

    # compute the size of a single ULP
    exponent = torch.max(a.abs().log2().ceil()).item()
    ulp = torch.finfo(a.dtype).eps * 2 ** exponent

    # error should be within 100 ULPs.
    # This corresponds to a relative error of 1e-5 for float32 and 1e-12 for float64
    # (we usually do quite some back-and-forth ffts, which may cause errors to accumulate)
    return torch.allclose(a, b, atol=100 * ulp), f"Arrays are not close within 100 ULPs"
