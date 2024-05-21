import torch
from torch import tensor

device = "cuda"  # "cpu"  # "cuda:0"
dtype = torch.complex64


def allclose(a, b, rtol=0.0, atol=0.0, ulptol=100):
    """Check if two tensors are close to each other.

    Condition: |a-b| <= atol + rtol * maximum(|b|,|a|) + ulptol * ulp
    Where ulp is the size of the smallest representable difference between two numbers of magnitude ~max(|b[...]|)
    """

    # make sure that a and b are tensors of the same dtype and device
    if not torch.is_tensor(a):
        a = tensor(a, dtype=b.dtype)
    if not torch.is_tensor(b):
        b = tensor(b, dtype=a.dtype)
    if a.dtype != b.dtype:
        a = a.type(b.dtype)
    if a.device != b.device:
        a = a.to('cpu')
        b = b.to('cpu')
    a = a.to_dense()
    b = b.to_dense()

    # compute the size of a single ULP
    ab_max = torch.maximum(a.abs(), b.abs())
    exponent = ab_max.max().log2().ceil().item()
    ulp = torch.finfo(b.dtype).eps * 2 ** exponent
    tolerance = atol + rtol * ab_max + ulptol * ulp
    diff = (a - b).abs()

    if (diff - tolerance).max() <= 0.0:
        return True
    else:
        abs_err = diff.max().item()
        rel_err = (diff / ab_max).max()
        print(f"\nabsolute error {abs_err} = {abs_err / ulp} ulp\nrelative error {rel_err}")
        return False


def random_vector(n_size):
    """Construct a random vector for testing operators"""
    return torch.randn(n_size, device=device, dtype=dtype) + 1.0j * torch.randn(n_size, device=device, dtype=dtype)


def random_refractive_index(n_size):
    """Construct a random refractive index between 1 and 2 with a small positive imaginary part

    The sign of the imaginary part is such that the imaginary part of n² is positive
    """
    n = (1.0 + torch.rand(n_size, device=device, dtype=dtype) +
         0.1j * torch.rand(n_size, device=device, dtype=dtype))

    # make sure that the imaginary part of n² is positive
    mask = (n ** 2).imag < 0
    n.imag[mask] *= -1.0
    return n
