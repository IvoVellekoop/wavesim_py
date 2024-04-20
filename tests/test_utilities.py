import pytest
import torch
from utilities import partition

"""Test of the utility functions"""


@pytest.mark.parametrize("size", [(1, 2, 3), (7, 15, 32), (1, 5, 6)])
@pytest.mark.parametrize("n_domain", [(1, 2, 3), (7, 15, 32), (2, 1, 1)])
def test_partition_combine(size, n_domains):
    x = torch.randn(size, dtype=torch.complex64) + 1j * torch.randn(size, dtype=torch.complex64)
    parts = partition(x, n_domains)
    x_reconstructed = torch.cat(parts, dim=0)


@pytest.mark.parametrize("n_domains", [(0, 0, 0), (4, 3, 3)])
def test_partition_with_invalid_input(n_domains):
    array = torch.randn((3, 3, 3), dtype=torch.complex64)
    with pytest.raises(ValueError):
        partition(array, n_domains)
