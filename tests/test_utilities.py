import pytest
import torch
from utilities import partition, combine
from . import allclose

"""Test of the utility functions"""


@pytest.mark.parametrize("size", [(5, 4, 6), (7, 15, 32), (3, 5, 6)])
@pytest.mark.parametrize("n_domains", [(1, 2, 3), (3, 3, 3), (2, 4, 1)])
@pytest.mark.parametrize("sparse", [False, True])
def test_partition_combine(size: tuple[int, int, int], n_domains: tuple[int, int, int], sparse: bool):
    x = torch.randn(size, dtype=torch.complex64) + 1j * torch.randn(size, dtype=torch.complex64)
    if sparse:
        x[x.real < 0.5] = 0
        x = x.to_sparse()

    partitions = partition(x, n_domains)
    assert partitions.shape == n_domains

    combined = combine(partitions)
    assert allclose(combined, x)


@pytest.mark.parametrize("n_domains", [(0, 0, 0), (4, 3, 3)])
def test_partition_with_invalid_input(n_domains):
    array = torch.randn((3, 3, 3), dtype=torch.complex64)
    with pytest.raises(ValueError):
        partition(array, n_domains)
