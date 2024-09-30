import pytest
import torch
from wavesim.utilities import partition, combine
from . import allclose

"""Test of the utility functions."""


@pytest.mark.parametrize("size", [(5, 4, 6), (7, 15, 32), (3, 5, 6)])
@pytest.mark.parametrize("n_domains", [(1, 2, 3), (3, 3, 3), (2, 4, 1)])
@pytest.mark.parametrize("type", ['full', 'sparse', 'hybrid1', 'hybrid2'])
@pytest.mark.parametrize("expanded", [False, True])
def test_partition_combine(size: tuple[int, int, int], n_domains: tuple[int, int, int], type: str, expanded: bool):
    if expanded:
        x = torch.tensor(1.0, dtype=torch.complex64).expand(size)
    else:
        x = torch.randn(size, dtype=torch.complex64) + 1j * torch.randn(size, dtype=torch.complex64)
    if type == 'sparse':
        x[x.real < 0.5] = 0
        x = x.to_sparse()
    elif type == 'hybrid1':
        # select half of  the slices, make rest zero
        indices = torch.range(0, size[0] - 1, 2, dtype=torch.int64)  # construct indices for the other half
        values = x[0::2, :, :]
        x = torch.sparse_coo_tensor(indices.reshape(1, -1), values, size)
    elif type == 'hybrid2':
        indices0 = torch.range(0, size[0] - 1, 2, dtype=torch.int64)
        indices1 = torch.range(0, size[1] - 1, 2, dtype=torch.int64)
        i0, i1 = torch.meshgrid(indices0, indices1)
        indices = torch.stack((i0.reshape(-1), i1.reshape(-1)), dim=0)
        values = x[0::2, 0::2, :].reshape(-1, x.shape[2])
        x = torch.sparse_coo_tensor(indices, values, size)

    partitions = partition(x, n_domains)
    assert partitions.shape == n_domains

    combined = combine(partitions)
    assert allclose(combined, x)


@pytest.mark.parametrize("n_domains", [(0, 0, 0), (4, 3, 3)])
def test_partition_with_invalid_input(n_domains):
    array = torch.randn((3, 3, 3), dtype=torch.complex64)
    with pytest.raises(ValueError):
        partition(array, n_domains)
