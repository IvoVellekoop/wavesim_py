import pytest
from wavesim.multidomain import MultiDomain
from wavesim.helmholtzdomain import HelmholtzDomain
import torch
from torch import tensor
from . import allclose

""" Performs a set of basic consistency checks for the Domain class and the HelmholtzBase multi-domain class. """

device = "cuda:0"
dtype = torch.complex64


def construct_domain(n_size, n_domains, n_boundary, periodic=(False, False, True)):
    """ Construct a domain or multi-domain"""
    n = torch.rand(n_size, dtype=dtype, device=device) + 1.0  # random refractive index between 1 and 2
    n.imag = 0.1 * torch.maximum(n.imag, tensor(0.0))
    if n_domains is None:  # single domain
        return HelmholtzDomain(refractive_index=n, pixel_size=0.25, periodic=periodic, n_boundary=n_boundary)
    else:
        return MultiDomain(refractive_index=n, pixel_size=0.25, periodic=periodic, n_boundary=n_boundary,
                           n_domains=n_domains)


def construct_source(n_size):
    """ Construct a sparse-matrix source with some points at the corners and in the center"""
    locations = tensor([
        [n_size[0] // 2, 0, n_size[0] - 1],
        [n_size[1] // 2, 0, 0],
        [n_size[2] // 2, 0, 0]])

    return torch.sparse_coo_tensor(locations, tensor([1, 1, 1]), n_size, dtype=torch.complex64)


def random_vector(n_size):
    """Construct a random vector for testing operators"""
    return torch.randn(n_size, device=device) + 1.0j * torch.randn(n_size, device=device)


@pytest.mark.parametrize("n_size", [(128, 100, 93), (50, 49, 1)])
@pytest.mark.parametrize("n_domains", [None, (1, 1, 1), (3, 2, 1)])
def test_basics(n_size: tuple[int, int, int], n_domains: tuple[int, int, int] | None):
    """Tests the basic functionality of the Domain and MultiDomain classes

    Tests constructing a domain, subdividing data over subdomains,
    concatenating data from the subdomains, adding sparse sources,
    and computing the inner product.
    """

    # construct the (multi-) domain operator
    domain = construct_domain(n_size, n_domains, n_boundary=8)

    # test coordinates
    assert domain.shape == n_size
    for dim in range(3):
        coordinates = domain.coordinates(dim)
        assert coordinates.shape[dim] == n_size[dim]
        assert coordinates.numel() == n_size[dim]

        coordinates_f = domain.coordinates_f(dim)
        assert coordinates_f.shape == coordinates.shape
        assert coordinates_f[0, 0, 0] == 0

        if n_size[dim] > 1:
            assert allclose(coordinates.flatten()[1] - coordinates.flatten()[0], domain.pixel_size)
            assert allclose(coordinates_f.flatten()[1] - coordinates_f.flatten()[0],
                            2.0 * torch.pi / (n_size[dim] * domain.pixel_size))

    # construct a random vector for testing operators
    x = random_vector(n_size)
    y = random_vector(n_size)

    # perform some very basic checks
    # mainly, this tests if the partitioning and composition works correctly
    domain.set(0, x)
    domain.set(1, y)
    assert x.device == domain.device
    assert allclose(domain.get(0), x)
    assert allclose(domain.get(1), y)

    inp = domain.inner_product(0, 1)
    assert torch.isclose(inp, torch.vdot(x.flatten(), y.flatten()))

    # construct a source and test adding it
    domain.clear(0)
    assert allclose(domain.get(0), 0.0)
    source = construct_source(n_size)
    domain.set_source(source)
    domain.add_source(0)
    domain.add_source(0)
    assert allclose(domain.get(0), 2.0 * source * domain.scale)

    # test mixing: α x + β y
    # make sure to include the special cases α=0, β=0, α=1, β=1 and α+β=1
    # since they may be optimized and thus have different code paths
    domain.set(0, x)
    for alpha in [0.0, 1.0, 0.25]:
        for beta in [0.0, 1.0, 0.75]:
            domain.set(1, y)
            domain.mix(alpha, 0, beta, 1, 0)
            assert allclose(domain.get(0), alpha * x + beta * y)


@pytest.mark.parametrize("n_size", [(128, 100, 93), (50, 49, 1)])
@pytest.mark.parametrize("n_domains", [None, (1, 1, 1), (3, 2, 1)])
def test_propagator(n_size: tuple[int, int, int], n_domains: tuple[int, int, int] | None):
    """Tests the forward and inverse propagator

    The wavesim algorithm only needs the propagator (L+1)^(-1) to be implemented.
    For testing, and for evaluating the final residue, the Domain and MultiDomain classes
    also implement the 'inverse propagator L+1', which is basically the homogeneous
    part of the forward operator A.

    This test checks that the forward and inverse propagator are consistent, namely
    (L+1)^(-1) (L+1) x = x.
    todo: check if the operators are actually correct (not just consistent)
    Note that the propagator is domain-local, so the wrapping correction and domain
    transfer functions are not tested here.
    """

    # construct the (multi-) domain operator
    domain = construct_domain(n_size, n_domains, n_boundary=8)

    # assert that (L+1) (L+1)^-1 x = x
    x = random_vector(n_size)
    domain.set(0, x)
    domain.propagator(0, 0)
    domain.inverse_propagator(0, 0)
    x_reconstructed = domain.get(0)
    assert allclose(x, x_reconstructed)

    # also assert that (L+1)^-1 (L+1) x = x, use different slots for input and output
    domain.set(0, x)
    domain.inverse_propagator(0, 1)
    domain.propagator(1, 1)
    x_reconstructed = domain.get(1)
    assert allclose(x, x_reconstructed)

    # for the non-decomposed case, test if the propagator gives the correct value
    if n_domains is None:
        k = 2 * torch.pi * tensor((3.0, 5.0, 7.0)) / tensor(n_size)  # in 1/pixels
        plane_wave = torch.exp(1j * (
                k[0] * torch.arange(n_size[0], device=device).reshape(-1, 1, 1) +
                k[1] * torch.arange(n_size[1], device=device).reshape(1, -1, 1) +
                k[2] * torch.arange(n_size[2], device=device).reshape(1, 1, -1)))
        domain.set(0, plane_wave)
        domain.inverse_propagator(0, 0)
        result = domain.get(0)
        laplace_kernel = - (k[0] ** 2 + k[1] ** 2 + k[2] ** 2) * domain.pixel_size ** 2
        correct_result = (1.0 + domain.scale * (laplace_kernel + domain.shift)) * plane_wave  # L+1 =  scale·∇² + 1.
        assert allclose(result, correct_result)


def test_wrapped_propagator():
    """Tests the inverse propagator L+1 with wrapping corrections

    This test compares the situation of a single large domain to that of a multi-domain.
    If the wrapping and transfer corrections are implemented correctly, the results should be the same
    up to the difference in scaling factor.
    """
    n_size = (128, 100, 93)
    domain_single = construct_domain(n_size, n_domains=None, n_boundary=8, periodic=(True, False, True))
    domain_multi = construct_domain(n_size, n_domains=(3, 3, 3), n_boundary=8, periodic=(False, False, False))
    source = construct_source(n_size)

    for domain in [domain_single, domain_multi]:
        V = 0
        L1 = 1
        domain.clear(0)
        domain.set_source(source)
        domain.add_source(0)
        domain.inverse_propagator(0, L1)  # (L+1) y
        domain.medium(0, V)  # (1-V) y
        domain.mix(1.0, L1, -1.0, V, 0)  # (L+V) y

    x_single = domain_single.get(0)
    x_multi = domain_multi.get(0)
    assert allclose(x_single, x_multi)
