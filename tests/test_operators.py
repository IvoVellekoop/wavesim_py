import pytest
from anysim import domain_operator
from utilities import full_matrix
import torch
from torch import tensor
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from . import random_vector, device, dtype, allclose
import matplotlib.pyplot as plt

""" Performs checks on the operators represented as matrices (accretivity, norm). """

parameters = [
    {'n_size': (1, 1, 12), 'n_domains': None, 'n_boundary': 0, 'periodic': (False, False, True)},
    {'n_size': (1, 1, 12), 'n_domains': (1, 1, 1), 'n_boundary': 0, 'periodic': (False, False, True)},
    {'n_size': (1, 1, 12), 'n_domains': (1, 1, 1), 'n_boundary': 5, 'periodic': (True, True, False)},
    {'n_size': (1, 1, 32), 'n_domains': (1, 1, 2), 'n_boundary': 5, 'periodic': (True, True, True)},
    {'n_size': (1, 1, 32), 'n_domains': (1, 1, 2), 'n_boundary': 5, 'periodic': (True, True, False)},
    {'n_size': (1, 32, 1), 'n_domains': (1, 2, 1), 'n_boundary': 5, 'periodic': (True, True, True)},
    {'n_size': (1, 32, 1), 'n_domains': (1, 2, 1), 'n_boundary': 5, 'periodic': (True, False, True)},
    {'n_size': (32, 1, 1), 'n_domains': (2, 1, 1), 'n_boundary': 5, 'periodic': (False, True, True)},
    {'n_size': (32, 1, 1), 'n_domains': (2, 1, 1), 'n_boundary': 5, 'periodic': (True, True, True)},
    # test different-sized domains
    {'n_size': (23, 1, 1), 'n_domains': (3, 1, 1), 'n_boundary': 2, 'periodic': (True, True, True)},
    {'n_size': (23, 1, 1), 'n_domains': (3, 1, 1), 'n_boundary': 3, 'periodic': (False, True, True)},
    {'n_size': (1, 5, 19), 'n_domains': (1, 1, 2), 'n_boundary': 3, 'periodic': (True, True, True)},
    {'n_size': (1, 14, 19), 'n_domains': (1, 2, 2), 'n_boundary': 3, 'periodic': (True, False, True)},
    {'n_size': (17, 30, 1), 'n_domains': (2, 3, 1), 'n_boundary': 3, 'periodic': (True, True, True)},
    {'n_size': (12, 12, 12), 'n_domains': (2, 2, 2), 'n_boundary': 3, 'periodic': (False, False, True)},
    {'n_size': (18, 24, 18), 'n_domains': (2, 3, 2), 'n_boundary': 3, 'periodic': (True, True, True)},
    {'n_size': (17, 23, 19), 'n_domains': (2, 3, 2), 'n_boundary': 3, 'periodic': (True, True, True)},
]


def construct_domain(n_size, n_domains, n_boundary, periodic=(False, False, True)):
    """ Construct a domain or multi-domain"""
    torch.manual_seed(12345)
    n = torch.rand(n_size, dtype=dtype, device=device) + 1.0  # random refractive index between 1 and 2
    n.imag = 0.1 * torch.maximum(n.imag, tensor(0.0))  # a positive imaginary part of n corresponds to absorption
    if n_domains is None:  # single domain
        return HelmholtzDomain(refractive_index=n, pixel_size=0.25, periodic=periodic, n_boundary=n_boundary)
    else:
        return MultiDomain(refractive_index=n, pixel_size=0.25, periodic=periodic, n_boundary=n_boundary,
                           n_domains=n_domains)


@pytest.mark.parametrize("params", parameters)
def test_operators(params):
    """ Check that operator definitions are consistent:
        - forward = inverse_propagetor - medium: A= L + 1 - B
        - preconditioned_operator = preconditioned(operator)
        - richardson = x + α (Γ⁻¹b - Γ⁻¹A x)
    """
    domain = construct_domain(**params)
    x = random_vector(domain.shape)
    B = domain_operator(domain, 'medium')
    L1 = domain_operator(domain, 'inverse_propagator')
    A = domain_operator(domain, 'forward')
    Ax = A(x)
    assert allclose(domain.scale * Ax, L1(x) - B(x))

    Γ = domain_operator(domain, 'preconditioner')
    ΓA = domain_operator(domain, 'preconditioned_operator')
    ΓAx = ΓA(x)
    assert allclose(ΓAx, Γ(Ax))

    α = 0.1
    b = random_vector(domain.shape)
    Γb = Γ(b)
    domain.set_source(b)
    M = domain_operator(domain, 'richardson', alpha=α)
    assert allclose(M(0), α * Γb)

    residual = Γb - ΓAx
    assert allclose(M(x), x + α * residual)


@pytest.mark.parametrize("params", parameters)
def test_accretivity(params):
    """ Checks norm and lower bound of real part for various operators

     B (medium) should have real part between -0.05 and 1.0 (if we don't put the absorption in V0. If we do, the upper limit may be 1.95)
        The operator B-1 should have a norm of less than 0.95

    L + 1 (inverse propagator) should be accretive with a real part of at least 1.0
    (L+1)^-1 (propagator) should be accretive with a real part of at least 0.0
    A (forward) should be accretive with a real part of at least 0.0
    ΓA (preconditioned_operator) should be such that 1-ΓA is a contraction (a norm of less than 1.0)
     """
    domain = construct_domain(**params)
    domain.set_source(0)
    assert_accretive(domain_operator(domain, 'medium'), 'B', real_min=0.05, real_max=1.0, norm_max=0.95,
                     norm_offset=1.0)
    assert_accretive(domain_operator(domain, 'inverse_propagator'), 'L + 1', real_min=1.0)
    assert_accretive(domain_operator(domain, 'propagator'), '(L + 1)^-1', real_min=0.0)
    assert_accretive(domain_operator(domain, 'preconditioned_operator'), 'ΓA', norm_max=1.0, norm_offset=1.0)
    assert_accretive(domain_operator(domain, 'richardson', alpha=0.75), '1- α ΓA', norm_max=1.0)
    assert_accretive(domain_operator(domain, 'forward'), 'A', real_min=0.0, pre_factor=domain.scale)


def assert_accretive(operator, name, *, real_min=None, real_max=None, norm_max=None, norm_offset=None, pre_factor=None):
    """Helper function to check if an operator is accretive, and to compute the norm around a given offset.
    This function constructs a full matrix from the operator, so it only works if the domain is not too large.
    """
    M = full_matrix(operator)
    if pre_factor is not None:
        M *= pre_factor

    if norm_max is not None:
        if norm_offset is not None:
            M.diagonal().add_(-norm_offset)
        norm = torch.linalg.norm(M, ord=2)
        print(f'norm {norm:.2e}')
        assert norm <= norm_max, f'operator {name} has norm {norm} > {norm_max}'

    if real_min is not None or real_max is not None:
        M.add_(M.mH)
        eigs = torch.linalg.eigvalsh(M)
        if norm_offset is not None:
            eigs.add_(norm_offset)
        if real_min is not None:
            acc = eigs.min()
            print(f'acc {acc:.2e}')
            assert acc >= real_min, f'operator {name} is not accretive, min λ_(A+A*) = {acc} < {real_min}'
        if real_max is not None:
            acc = eigs.max()
            print(f'acc {acc:.2e}')
            assert acc <= real_max, f'operator {name} has eigenvalues that are too large, max λ_(A+A*) = {acc} > {real_max}'
