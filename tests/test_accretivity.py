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
    """ Check that operator A = L + 1 - B  """
    domain = construct_domain(**params)
    x = random_vector(domain.shape)
    B = domain_operator(domain, 'medium')
    L1 = domain_operator(domain, 'inverse_propagator')
    A = domain_operator(domain, 'forward')
    y = A(x)
    assert allclose(y, L1(x) - B(x))

    Γ = domain_operator(domain, 'preconditioner')
    ΓA = domain_operator(domain, 'preconditioned_operator')
    Γy1 = Γ(y)
    Γy2 = ΓA(x)
    assert allclose(Γy1, Γy2)


@pytest.mark.parametrize("params", parameters)
def test_accretivity(params):
    """ Check that operator A = L + V is accretive, i.e., has a non-negative real part """
    domain = construct_domain(**params)
    assert_accretive(domain_operator(domain, 'medium'), 'B', real_min=0.05, real_max=1.0, norm_max=0.95,
                     norm_offset=1.0)
    assert_accretive(domain_operator(domain, 'inverse_propagator'), 'L + 1', real_min=1.0)
    assert_accretive(domain_operator(domain, 'propagator'), '(L + 1)^-1', real_min=0.0)
    assert_accretive(domain_operator(domain, 'forward'), 'A', real_min=0.0)
    assert_accretive(domain_operator(domain, 'preconditioned_operator'), 'ΓA', norm_max=1.0, norm_offset=1.0)


def assert_accretive(operator, name, *, real_min=None, real_max=None, norm_max=None, norm_offset=None):
    M = full_matrix(operator)
    # plt.imshow(M.abs().log().cpu())
    # plt.show()

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

# @pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
# @pytest.mark.parametrize("n_domains", [1])
# @pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
# def test_1domain_wrap_options(accretivity):
#     """ Check that operator A is accretive for 1-domain scenario for all wrapping correction options """
#     # round(., 12) with numpy works. 3 with torch??
#     assert round(accretivity, 3) >= 0, f'a is not accretive. {accretivity}'
#
#
# @pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
# @pytest.mark.parametrize("n_domains", [2])
# @pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
# def test_ndomains(accretivity):
#     """ Check that operator A is accretive when number of domains > 1
#     (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
#     # round(., 12) with numpy works. 3 with torch??
#     assert round(accretivity, 3) >= 0, f'a is not accretive. {accretivity}'
