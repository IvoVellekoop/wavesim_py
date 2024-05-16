"""Tests to compare the result of Wavesim to analytical results"""
import pytest

from anysim import run_algorithm, domain_operator
from wavesim.multidomain import MultiDomain
from wavesim.helmholtzdomain import HelmholtzDomain
import torch
from torch import tensor
from . import allclose, random_vector, device, dtype, random_refractive_index


def test_no_propagation():
    """Basic test where the L-component is zero
    By manually removing the laplacian, we are solving the equation (2 π n / λ)² x = y
    """
    n_size = (10, 10, 10)
    n = random_refractive_index(n_size)
    domain = HelmholtzDomain(refractive_index=n, pixel_size=0.25, periodic=(True, True, True), n_boundary=0)
    y = random_vector(domain.shape)

    # manually disable the propagator, and test if, indeed, we are solving the system (2 π n / λ)² x = y
    L1 = 1.0 + domain.shift * domain.scale
    domain.propagator_kernel = 1.0 / L1
    domain.inverse_propagator_kernel = L1
    yy = domain_operator(domain, 'inverse_propagator')(y)
    assert allclose(yy, y * L1)
    assert allclose(domain_operator(domain, 'propagator')(yy), y)

    By = domain_operator(domain, 'medium')(y)
    By_correct = (1.0 - ((2 * torch.pi * n * domain.pixel_size) ** 2 - domain.shift) * domain.scale) * y
    assert allclose(By, By_correct)

    A = domain_operator(domain, 'forward')
    nny = A(y)
    assert allclose(nny, (2 * torch.pi * n * domain.pixel_size) ** 2 * y * domain.scale)

    # todo: the algorithm currently diverges!!
    x_wavesim = run_algorithm(domain, y)
    assert allclose(y * (domain.pixel_size / n / 2 / torch.pi) ** 2, x_wavesim)

# @pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'), (1, 'L_omega'),
#                                                         (2, 'wrap_corr'), (3, 'wrap_corr'), (4, 'wrap_corr')])
# def test_1d_homogeneous(n_domains, wrap_correction):
#     """ Test for 1D free-space propagation. Compare with analytic solution """
#     n_size = (256, 1, 1)
#     n = np.ones(n_size, dtype=np.complex64)
#     u_ref = u_ref_1d_h(n)
#     source = np.zeros_like(n)
#     source[0] = 1.
#     base = MultiDomain(refractive_index=n, source=source, n_domains=n_domains, wrap_correction=wrap_correction)
#     u_computed, state = run_algorithm(base)
#     LogPlot(base, state, u_computed, u_ref).log_and_plot()
#     compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)
