"""Tests to compare the result of Wavesim to analytical results"""
import pytest

from anysim import run_algorithm, domain_operator
from wavesim.helmholtzdomain import HelmholtzDomain
import torch
from . import allclose, random_vector, random_refractive_index


def test_no_propagation():
    """Basic test where the L-component is zero
    By manually removing the laplacian, we are solving the equation (2 π n / λ)² x = y
    """
    n = random_refractive_index((2, 3, 4))
    domain = HelmholtzDomain(refractive_index=n, pixel_size=0.25, periodic=(True, True, True))
    x = random_vector(domain.shape)

    # manually disable the propagator, and test if, indeed, we are solving the system (2 π n / λ)² x = y
    L1 = 1.0 + domain.shift * domain.scale
    domain.propagator_kernel = 1.0 / L1
    domain.inverse_propagator_kernel = L1
    k2 = (2 * torch.pi * n * domain.pixel_size) ** 2
    B = (1.0 - (k2 - domain.shift) * domain.scale)
    assert allclose(domain_operator(domain, 'inverse_propagator')(x), x * L1)
    assert allclose(domain_operator(domain, 'propagator')(x), x / L1)
    assert allclose(domain_operator(domain, 'medium')(x), B * x)

    y = domain_operator(domain, 'forward')(x)
    assert allclose(y, k2 * x)

    domain.set_source(y)
    alpha = 0.75
    M = domain_operator(domain, 'richardson', alpha=alpha)
    x_wavesim = M(0)
    assert allclose(x_wavesim, (domain.scale * alpha / L1) * B * y)

    for _ in range(500):
        x_wavesim = M(x_wavesim)

    assert allclose(x_wavesim, x)

    x_wavesim = run_algorithm(domain, y)
    assert allclose(x_wavesim, x)


def u_ref_1d_h(n):
    """ Compute analytic solution for 1D case """
    base_ = MultiDomain(refractive_index=n, setup_operators=False)

    x = np.arange(0, base_.n_roi[0] * base_.pixel_size, base_.pixel_size, dtype=np.complex64)
    x = np.pad(x, (64, 64), mode='constant')
    h = base_.pixel_size
    k = (1. * 2. * np.pi) / 1.
    phi = k * x
    u_theory = 1.0j * h / (2 * k) * np.exp(1.0j * phi) - h / (4 * np.pi * k) * (
            np.exp(1.0j * phi) * (np.exp(1.0j * (k - np.pi / h) * x) - np.exp(1.0j * (k + np.pi / h) * x)) - np.exp(
        -1.0j * phi) * (-np.exp(-1.0j * (k - np.pi / h) * x) + np.exp(-1.0j * (k + np.pi / h) * x)))
    small = np.abs(k * x) < 1.e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory[64:-64]

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
