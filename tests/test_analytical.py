"""Tests to compare the result of Wavesim to analytical results"""
import pytest
import torch
import numpy as np
from scipy.special import exp1
import matplotlib.pyplot as plt
from anysim import run_algorithm, domain_operator
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from . import allclose, random_vector, random_refractive_index
from utilities import preprocess, relative_error

def test_no_propagation():
    """Basic test where the L-component is zero
    By manually removing the laplacian, we are solving the equation (2 π n / λ)² x = y
    """
    n = random_refractive_index((2, 3, 4))
    domain = HelmholtzDomain(refractive_index=(n**2), pixel_size=0.25, periodic=(True, True, True))
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


def u_ref_1d_h(n_size0, pixel_size):
    """ Compute analytic solution for 1D case """
    x = np.arange(0, n_size0 * pixel_size, pixel_size, dtype=np.float32)
    x = np.pad(x, (n_size0, n_size0), mode='constant', constant_values=np.nan)
    h = pixel_size
    k = 1. * 2. * np.pi * pixel_size  # wavenumber
    phi = k * x
    u_theory = (1.0j * h / (2 * k) * np.exp(1.0j * phi)  # propagating plane wave
               - h / (4 * np.pi * k) * (
               np.exp(1.0j * phi) * (exp1(1.0j * (k - np.pi / h) * x) - exp1(1.0j * (k + np.pi / h) * x)) - 
               np.exp(-1.0j * phi) * (-exp1(-1.0j * (k - np.pi / h) * x) + exp1(-1.0j * (k + np.pi / h) * x))) 
               )
    small = np.abs(k * x) < 1.e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return torch.tensor(u_theory[n_size0:-n_size0])


@pytest.mark.parametrize("n_domains, periodic", [
    ((1, 1, 1), (True, True, True)),  # perodic boundaries, wrapped field.
    ((1, 1, 1), (False, True, True)),  # wrapping correction (here and beyond)
    # ((2, 1, 1), (False, True, True)), 
    # ((3, 1, 1), (False, True, True)), 
    # ((4, 1, 1), (False, True, True))
    ])
def test_1d_homogeneous(n_domains, periodic):
    """ Test for 1D free-space propagation. Compare with analytic solution """
    n_size = (1000, 1, 1)
    refractive_index = np.ones(n_size, dtype=np.complex64)
    source = np.zeros_like(refractive_index)
    source[0] = 1.
    boundary_widths = 50
    refractive_index, source = preprocess(refractive_index, source, boundary_widths)
    domain = HelmholtzDomain(refractive_index=refractive_index, pixel_size=0.25, periodic=periodic)
    # domain = MultiDomain(refractive_index=refractive_index, pixel_size=0.25, periodic=periodic, n_domains=n_domains)
    u_computed = run_algorithm(domain, -source, max_iterations=1000)
    u_computed = u_computed.squeeze()[boundary_widths:-boundary_widths]
    u_ref = u_ref_1d_h(n_size[0], domain.pixel_size)

    re = relative_error(u_computed.cpu().numpy(), u_ref.cpu().numpy())
    print(f'Relative error: {re:.2e}')
    plot(u_computed.cpu().numpy(), u_ref.cpu().numpy(), re)

    assert allclose(u_computed, u_ref)


def plot(a, b, re=None):
    if re is None:
        re = relative_error(a, b)
    plt.subplot(211)
    plt.plot(a.real, label='Computed')
    plt.plot(b.real, label='Analytic')
    plt.legend()
    plt.title(f'Real part (RE = {relative_error(a.real, b.real):.2e})')
    plt.grid()

    plt.subplot(212)
    plt.plot(a.imag, label='Computed')
    plt.plot(b.imag, label='Analytic')
    plt.legend()
    plt.title(f'Imaginary part (RE = {relative_error(a.imag, b.imag):.2e})')
    plt.grid()

    plt.suptitle(f'Relative error (RE) = {re:.2e}')
    plt.tight_layout()
    plt.show()
