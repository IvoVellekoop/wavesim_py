import pytest
import torch
import numpy as np
from scipy.special import exp1
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from wavesim.iteration import domain_operator, preconditioned_iteration, preconditioner, run_algorithm
from wavesim.utilities import preprocess, relative_error
from . import allclose, random_vector, random_refractive_index

"""Tests to compare the result of Wavesim to analytical results"""


def analytical_solution(n_size0, pixel_size, wavelength=None):
    """ Compute analytic solution for 1D case """
    x = np.arange(0, n_size0 * pixel_size, pixel_size, dtype=np.float32)
    x = np.pad(x, (n_size0, n_size0), mode='constant', constant_values=np.nan)
    h = pixel_size
    # wavenumber (k)
    if wavelength is None:
        k = 1. * 2. * np.pi * pixel_size
    else:
        k = 1. * 2. * np.pi / wavelength
    phi = k * x
    u_theory = (1.0j * h / (2 * k) * np.exp(1.0j * phi)  # propagating plane wave
                - h / (4 * np.pi * k) * (
                        np.exp(1.0j * phi) * (exp1(1.0j * (k - np.pi / h) * x) - exp1(1.0j * (k + np.pi / h) * x)) -
                        np.exp(-1.0j * phi) * (-exp1(-1.0j * (k - np.pi / h) * x) + exp1(-1.0j * (k + np.pi / h) * x)))
                )
    small = np.abs(k * x) < 1.e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory[n_size0:-n_size0]


def test_no_propagation():
    """Basic test where the L-component is zero
    By manually removing the laplacian, we are solving the equation (2 π n / λ)² x = y
    """
    n = random_refractive_index((2, 3, 4))
    domain = HelmholtzDomain(permittivity=(n ** 2), periodic=(True, True, True))
    x = random_vector(domain.shape)

    # manually disable the propagator, and test if, indeed, we are solving the system (2 π n / λ)² x = y
    L1 = 1.0 + domain.shift * domain.scale
    domain._propagator_kernel = 1.0 / L1
    k2 = -(2 * torch.pi * n * domain.pixel_size) ** 2  # -(2 π n / λ)²
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

    x_wavesim = run_algorithm(domain, y, threshold=1.e-16)[0]
    assert allclose(x_wavesim, x)


@pytest.mark.parametrize("size", [(32, 1, 1), (7, 15, 1), (13, 25, 46)])
@pytest.mark.parametrize("boundary_widths", [0, 10])
@pytest.mark.parametrize("periodic", [(True, True, True),  # periodic boundaries, wrapped field
                                      (False, True, True)])  # wrapping correction
def test_residual(size, boundary_widths, periodic):
    """ Check that the residual_norm at first iteration == 1
        residual_norm is normalized with the preconditioned source
        residual_norm = norm ( B(x - (L+1)⁻¹ (B·x + c·y)) )
        norm of preconditioned source = norm( B(L+1)⁻¹y )
    """
    torch.manual_seed(0)  # Set the random seed for reproducibility
    n = (torch.normal(mean=1.3, std=0.1, size=size, dtype=torch.float32)
         + 1j * abs(torch.normal(mean=0.05, std=0.02, size=size, dtype=torch.float32))).numpy()
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    indices = torch.tensor([[0 + boundary_array[i] for i, v in enumerate(size)]]).T  # Location: center of the domain
    values = torch.tensor([1.0])  # Amplitude: 1
    n_ext = tuple(np.array(size) + 2 * boundary_array)
    source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

    wavelength = 1.
    domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
    # domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=1., n_domains=n_domains)

    # Reset the field u to zero
    slot_x = 0
    slot_tmp = 1
    domain.clear(slot_x)
    domain.set_source(source)

    # compute initial residual
    domain.add_source(slot_x, weight=1.)  # [x] = y
    preconditioner(domain, slot_x, slot_x)  # [x] = B(L+1)⁻¹y
    init_norm = domain.inner_product(slot_x, slot_x)  # inverse of initial norm, 1 / norm([x])
    domain.clear(slot_x)  # Clear [x]

    residual_norm = preconditioned_iteration(domain, slot_x, slot_x, slot_tmp, alpha=0.75, compute_norm2=True)

    assert np.allclose(residual_norm, init_norm)


@pytest.mark.parametrize("n_domains, periodic", [
    ((1, 1, 1), (True, True, True)),  # periodic boundaries, wrapped field.
    ((1, 1, 1), (False, True, True)),  # wrapping correction (here and beyond)
    ((2, 1, 1), (False, True, True)),
    ((3, 1, 1), (False, True, True)),
])
def test_1d_analytical(n_domains, periodic):
    """ Test for 1D free-space propagation. Compare with analytic solution """
    wavelength = 1.  # wavelength in micrometer (um)
    pixel_size = wavelength / 4  # pixel size in wavelength units
    n_size = (512, 1, 1)  # size of simulation domain (in pixels in x, y, and z direction)
    n = np.ones(n_size, dtype=np.complex64)
    boundary_widths = 50  # width of the boundary in pixels
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    indices = torch.tensor([[0 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of the domain
    values = torch.tensor([1.0])  # Amplitude: 1
    n_ext = tuple(np.array(n_size) + 2 * boundary_array)
    source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

    # domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
    domain = MultiDomain(permittivity=n, periodic=periodic,
                         wavelength=wavelength, pixel_size=pixel_size, n_domains=n_domains)
    u_computed = run_algorithm(domain, source, max_iterations=10000)[0]
    u_computed = u_computed.squeeze()[boundary_widths:-boundary_widths]
    u_ref = analytical_solution(n_size[0], domain.pixel_size, wavelength)

    re = relative_error(u_computed.cpu().numpy(), u_ref)
    print(f'Relative error: {re:.2e}')

    assert re <= 1.e-3, f'Relative error: {re:.2e}'
    assert allclose(u_computed, u_ref, atol=1.e-3, rtol=1.e-3)
