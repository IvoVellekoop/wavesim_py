import pytest

import numpy as np
from matplotlib import pyplot as plt

from wavesim.engine import Array, scale, norm_squared, copy, add, SparseArray
from wavesim.helmholtzdomain import Helmholtz
from wavesim.utilities import add_absorbing_boundaries
from wavesim.utilities.create_source import point_source
from wavesim.iteration import preconditioned_iteration, preconditioner, preconditioned_richardson
from wavesim.simulate import simulate
from tests import domain_operator, analytical_solution, relative_error
from . import random_vector, all_close, random_permittivity

"""Tests to compare the result of Wavesim to analytical results"""


def test_no_propagation():
    """Basic test where the L-component is zero
    By manually removing the laplacian, we are solving the equation (2 π n / λ)² x = y
    """

    # construct a domain with a random permittivity
    shape = (2, 3, 4)
    x = random_vector(shape)
    wavelength = 0.7
    xg = x.gather()
    permittivity = random_permittivity(shape)
    k2 = (2 * np.pi / wavelength) ** 2 * permittivity.gather()  # -(2 π n / λ)²
    domain = Helmholtz(
        permittivity=permittivity,
        pixel_size=0.25,
        wavelength=wavelength,
        periodic=(True, True, True),
        boundary_width=0,
    )
    # manually disable the propagator
    L1 = domain.domains[0, 0, 0].shift.value

    def propagator_replacement(data: Array, *, out: Array):
        """Applies the operator (L+1)^-1 x, with L only containing shift."""
        scale(1.0 / L1, data, out=out)

    def inverse_propagator_replacement(data: Array, *, out: Array):
        """Applies the operator (L+1) x, with L only containing shift"""
        scale(L1, data, out=out)

    domain.propagator = propagator_replacement
    domain.inverse_propagator = inverse_propagator_replacement

    # When running the algorithm, we are now solving the system (2 π n / λ)² x = y

    # first check if all operators are correct
    B = domain.shift.blocks[0, 0, 0].value - k2 * domain.scale
    alpha = 0.75
    assert all_close(domain_operator(domain, "inverse_propagator")(x), xg * L1)
    assert all_close(domain_operator(domain, "propagator")(x), xg / L1)
    assert all_close(domain_operator(domain, "medium")(x), B * xg)
    y = domain_operator(domain, "forward")(x)
    assert all_close(y, k2 * xg)
    M = domain_operator(domain, "richardson", alpha=alpha, source=y)
    assert all_close(M(0), (-domain.scale * alpha / L1) * B * y.gather())

    scale(-1.0, x, out=x)  # x = -x to compensate for negative source in the richardson iteration

    # invert the system, first by manually performing the richardson iteration
    x_wavesim = 0
    for _ in range(500):
        x_wavesim = M(x_wavesim)
    assert all_close(x_wavesim, x)

    # now use the run_algorithm function to do the same
    x_wavesim = preconditioned_richardson(domain, y, threshold=1.0e-16)[0]
    assert all_close(x_wavesim, x)


def callback_1d(domain, iteration, x, /, *, residual_norm, max_iterations, threshold, **kwargs):
    """Callback function for the preconditioned Richardson iteration

    Generates the following plots:
    - the real and imaginary part of the scattering potential domain._Bscat
    - the real and imaginary part of the field x
    """
    if iteration == 0:
        plt.figure()
        plt.subplot(2, 2, 1)
        B = domain.B_scat.gather().squeeze()
        c = domain.coordinates(0).squeeze()
        plt.plot(c, B.real, label="Re(B)")
        plt.plot(c, B.imag, label="Im(B)")
        plt.legend()

    if iteration % 20 == 0 or iteration == -1:
        xg = x.gather().squeeze()
        c = domain.coordinates(0).squeeze()
        plt.subplot(2, 2, 2)
        plt.cla()
        plt.plot(c, xg.real, label="Re(x)")
        plt.plot(c, xg.imag, label="Im(x)")
        plt.legend()
        plt.title(f"Iteration {iteration} ({max_iterations}), residual norm {residual_norm:.2e} ({threshold:.2e})")
        plt.draw()
        plt.pause(0.001)


@pytest.mark.parametrize("shape", [[32, 1, 1], [7, 15, 1], [13, 25, 46]])
@pytest.mark.parametrize("boundary_width", [0, 8])
def test_residual(shape: tuple[int, ...], boundary_width: int):
    """Check that the residual_norm at first iteration == 1
    residual_norm is normalized with the preconditioned source
    residual_norm = norm ( B(x - (L+1)⁻¹ (B·x + c·y)) )
    norm of preconditioned source = norm( B(L+1)⁻¹y )
    """
    # construct a random permittivity n²
    permittivity = random_permittivity(shape)

    # return permittivity (n²) with boundaries, and edge_widths in format (ax0, ax1, ax2)
    permittivity, roi = add_absorbing_boundaries(permittivity, boundary_width, strength=1.0)

    # construct a simulation
    domain = Helmholtz(
        permittivity=permittivity,
        pixel_size=0.25,
        wavelength=1.0,
        periodic=(True, True, True),
        boundary_width=boundary_width
    )

    # construct a source term
    source = SparseArray.point(at=np.array(shape) // 2, shape=shape)

    # Reset the field u to zero
    x = domain.allocate(0)
    out = domain.allocate()

    # compute initial residual
    add(x, source, out=x)
    preconditioner(domain, x, out=x)  # [x] = B(L+1)⁻¹y
    init_norm = norm_squared(x)  # inverse of initial norm, 1 / norm([x])
    copy(0, out=x)  # Clear [x]

    residual_norm = preconditioned_iteration(domain, x, out=out, tmp=out, alpha=0.75, compute_norm2=True, source=source)

    assert np.allclose(residual_norm, init_norm)


@pytest.mark.parametrize(
    "n_domains",
    [
        None,  # periodic boundaries, wrapped field.
        (1, 1, 1),  # wrapping correction (here and beyond)
        (2, 1, 1),
    ],
)
@pytest.mark.parametrize("use_gpu", [False, True])  # False for CPU, True for GPU acceleration
def test_1d_analytical(n_domains: tuple[int, int, int], use_gpu):
    """Test for 1D free-space propagation. Compare with analytic solution"""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 4  # pixel size in micrometer (μm)

    # Refractive index map
    sim_size = 128  # size of simulation domain in x direction in micrometer (μm)
    n_size = (int(sim_size / pixel_size), 1, 1)  # We want to set up a 1D simulation, so y and z are 1.
    permittivity = np.ones(n_size, dtype=np.complex64)  # permittivity (refractive index squared) of 1

    source_values, source_position = point_source(
        position=[sim_size//2, 0, 0],  # source position in the center of the domain in micrometer (μm)
        pixel_size=pixel_size
    )

    # Run the wavesim iteration and get the computed field
    u = simulate(
        permittivity=permittivity,
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength,
        pixel_size=pixel_size,
        boundary_width=5,  # boundary width in micrometer (μm)
        periodic=(False, True, True),
        use_gpu=use_gpu,
        n_domains=n_domains,
    )[0]

    # Compute the analytical solution
    c = pixel_size * np.arange(n_size[0])
    c = c - c[source_position[0]]
    u_ref = analytical_solution(c, wavelength)
    assert relative_error(u, u_ref) < 1.0e-3
    assert all_close(u, u_ref, rtol=4e-2)  # todo: error is too high
