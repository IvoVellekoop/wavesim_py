from pathlib import Path

import pytest

import numpy as np
from scipy.io import loadmat
from scipy.signal.windows import tukey

from wavesim.utilities.create_medium import sphere_permittivity
from wavesim.utilities.create_source import gaussian_beam
from wavesim.simulate import simulate
from . import relative_error_check

"""Tests to compare the results of wavesim_py Maxwell with that of wavesim (MATLAB)"""
script_dir = Path(__file__).parent
data_file_path = script_dir.parent / "examples" / "wavesim_mat_to_py" / "matlab_results.mat"
reference_data = loadmat(str(data_file_path))

def test_maxwell_mie():
    """Test for Maxwell's equations for Mie scattering. Compare with reference solution (matlab repo result)."""
    wavelength = 1.0  # Wavelength in micrometer (μm)
    pixel_size = wavelength / 5  # Pixel size in micrometer (μm)
    periodic = (False, False, False)
    boundary_width = 4  # Boundary width in micrometer (μm)
    n_size = (120, 120, 120)  # Size of the simulation domain in pixels

    # Generate a refractive index map
    sphere_radius = 6.0
    sphere_epsilon = 1.2**2  # Permittivity of the sphere
    bg_epsilon = 1**2  # Permittivity of the background
    permittivity, x_r, y_r, z_r = sphere_permittivity(n_size, pixel_size, sphere_radius, sphere_epsilon, bg_epsilon)

    # Source term
    # calculate source prefactor
    k = np.sqrt(bg_epsilon) * 2 * np.pi / wavelength
    prefactor = 2 * k / (1.0j * pixel_size)

    # Linearly-polarized apodized plane wave
    # (filter edges to reduce diffraction and the source extends into the absorbing boundaries)
    src0 = tukey(n_size[0], 0.5).reshape((1, -1, 1, 1))
    src1 = tukey(n_size[1], 0.5).reshape((1, 1, -1, 1))
    source_values = (prefactor * np.exp(1.0j * k * z_r[0, 0, 0]) * src0 * src1).astype(np.complex64)
    source_position = [1, 0, 0, 0]  # 1 for y-polarization

    u_sphere = simulate(
        permittivity=permittivity,
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength,
        pixel_size=pixel_size,
        boundary_width=boundary_width,
        periodic=periodic
    )[0]

    # Run similar simulation, but without the medium (to get the background field)
    permittivity_bg = np.full(n_size, bg_epsilon, dtype=np.complex64)

    u_bg = simulate(
        permittivity=permittivity_bg,
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength,
        pixel_size=pixel_size,
        boundary_width=boundary_width,
        periodic=periodic
    )[0]

    u = (u_sphere - u_bg)[1, n_size[0]//2, ...]  # 2D section to compare with analytical solution

    # load results from matlab wavesim for comparison and validation
    u_ref = np.squeeze(reference_data["maxwell_mie"])

    assert relative_error_check(u, u_ref, threshold=2.e-2)  # In this simulation, the maximum accuracy is limited by the discretization of the Mie sphere, and can be further improved by performing the simulation on a finer grid. (Osnabrugge et al., 2021)


@pytest.mark.parametrize(
    "use_gpu",
    [
        True,  # GPU acceleration
        False  # CPU acceleration
    ],
)
def test_maxwell_2d(use_gpu):
    """Test for Maxwell's equations for 2D propagation. Compare with reference solution (matlab repo result).

    todo: compare to analytical solution for 2D plane wave propagation:
     - Compare angle of reflection and refraction
     - Compare amplitude of reflected and refracted wave (energy conservation)
     - Compare phase of reflected and refracted wave
     - For more accurate results, use ifft to get result for true Gaussian wave
     - Check if the matlab result is correct, is energy conserved?
    """
    # simulation parameters
    wavelength = 1.0  # Wavelength in micrometer (μm)
    pixel_size = wavelength / 8  # Pixel size in micrometer (μm)

    # Size of the simulation domain
    sim_size = np.array([16, 32])  # Simulation size in micrometer (μm)
    n_size = sim_size * wavelength / pixel_size  # Size of the simulation domain in pixels
    n_size = tuple(n_size.astype(int)) + (1,)  # Add 3rd dimension for z-axis

    # Generate a refractive index map
    epsilon1 = 1.0
    epsilon2 = 2.0**2
    permittivity = np.full(n_size, epsilon1, dtype=np.complex64)  # medium 1
    permittivity[n_size[0] // 2 :, ...] = epsilon2  # medium 2

    # Source term
    # Create a plane wave source with Gaussian intensity profile with incident angle theta
    source_values, source_position = gaussian_beam(
        shape=(sim_size[1]//2),  # source shape, 1D in Y direction, in micrometer (μm)
        origin='topleft',  # source position is defined with respect to this origin
        position=[2, 0, 0, 0],  # source top left position in [polarization axis, x , y, z]. 2 for z-polarization. x, y, z in micrometer (μm)
        source_plane='y',
        pixel_size=pixel_size,
        theta=np.pi / 4, 
        wavelength=wavelength, 
    )

    u = simulate(
        permittivity=permittivity,
        sources=[ (source_values, source_position) ],
        wavelength=wavelength,
        pixel_size=pixel_size,
        boundary_width=4,  # Boundary width in micrometer (μm)
        periodic=(False, False, True),
        use_gpu=use_gpu
    )[0]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(reference_data["maxwell_2d"])
    u_ref = np.moveaxis(u_ref, -1, 0)  # polarization is in the last axis in MATLAB
    u_ref = u_ref[(1, 0, 2), ...]  # x and y polarization are switched in MATLAB

    assert relative_error_check(u, u_ref)
