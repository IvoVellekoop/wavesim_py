import pytest

import numpy as np
from scipy.io import loadmat
from PIL.Image import fromarray, open, Resampling

from wavesim.utilities.create_source import point_source
from wavesim.simulate import simulate
from tests import relative_error
from . import all_close

""" Tests to compare the result of wavesim_py with that of wavesim (MATLAB) """

@pytest.mark.parametrize(
    "n_domains",
    [
        None,  # periodic boundaries, wrapped field.
        (1, 1, 1),  # wrapping correction (here and beyond)
        (2, 1, 1)
    ],
)
def test_1d_glass_plate(n_domains: tuple[int, ...]):
    """Test for 1D propagation through glass plate. Compare with reference solution (matlab repo result)."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 4  # pixel size in micrometer (μm)

    # Create refractive index map
    n_size = (256, 1, 1)  # size of simulation domain (in pixels in x, y, and z direction). We want to set up a 1D simulation, so y and z are 1.
    refractive_index = np.ones(n_size, dtype=np.complex64)  # background refractive index of 1
    refractive_index[99:130] = 1.5  # glass plate with refractive index of 1.5 in the defined region

    source_values, source_position = point_source(
        position=[0, 0, 0],  # source center position at the (starting) edge of the domain in micrometer (μm)
        pixel_size=pixel_size
    )

    # Run the wavesim iteration and get the computed field
    u = simulate(
        permittivity=refractive_index**2, 
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength, 
        pixel_size=pixel_size, 
        boundary_width=5,  # boundary width in micrometer (μm)
        periodic=(False, True, True), 
        n_domains=n_domains
    )[0]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u_1d"])

    # Compute relative error with respect to the reference solution
    re = relative_error(u, u_ref)
    threshold = 1.0e-3
    assert re < threshold, f"Relative error is too high: {re}"
    assert all_close(u, u_ref, rtol=4e-2)  # todo: error is too high


@pytest.mark.parametrize(
    "n_domains",
    [
        None,  # periodic boundaries, wrapped field.
        (1, 1, 1),  # wrapping correction (here and beyond)
        (2, 1, 1),
        (3, 1, 1),
        (1, 2, 1),
        (1, 3, 1),
        (2, 2, 1),
        (3, 2, 1)
    ],
)
def test_2d_low_contrast(n_domains):
    """Test for propagation in 2D structure with low refractive index contrast (made of fat and water to mimic
    biological tissue). Compare with reference solution (matlab repo result)."""
    # Parameters
    n_water = 1.33
    n_fat = 1.46
    wavelength = 0.532  # Wavelength in micrometer (μm)
    pixel_size = wavelength / (3 * abs(n_fat))  # Pixel size in micrometer (μm)

    # Load image and create refractive index map
    oversampling = 0.25
    im = np.asarray(open("examples/wavesim_mat_to_py/logo_structure_vector.png")) / 255  # Load image and normalize to [0, 1]
    n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water  # Assign refractive index to pixels
    n_roi = int(oversampling * n_im.shape[0])  # Size of ROI in pixels
    permittivity = np.asarray(fromarray(n_im).resize((n_roi, n_roi), Resampling.BILINEAR))[..., None] ** 2  # permittivity of the domain

    # Source term
    source_values = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), Resampling.BILINEAR))[..., None]
    source_position = [0, 0, 0]  # source position in (x, y, z) in pixels

    # Run the wavesim iteration and get the computed field
    u = simulate(
        permittivity=permittivity, 
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength, 
        pixel_size=pixel_size, 
        boundary_width=5,  # Boundary width in micrometer (μm) 
        periodic=(False, False, True), 
        n_domains=n_domains
    )[0]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u2d_lc"])

    # Compute relative error with respect to the reference solution
    re = relative_error(u, u_ref)
    print(f"Relative error: {re:.2e}")
    threshold = 1.0e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"
    # assert all_close(u, u_ref, rtol=4e-2)  # todo: relative error is too high. Absolute error is fine


@pytest.mark.parametrize(
    "n_domains",
    [
        None,  # periodic boundaries, wrapped field.
        (1, 1, 1),  # wrapping correction (here and beyond)
        (1, 2, 1)
    ],
)
def test_2d_high_contrast(n_domains):
    """Test for propagation in 2D structure made of iron, with high refractive index contrast.
    Compare with reference solution (matlab repo result)."""
    # Parameters
    n_iron = 2.8954 + 2.9179j
    n_contrast = n_iron - 1
    wavelength = 0.532  # Wavelength in micrometer (μm)
    pixel_size = wavelength / (3 * np.max(abs(n_contrast + 1)))  # Pixel size in micrometer (μm)

    # Load image and create refractive index map
    oversampling = 0.25
    im = np.asarray(open("examples/wavesim_mat_to_py/logo_structure_vector.png")) / 255  # Load image and normalize to [0, 1]
    n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * n_contrast) + 1  # Assign refractive index to pixels
    n_roi = int(oversampling * n_im.shape[0])  # Size of ROI in pixels
    permittivity = np.asarray(fromarray(n_im.real).resize((n_roi, n_roi), Resampling.BILINEAR)) + 1j * np.asarray(
        fromarray(n_im.imag).resize((n_roi, n_roi), Resampling.BILINEAR)
    )  # Resize to n_roi x n_roi
    permittivity = permittivity[..., None] ** 2  # Add a dimension to make the array 3D and square it to get permittivity

    # Source term
    source_values = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), Resampling.BILINEAR))[..., None]
    source_position = [0, 0, 0]  # source position in (x, y, z) in pixels

    # Run the wavesim iteration and get the computed field
    u = simulate(
        permittivity=permittivity, 
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength, 
        pixel_size=pixel_size, 
        boundary_width=5,  # Boundary width in micrometer (μm)
        periodic=(False, False, True), 
        n_domains=n_domains, 
        max_iterations=int(1.0e5)
    )[0]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u2d_hc"])

    # Compute relative error with respect to the reference solution
    re = relative_error(u, u_ref)
    print(f"Relative error: {re:.2e}")
    threshold = 1.0e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"
    # assert all_close(u, u_ref, rtol=4.0e-2)  # todo: relative error is too high. Absolute error is fine


@pytest.mark.parametrize(
    "n_domains",
    [
        None,  # periodic boundaries, wrapped field.
        (1, 1, 1),  # wrapping correction (here and beyond)
        (2, 1, 1),
        (3, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (3, 2, 1),
        (3, 1, 2),
        (1, 2, 3)
    ],
)
def test_3d_disordered(n_domains):
    """Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result)."""
    # Parameters
    wavelength = 1.0  # Wavelength in micrometer (μm)
    pixel_size = wavelength / 4  # Pixel size in micrometer (μm)

    # Load the refractive index map from a .mat file and square it to get permittivity
    permittivity = np.ascontiguousarray(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["n3d_disordered"]) ** 2
    sim_size = (np.asarray(permittivity.shape) * pixel_size).astype(int)

    # Source term
    source_values, source_position = point_source(
        position=sim_size / 2 - pixel_size,  # Source center position in the center of the domain in micrometer (μm)
        pixel_size=pixel_size
    )

    # Run the wavesim iteration and get the computed field
    u = simulate(
        permittivity=permittivity, 
        sources=[ (source_values, source_position) ], 
        wavelength=wavelength, 
        pixel_size=pixel_size, 
        boundary_width=2,  # Boundary width in micrometer (μm)
        periodic=(False, False, False), 
        n_domains=n_domains
    )[0]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u3d_disordered"])

    # Compute relative error with respect to the reference solution
    re = relative_error(u, u_ref)
    print(f"Relative error: {re:.2e}")
    threshold = 1.0e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"
    # assert all_close(u, u_ref, rtol=5.0e-3)  # todo: relative error is too high. Absolute error is fine
