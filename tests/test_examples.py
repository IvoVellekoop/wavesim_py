import pytest
import os
import torch
import numpy as np
from scipy.io import loadmat
from PIL.Image import BILINEAR, fromarray, open
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import pad_boundaries, preprocess, relative_error

if os.path.basename(os.getcwd()) == 'tests':
    os.chdir('..')


@pytest.mark.parametrize("n_domains, periodic", [
    (None, (True, True, True)),  # periodic boundaries, wrapped field.
    ((1, 1, 1), (False, True, True)),  # wrapping correction (here and beyond)
    ((2, 1, 1), (False, True, True)),
    ((3, 1, 1), (False, True, True)),
])
def test_1d_glass_plate(n_domains, periodic):
    """ Test for 1D propagation through glass plate. Compare with reference solution (matlab repo result). """
    wavelength = 1.
    n_size = (256, 1, 1)
    n = np.ones(n_size, dtype=np.complex64)
    n[99:130] = 1.5
    boundary_widths = 50
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    indices = torch.tensor([[0 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of the domain
    values = torch.tensor([1.0])  # Amplitude: 1
    n_ext = tuple(np.array(n_size) + 2*boundary_array)
    source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

    if n_domains is None:  # 1-domain, periodic boundaries (without wrapping correction)
        domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
    else:  # OR. Domain decomposition
        domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=n_domains)

    u_computed = run_algorithm(domain, source, max_iterations=2000)[0]
    u_computed = u_computed.squeeze()[boundary_widths:-boundary_widths]
    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['u'])

    re = relative_error(u_computed.cpu().numpy(), u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error higher than {threshold}"


@pytest.mark.parametrize("n_domains", [
    None,  # periodic boundaries, wrapped field.
    (1, 1, 1),  # wrapping correction (here and beyond)
    (2, 1, 1),
    (3, 1, 1),
    (1, 2, 1),
    (1, 3, 1),
    (2, 2, 1),
    (3, 2, 1),
])
def test_2d_low_contrast(n_domains):
    """ Test for propagation in 2D structure with low refractive index contrast (made of fat and water to mimic
        biological tissue). Compare with reference solution (matlab repo result). """
    oversampling = 0.25
    im = np.asarray(open('examples/logo_structure_vector.png')) / 255
    n_water = 1.33
    n_fat = 1.46
    n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water
    n_roi = int(oversampling * n_im.shape[0])
    n = np.asarray(fromarray(n_im).resize((n_roi, n_roi), BILINEAR))
    boundary_widths = 50
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    source = pad_boundaries(source, boundary_array)
    source = torch.tensor(source, dtype=torch.complex64)

    wavelength = 0.532
    pixel_size = wavelength / (3 * abs(n_fat))

    if n_domains is None:  # 1-domain, periodic boundaries (without wrapping correction)
        periodic = (True, True, True)  # periodic boundaries, wrapped field.
        domain = HelmholtzDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
    else:  # OR. Domain decomposition
        periodic = np.where(np.array(n_domains) == 1, True, False)  # True for 1 domain in direction, False otherwise
        periodic = tuple(periodic)
        domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
                             n_domains=n_domains)

    u_computed = run_algorithm(domain, source, max_iterations=10000)[0]
    u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*2)]
    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['u2d_lc'])

    re = relative_error(u_computed.cpu().numpy(), u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error higher than {threshold}"


@pytest.mark.parametrize("n_domains", [
    None,  # periodic boundaries, wrapped field.
    (1, 1, 1),  # wrapping correction (here and beyond)
    (1, 2, 1),
])
def test_2d_high_contrast(n_domains):
    """ Test for propagation in 2D structure made of iron, with high refractive index contrast.
        Compare with reference solution (matlab repo result). """

    oversampling = 0.25
    im = np.asarray(open('examples/logo_structure_vector.png')) / 255
    n_iron = 2.8954 + 2.9179j
    n_contrast = n_iron - 1
    n_im = ((np.where(im[:, :, 2] > 0.25, 1, 0) * n_contrast) + 1)
    n_roi = int(oversampling * n_im.shape[0])
    n = np.asarray(fromarray(n_im.real).resize((n_roi, n_roi), BILINEAR)) + 1j * np.asarray(
        fromarray(n_im.imag).resize((n_roi, n_roi), BILINEAR))
    # # load dictionary of results from matlab wavesim/anysim for comparison and validation
    # n = loadmat('examples/matlab_results.mat')['n2d_hc']
    boundary_widths = 50
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    source = pad_boundaries(source, boundary_array)
    source = torch.tensor(source, dtype=torch.complex64)

    wavelength = 0.532
    pixel_size = wavelength / (3 * np.max(abs(n_contrast + 1)))

    if n_domains is None:  # 1-domain, periodic boundaries (without wrapping correction)
        periodic = (True, True, True)  # periodic boundaries, wrapped field.
        domain = HelmholtzDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
    else:  # OR. Domain decomposition
        periodic = np.where(np.array(n_domains) == 1, True, False)  # True for 1 domain in direction, False otherwise
        periodic = tuple(periodic)
        domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
                             n_domains=n_domains)

    u_computed = run_algorithm(domain, source, max_iterations=int(1.e+5))[0]
    u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*2)]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['u2d_hc'])

    re = relative_error(u_computed.cpu().numpy(), u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"


@pytest.mark.parametrize("n_domains", [
    None,  # periodic boundaries, wrapped field.
    (1, 1, 1),  # wrapping correction (here and beyond)
    (2, 1, 1),
    (3, 1, 1),
    (1, 2, 1),
    (1, 3, 1),
    (1, 1, 2),
    (1, 1, 2),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (2, 2, 2),
    (3, 2, 1),
    (3, 1, 2),
    (1, 3, 2),
    (1, 2, 3),
])
def test_3d_disordered(n_domains):
    """ Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result). """
    wavelength = 1.
    n_size = (128, 48, 96)
    n = np.ascontiguousarray(loadmat('examples/matlab_results.mat')['n3d_disordered'])
    boundary_widths = 50
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    # Source: single point source in the center of the domain
    indices = torch.tensor([[int(v/2 - 1) + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location
    values = torch.tensor([1.0])  # Amplitude: 1
    n_ext = tuple(np.array(n_size) + 2*boundary_array)
    source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

    if n_domains is None:  # 1-domain, periodic boundaries (without wrapping correction)
        periodic = (True, True, True)  # periodic boundaries, wrapped field.
        domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
    else:  # OR. Domain decomposition
        periodic = np.where(np.array(n_domains) == 1, True, False)  # True for 1 domain in direction, False otherwise
        periodic = tuple(periodic)
        domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=n_domains)

    u_computed = run_algorithm(domain, source, max_iterations=1000)[0]
    u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['u3d_disordered'])

    re = relative_error(u_computed.cpu().numpy(), u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"


@pytest.mark.parametrize("n_domains", [
    None,  # periodic boundaries, wrapped field.
    (1, 1, 1),  # wrapping correction (here and beyond)
    (2, 1, 1),
    (3, 1, 1),
    (1, 2, 1),
    (1, 3, 1),
    (1, 1, 2),
    (1, 1, 2),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (3, 2, 1),
    (3, 1, 2),
    (1, 3, 2),
    (1, 2, 3),
])
def test_3d_homogeneous(n_domains):
    """ Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result). """
    wavelength = 1.
    n_size = (128, 128, 128)
    n = np.ones(tuple(n_size), dtype=np.complex64)
    boundary_widths = 50
    # add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n, boundary_widths)

    # Source: single point source in the center of the domain
    indices = torch.tensor([[int(v/2 - 1) + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location
    values = torch.tensor([1.0])  # Amplitude: 1
    n_ext = tuple(np.array(n_size) + 2*boundary_array)
    source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

    if n_domains is None:  # 1-domain, periodic boundaries (without wrapping correction)
        periodic = (True, True, True)  # periodic boundaries, wrapped field.
        domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
    else:  # OR. Domain decomposition
        periodic = np.where(np.array(n_domains) == 1, True, False)  # True for 1 domain in direction, False otherwise
        periodic = tuple(periodic)
        domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=n_domains)

    u_computed = run_algorithm(domain, source, max_iterations=2000)[0]
    u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)]

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')[f'u3d_{n_size[0]}_{n_size[1]}_{n_size[2]}_bw_20_24_32'])

    re = relative_error(u_computed.cpu().numpy(), u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"
