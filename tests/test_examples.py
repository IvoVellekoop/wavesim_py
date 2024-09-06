import pytest
import os
import torch
import numpy as np
from scipy.io import loadmat
from scipy.signal.windows import gaussian, tukey
from PIL.Image import BILINEAR, fromarray, open
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.maxwelldomain import MaxwellDomain
from wavesim.multidomain import MultiDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import create_sphere, pad_boundaries, preprocess, relative_error

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
    n_size = (256, 1, 1, 1)
    n = np.ones(n_size, dtype=np.complex64)
    n[99:130] = 1.5
    boundary_widths = 24
    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n ** 2, boundary_widths)  # permittivity is n², but uses the same variable n

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
    boundary_widths = 40
    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n ** 2, boundary_widths)  # permittivity is n², but uses the same variable n

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
    boundary_widths = 8
    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n ** 2, boundary_widths)  # permittivity is n², but uses the same variable n

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
    n_size = (128, 48, 96, 1)
    n = np.ascontiguousarray(loadmat('examples/matlab_results.mat')['n3d_disordered'])
    boundary_widths = 12
    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n ** 2, boundary_widths)  # permittivity is n², but uses the same variable n

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


def test_maxwell_mie():
    """ Test for Maxwell's equations for Mie scattering. Compare with reference solution (matlab repo result). """
    wavelength = 1
    pixel_size = wavelength/5
    boundary_wavelengths = 5  # Boundary width in wavelengths
    boundary_widths = int(boundary_wavelengths * wavelength / pixel_size)  # Boundary width in pixels
    sphere_radius = 1
    sphere_index = 1.2
    bg_index = 1
    n_size = (60, 40, 30)

    # generate a refractive index map
    n, x_r, y_r, z_r = create_sphere(n_size, pixel_size, sphere_radius, sphere_index, bg_index)
    n = n[..., None]  # Add dimension for polarization

    n_size += (3,)  # Add 4th dimension for polarization

    # Define source
    # calculate source prefactor
    k = bg_index * 2 * np.pi / wavelength
    prefactor = 1.0j * pixel_size / (2 * k)

    # Linearly-polarized apodized plane wave
    sx = n.shape[0]
    sy = n.shape[1]
    srcx = np.reshape(tukey(sx, 0.5), (1, sx, 1))
    srcy = np.reshape(tukey(sy, 0.5), (sy, 1, 1))
    source_amplitude = np.squeeze(1 / prefactor * np.exp(1.0j * k * z_r[0,0,0]) * srcx * srcy).T
    source = np.zeros(n_size, dtype=np.complex64)
    p = 0  # x-polarization
    source[..., 0, p] = source_amplitude

    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n**2, boundary_widths)  # permittivity is n², but uses the same variable n
    # pad the source with boundaries
    source = torch.tensor(pad_boundaries(source, boundary_array), dtype=torch.complex64)

    periodic = (True, True, True)  # periodic boundaries, wrapped field.
    domain = MaxwellDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
    
    u_sphere = run_algorithm(domain, source)[0]
    u_sphere = u_sphere.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)][..., 0].cpu().numpy()

    # Run similar simulation, but without the medium (to get the background field)
    n_bg = bg_index * np.ones(n_size[:3], dtype=np.complex64)
    n_bg = n_bg[..., None]  # Add dimension for polarization
    n_bg, _ = preprocess(n_bg, boundary_widths)

    domain2 = MaxwellDomain(permittivity=n_bg, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
    u_bg = run_algorithm(domain2, source)[0]
    u_bg = u_bg.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)][..., 0].cpu().numpy()

    u_computed = u_sphere - u_bg

    # load results from matlab wavesim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['maxwell_mie'])[..., 0]

    re = relative_error(u_computed, u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error {re} higher than {threshold}"


def test_maxwell_2d():
    """ Test for Maxwell's equations for 2D propagation. Compare with reference solution (matlab repo result). """
    # generate a refractive index map
    boundary_wavelengths = 4  # Boundary width in wavelengths
    sim_size = np.array([16 + boundary_wavelengths*2, 32 + boundary_wavelengths*2])  # Simulation size in micrometers
    wavelength = 1.  # Wavelength in micrometers
    pixel_size = wavelength/8  # Pixel size in wavelength units
    boundary_widths = int(boundary_wavelengths * wavelength / pixel_size)  # Boundary width in pixels
    n_dims = len(sim_size.squeeze())  # Number of dimensions

    # Size of the simulation domain
    n_size = sim_size * wavelength / pixel_size  # Size of the simulation domain in pixels
    n_size = n_size - 2 * boundary_widths  # Subtract the boundary widths
    n_size = tuple(n_size.astype(int))  # Convert to integer for indexing
    n_size += (1,3,)

    n1 = 1
    n2 = 2
    n = np.ones((n_size[0]//2, n_size[1]), dtype=np.complex64)
    n = np.concatenate((n1 * n, n2 * n), axis=0)
    n = n[..., None, None]  # Add dimensions for z-axis and polarization

    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess(n**2, boundary_widths)  # permittivity is n², but uses the same variable n

    # define plane wave source with Gaussian intensity profile with incident angle theta
    # properties
    theta = np.pi/4  # angle of plane wave
    kx = 2 * np.pi/wavelength * np.sin(theta)
    x = np.arange(1, n_size[1] + 1) * pixel_size

    # create source object
    m = n_size[1]//2
    a = 3
    std = (m - 1)/(2 * a)
    values = torch.tensor(np.concatenate((gaussian(m, std)*np.exp(1j*kx*x[:m]), np.zeros(m))))
    idx = [[boundary_array[0], i, 0, 0] for i in range(boundary_array[1], boundary_array[1]+n_size[1])]  # [x, y, z, polarization]
    indices = torch.tensor(idx).T  # Location: beginning of domain
    n_ext = tuple(np.array(n_size) + 2*boundary_array)
    source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

    # 1-domain, periodic boundaries (without wrapping correction)
    periodic = (True, True, True)  # periodic boundaries, wrapped field.
    domain = MaxwellDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)

    u_computed = run_algorithm(domain, source)[0]
    u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*2)][..., 0].cpu().numpy()

    # load dictionary of results from matlab wavesim/anysim for comparison and validation
    u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['maxwell_2d'])[:,:,0]

    re = relative_error(u_computed, u_ref)
    print(f'Relative error: {re:.2e}')
    threshold = 1.e-3
    assert re < threshold, f"Relative error higher than {threshold}"
