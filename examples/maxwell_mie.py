""" 
Maxwell Mie sphere scattering test
==================================
Test for simulating a plane wave through a Mie scattering medium.
Compare with reference solutions (matlab repo and MatScat results).
"""

import os
import torch
import numpy as np
from time import time
from scipy.io import loadmat
from scipy.signal.windows import tukey
import sys
sys.path.append(".")
from wavesim.maxwelldomain import MaxwellDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import create_sphere, pad_boundaries, preprocess, relative_error
from __init__ import plot

if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')


# Parameters
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

# Run the wavesim iteration and get the computed field
start = time()
u_sphere, iterations, residual_norm = run_algorithm(domain, source)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_sphere = u_sphere.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)][..., 0].cpu().numpy()

# Run similar simulation, but without the medium (to get the background field)
print('\nRunning similar simulation, but without the medium (to get the background field)')
n_bg = bg_index * np.ones(n_size[:3], dtype=np.complex64)
n_bg = n_bg[..., None]  # Add dimension for polarization
n_bg, _ = preprocess(n_bg, boundary_widths)

domain2 = MaxwellDomain(permittivity=n_bg, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
start_bg = time()
u_bg = run_algorithm(domain2, source)[0]
end_bg = time() - start_bg
print(f'\nTime {end_bg:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_bg = u_bg.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)][..., 0].cpu().numpy()

u_computed = u_sphere - u_bg

# load results from matlab wavesim for comparison and validation
u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['maxwell_mie'])[..., 0]

re = relative_error(u_computed, u_ref)
print(f'Relative error (theory): {re:.2e}')
plot(u_computed, u_ref, re)#, normalize_x=False)
