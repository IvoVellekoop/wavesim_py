""" 
Maxwell 2D medium interface test
================================
Test for simulating a plane wave through an interface 
of two media with different refractive indices.
Compare with reference solution (matlab repo result). 
"""

import os
import torch
import numpy as np
from time import time
from scipy.io import loadmat
from scipy.signal.windows import gaussian
import sys
sys.path.append(".")
from wavesim.maxwelldomain import MaxwellDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess, relative_error
from __init__ import plot

if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')


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

start = time()
u_computed, iterations, residual_norm = run_algorithm(domain, source)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*2)][..., 0].cpu().numpy()

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['maxwell_2d'])[:,:,0]

re = relative_error(u_computed, u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed, u_ref, re, normalize_x=False)

threshold = 1.e-3
assert re < threshold, f"Relative error {re:.2f} higher than {threshold}"
