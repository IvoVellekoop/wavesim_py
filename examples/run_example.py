"""
Run Helmholtz example
=====================
Example script to run a simulation of a point source in a random refractive index map using the Helmholtz equation.
"""

import os
import sys

import torch
import numpy as np
from time import time
from scipy.signal.windows import gaussian
from torch.fft import fftn, ifftn, fftshift
import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.append(".")
from __init__ import random_permittivity, construct_source
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess, normalize

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')

# generate a refractive index map
sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers
periodic = (False, True, True)
wavelength = 1.  # Wavelength in micrometers
pixel_size = wavelength/4  # Pixel size in wavelength units
boundary_wavelengths = 10  # Boundary width in wavelengths
boundary_widths = [int(boundary_wavelengths * wavelength / pixel_size), 0, 0]  # Boundary width in pixels
n_dims = len(sim_size.squeeze())  # Number of dimensions

# Size of the simulation domain
n_size = sim_size * wavelength / pixel_size  # Size of the simulation domain in pixels
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

# return permittivity (n²) with absorbing boundaries
permittivity = random_permittivity(n_size)
permittivity = preprocess(permittivity, boundary_widths)[0]
assert permittivity.imag.min() >= 0, 'Imaginary part of n² is negative'
assert (permittivity.shape == np.asarray(n_size) + 2*np.asarray(boundary_widths)).all(), 'permittivity and n_size do not match'
assert permittivity.dtype == np.complex64, f'permittivity is not complex64, but {permittivity.dtype}'

# construct a source at the center of the domain
source = construct_source(source_type='point', at=np.asarray(permittivity.shape) // 2, shape=permittivity.shape)
# source = construct_source(source_type='plane_wave', at=[[boundary_widths[0]]], shape=permittivity.shape)
# source = construct_source(source_type='gaussian_beam', at=[[boundary_widths[0]]], shape=permittivity.shape)

# # 1-domain
# n_domains = (1, 1, 1)  # number of domains in each direction
# domain = HelmholtzDomain(permittivity=permittivity, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size)

# 1-domain or more with domain decomposition
n_domains = (1, 1, 1)  # number of domains in each direction
domain = MultiDomain(permittivity=permittivity, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
                     n_domains=n_domains)

start = time()
# Field u and state object with information about the run
u, iterations, residual_norm = run_algorithm(domain, source, max_iterations=1000)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')

# %% Postprocessing

file_name = './logs/size'
for i in range(n_dims):
    file_name += f'{n_size[i]}_'
file_name += f'bw{boundary_widths}_domains'
for i in range(n_dims):
    file_name += f'{n_domains[i]}'

output = (f'Size {n_size}; Boundaries {boundary_widths}; Domains {n_domains}; '
          + f'Time {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e} \n')
if not os.path.exists('logs'):
    os.makedirs('logs')
with open('logs/output.txt', 'a') as file:
    file.write(output)

# %% crop and save the field
# crop the field to the region of interest
u = u.squeeze()[*([slice(boundary_widths[i], 
                         u.shape[i] - boundary_widths[i]) for i in range(3)])].cpu().numpy()
np.savez_compressed(f'{file_name}.npz', u=u)  # save the field

# %% plot the field
extent = np.array([0, n_size[0], n_size[1], 0])*pixel_size
u = normalize(np.abs(u[:, :, u.shape[2]//2].T))
plt.imshow(u, cmap='inferno', extent=extent, norm=colors.LogNorm())
plt.xlabel(r'$x~(\mu m)$')
plt.ylabel(r'$y~(\mu m)$')
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.set_title(r'$|E|$')
plt.tight_layout()
plt.savefig(f'{file_name}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.show()
# plt.close('all')
