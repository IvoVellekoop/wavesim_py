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
wavelength = 1.  # Wavelength in micrometers
pixel_size = wavelength/4  # Pixel size in wavelength units
boundary_wavelengths = 10  # Boundary width in wavelengths
boundary_widths = int(boundary_wavelengths * wavelength / pixel_size)  # Boundary width in pixels
n_dims = len(sim_size.squeeze())  # Number of dimensions

# Size of the simulation domain
n_size = sim_size * wavelength / pixel_size  # Size of the simulation domain in pixels
n_size = n_size - 2 * boundary_widths  # Subtract the boundary widths
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

torch.manual_seed(0)  # Set the random seed for reproducibility
n = (torch.normal(mean=1.3, std=0.1, size=n_size, dtype=torch.float32)
     + 1j * abs(torch.normal(mean=0.01, std=0.01, size=n_size, dtype=torch.float32)))

# low pass filter to remove sharp edges
n_fft = fftn(n)
def W(x):
    c0 = int(x*7/20)
    cl = (x - c0) // 2
    cr = cl
    if c0 + cl + cr != x:
        c0 = x - cl - cr
    return torch.cat((torch.zeros(1, cl), 
                      torch.ones(1, c0), 
                      torch.zeros(1, cr)), dim=1)

window = (W(n_size[1]).T @ W(n_size[0])).T[:,:,None] * (W(n_size[2])).view(1, 1, n_size[2])
n = ifftn(torch.multiply(n_fft, fftshift(window)))
n = torch.clamp(n.real, min=1.0) + 1.0j * torch.clamp(n.imag, min=0.0)

print(f"Size of n: {n_size}")
print(f"Size of n in GB: {n.nbytes / (1024**3):.2f}")
assert n.imag.min() >= 0, 'Imaginary part of n is negative'
assert n.shape == n_size, 'n and n_size do not match'
assert n.dtype == torch.complex64, 'n is not complex64'

# return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess((n**2).numpy(), boundary_widths)  # permittivity is n², but uses the same variable n
assert n.imag.min() >= 0, 'Imaginary part of n² is negative'
assert (n.shape == np.asarray(n_size) + 2*boundary_array).all(), 'n and n_size do not match'
assert n.dtype == np.complex64, f'n is not complex64, but {n.dtype}'

# # set up source, with size same as n + 2*boundary_widths, and a point source at the center of the domain
# indices = torch.tensor([[v//2 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of domain
# values = torch.tensor([1.0])  # Amplitude: 1
# n_ext = tuple(np.array(n_size) + 2*boundary_array)
# source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# set up source, with size same as n + 2*boundary_widths, and a plane wave source on one edge of the domain
k_size = torch.tensor([n_size[0], n_size[2]], dtype=torch.float64)
# choose |k| <  Nyquist, make sure k is at exact grid point in Fourier space
k_relative = torch.tensor((0.2, -0.15), dtype=torch.float64)
k = 2 * torch.pi * torch.round(k_relative * k_size) / k_size  # in 1/pixels
source_amplitude = torch.exp(1j * (
    k[0] * torch.arange(k_size[0]).reshape(-1, 1, 1) +
    k[1] * torch.arange(k_size[1]).reshape(1, -1, 1))).squeeze_().to(torch.complex64)
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.zeros(n_ext, dtype=torch.complex64)
source[boundary_array[0]:boundary_array[0]+n_size[0], 
       boundary_array[1], 
       boundary_array[2]:boundary_array[2]+n_size[2]] = source_amplitude

# convert source to sparse tensor
indices = torch.nonzero(source, as_tuple=False)
values = source[indices[:, 0], indices[:, 1], indices[:, 2]]
source = torch.sparse_coo_tensor(indices.T, values, n_ext, dtype=torch.complex64)

# # 1-domain, periodic boundaries (without wrapping correction)
# periodic = (True, True, True)  # periodic boundaries, wrapped field.
# n_domains = (1, 1, 1)  # number of domains in each direction
# domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size)

# OR. Uncomment to test domain decomposition
n_domains = np.array([1, 2, 1])  # number of domains in each direction
periodic = np.where(n_domains == 1, True, False)  # True for 1 domain in that direction, False otherwise
n_domains = tuple(n_domains)
periodic = tuple(periodic)
domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
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
u = u.squeeze()[*([slice(boundary_widths, -boundary_widths)] * n_dims)].cpu().numpy()
np.savez_compressed(f'{file_name}.npz', u=u)  # save the field

# %% plot the field
extent = np.array([0, n_size[0], n_size[1], 0])*pixel_size
u = normalize(np.abs(u[:, :, u.shape[2]//2].T))
plt.imshow(u, cmap='hot_r', extent=extent, norm=colors.LogNorm())
plt.xlabel(r'$x~(\mu m)$')
plt.ylabel(r'$y~(\mu m)$')
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.set_title(r'$|E|$')
plt.tight_layout()
plt.savefig(f'{file_name}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.show()
# plt.close('all')
