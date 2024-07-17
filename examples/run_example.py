import os
import torch
import numpy as np
from time import time
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess


# generate a refractive index map
sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers
wavelength = 1.
pixel_size = 0.25
boundary_widths = 20
n_dims = len(sim_size.squeeze())

# Size of the simulation domain
n_size = sim_size * wavelength / pixel_size  # Size of the simulation domain in pixels
n_size = n_size - 2 * boundary_widths  # Subtract the boundary widths
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

torch.manual_seed(0)  # Set the random seed for reproducibility
n = (torch.normal(mean=1.3, std=0.1, size=n_size, dtype=torch.float32)
     + 1j * abs(torch.normal(mean=0.05, std=0.02, size=n_size, dtype=torch.float32))).numpy()
print(f"Size of n: {n_size}")
print(f"Size of n in GB: {n.nbytes / (1024**3):.2f}")
assert n.imag.min() >= 0, 'Imaginary part of n is negative'

# add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n, boundary_widths)
assert n.imag.min() >= 0, 'Imaginary part of n² is negative'

# set up source, with size same as n + 2*boundary_widths, and a point source at the center of the domain
indices = torch.tensor([[v // 2 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of domain
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
n_domains = (1, 1, 1)  # number of domains in each direction
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size)

# # OR. Uncomment to test domain decomposition
# n_domains = np.array([2, 1, 1])  # number of domains in each direction
# periodic = np.where(n_domains == 1, True, False)  # True for 1 domain in that direction, False otherwise
# n_domains = tuple(n_domains)
# periodic = tuple(periodic)
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
#                      n_domains=n_domains)

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
if not os.path.exists('../logs'):
    os.makedirs('../logs')
with open('../logs/output.txt', 'a') as file:
    file.write(output)

# %% crop and save the field
# # crop the field to the region of interest
# u = u.squeeze()[*([slice(boundary_widths, -boundary_widths)] * n_dims)].cpu().numpy()
# np.savez_compressed(f'{file_name}.npz', u=u)  # save the field

# %% plot the field
# extent = np.array([0, n_size[0], n_size[1], 0])*pixel_size
# u = np.abs(u[:, :, u.shape[2]//2])
# plt.imshow(u, cmap='hot_r', extent=extent)
# plt.xlabel(r'$x~(\mu m)$')
# plt.ylabel(r'$y~(\mu m)$')
# cbar = plt.colorbar(fraction=0.046, pad=0.04)
# cbar.ax.set_title(r'$|E|$')
# plt.tight_layout()
# plt.show()
# # plt.savefig(f'{file_name}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
# # plt.close('all')
