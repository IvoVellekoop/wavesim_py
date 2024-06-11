import os
import numpy as np
from time import time
import matplotlib.pyplot as plt
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from anysim import run_algorithm  # to run the anysim iteration
from utilities import preprocess
import numpy as np


# generate a refractive index map
sim_size = np.array([400, 200, 200])  # Simulation size in micrometers
wavelength = 1.
pixel_size = 0.25
boundary_widths = 20

# Size of the simulation domain
n_size = sim_size * wavelength / pixel_size  # Size of the simulation domain in pixels
n_size = n_size - 2 * boundary_widths  # Subtract the boundary widths
n_size = n_size.astype(int)  # Convert to integer for indexing

np.random.seed(0)
n = np.random.normal(1.3, 0.1, n_size) + 1j * np.maximum(np.random.normal(0.05, 0.02, n_size), 0.0)

# set up source, with size same as n, and a point source at the center of the domain
source = np.zeros_like(n)  # Source term
source[tuple(i // 2 for i in n_size)] = 1.  # Source term at the center of the domain

n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source

# other parameters
# periodic = (True, True, True)  # periodic boundary conditions, no wrapping correction.
periodic = (False, True, True)  # wrapping corrections
n_domains = (2, 1, 1)  # number of domains in each direction

# set up scaling, and medium, propagation, and if required, correction (wrapping and transfer) operators
# domain = HelmholtzDomain(permittivity=n, wavelength=wavelength, pixel_size=pixel_size, periodic=periodic)
domain = MultiDomain(permittivity=n, wavelength=wavelength, pixel_size=pixel_size, periodic=periodic, n_domains=n_domains)

start = time()
u, iterations, residual_norm = run_algorithm(domain, source, max_iterations=1000)  # Field u and state object with information about the run
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')

# %% Postprocessing

# crop the field to the region of interest
u = u.squeeze()[*([slice(boundary_widths, -boundary_widths)] * 2)]
n_dims = u.ndim
u = np.abs(u[:,:,u.shape[2]//2].cpu().numpy())

output = (f'Size {n_size}; Boundaries {boundary_widths}; Domains {n_domains}; ' 
          + f'Time {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e} \n')
if not os.path.exists('./logs'):
    os.makedirs('./logs')
with open('./logs/output.txt', 'a') as file:
    file.write(output)

extent = extent=np.array([0, n_size[0], n_size[1], 0])*pixel_size

fig_name = './logs/size'
for i in range(n_dims):
    fig_name += f'{n_size[i]}_'
fig_name += f'bw{boundary_widths}_domains'
for i in range(n_dims):
    fig_name += f'{n_domains[i]}'
fig_name += f'_iters{iterations}.pdf'

# plot the field
plt.imshow(u, cmap='hot_r', extent=extent)
plt.xlabel(r'$x~(\mu m)$')
plt.ylabel(r'$y~(\mu m)$')
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.set_title(r'$|E|$')
plt.tight_layout()
plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')
