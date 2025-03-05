import os
import sys
import numpy as np
from time import time
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import MultipleLocator

sys.path.append(".")
sys.path.append("..")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the anysim iteration
from wavesim.utilities import preprocess
from __init__ import random_spheres_refractive_index, construct_source

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 13}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_data', exist_ok=True)
    os.makedirs('paper_figures', exist_ok=True)
    filename = f'paper_data/fig7_dd_convergence.txt'
    figname = f'paper_figures/fig7_dd_convergence.pdf'
else:
    try:
        os.makedirs('examples/paper_data', exist_ok=True)
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = f'examples/paper_data/fig7_dd_convergence.txt'
        figname = (f'examples/paper_figures/fig7_dd_convergence.pdf')
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'paper_code' directory.")

sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers (excluding boundaries)
wavelength = 1.  # Wavelength in micrometers
pixel_size = wavelength/4  # Pixel size in wavelength units
boundary_wavelengths = 5  # Boundary width in wavelengths
n_dims = np.count_nonzero(sim_size != 1)  # Number of dimensions

# Size of the simulation domain
n_size = np.ones_like(sim_size, dtype=int)
n_size[:n_dims] = sim_size[:n_dims] * wavelength / pixel_size  # Size of the simulation domain in pixels
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

if os.path.exists(filename):
    print(f"File {filename} already exists. Loading data and plotting...")
else:
    domains = range(1, 11)
    for nx, ny in product(domains, domains):
        print(f'Domains {nx}/{domains[-1]}, {ny}/{domains[-1]}', end='\r')

        if nx == 1 and ny == 1:
            boundary_widths = [round(boundary_wavelengths * wavelength / pixel_size), 0, 0]
        elif nx > 1 and ny == 1:
            boundary_widths = [round(boundary_wavelengths * wavelength / pixel_size), 0, 0]
        elif nx == 1 and ny > 1:
            boundary_widths = [0, round(boundary_wavelengths * wavelength / pixel_size), 0]
        else:
            boundary_widths = [round(boundary_wavelengths * wavelength / pixel_size)]*2 + [0]

        periodic = tuple(np.where(np.array(boundary_widths) == 0, True, False))
        n = random_spheres_refractive_index(n_size, r=12, clearance=0)  # Random refractive index

        # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
        n, boundary_array = preprocess((n**2), boundary_widths)  # permittivity is n², but uses the same variable n

        print(f"Size of n: {n_size}")
        print(f"Size of n in GB: {n.nbytes / (1024**3):.2f}")
        assert n.imag.min() >= 0, 'Imaginary part of n² is negative'
        assert (n.shape == np.asarray(n_size) + 2*boundary_array).all(), 'n and n_size do not match'
        assert n.dtype == np.complex64, f'n is not complex64, but {n.dtype}'

        source = construct_source(n_size, boundary_array)

        n_domains = (nx, ny, 1)
        domain = MultiDomain(permittivity=n, periodic=periodic, 
                             wavelength=wavelength, pixel_size=pixel_size, 
                             n_boundary=8, n_domains=n_domains)

        start = time()
        u, iterations, residual_norm = run_algorithm(domain, source, max_iterations=10000)  # Field u and state object with information about the run
        end = time() - start
        print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')

        #%% Save data to file
        data = (f'Size {n_size}; Boundaries {boundary_widths}; Domains {n_domains}; ' 
                + f'Time {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e} \n')
        with open(filename, 'a') as file:
            file.write(data)

#%% Domains in x AND y direction vs iterations and time
data = np.loadtxt(filename, dtype=str, delimiter=';')

num_domains = [data[i, 2] for i in range(len(data))]
num_domains = [num_domains[i].split('(', maxsplit=1)[-1] for i in range(len(num_domains))]
num_domains = [num_domains[i].split(', 1)', maxsplit=1)[0] for i in range(len(num_domains))]
num_domains = [(int(num_domains[i].split(',', maxsplit=1)[0]), int(num_domains[i].split(',', maxsplit=1)[-1])) for i in range(len(num_domains))]

x, y = max(num_domains, key=lambda x: x[0])[0], max(num_domains, key=lambda x: x[1])[1]

#%% Both subplots in one figure

iterations = [data[i, 4] for i in range(len(data))]
iterations = np.array([int(iterations[i].split(' ')[2]) for i in range(len(iterations))])
iterations = np.reshape(iterations, (x, y), order='F')

times = [data[i, 3] for i in range(len(data))]
times = [float(times[i].split(' ')[2]) for i in range(len(times))]
times = np.reshape(times, (x, y), order='F')

fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=2, sharey=True, gridspec_kw={'wspace': 0.05})
cmap = 'jet'

im0 = ax[0].imshow(np.flipud(iterations), cmap=cmap, extent=[0.5, x+0.5, 0.5, y+0.5])
ax[0].set_xlabel('Domains in x direction')
ax[0].set_ylabel('Domains in y direction')
plt.colorbar(im0, label='Iterations', fraction=0.046, pad=0.04)
ax[0].set_title('Iterations vs Number of domains')
ax[0].text(0.5, -0.27, '(a)', color='k', ha='center', va='center', transform=ax[0].transAxes)
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
ax[0].set_xticks(np.arange(2, x+1, 2))
ax[0].set_yticks(np.arange(2, y+1, 2))

im1 = ax[1].imshow(np.flipud(times), cmap=cmap, extent=[0.5, x+0.5, 0.5, y+0.5])
ax[1].set_xlabel('Domains in x direction')
plt.colorbar(im1, label='Time (s)', fraction=0.046, pad=0.04)
ax[1].set_title('Time vs Number of domains')
ax[1].text(0.5, -0.27, '(b)', color='k', ha='center', va='center', transform=ax[1].transAxes)
ax[1].xaxis.set_minor_locator(MultipleLocator(1))
ax[1].yaxis.set_minor_locator(MultipleLocator(1))
ax[1].set_xticks(np.arange(2, x+1, 2))
ax[1].set_yticks(np.arange(2, y+1, 2))

plt.savefig(figname, bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')
print(f'Saved: {figname}')
