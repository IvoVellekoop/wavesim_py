import os
import sys
import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
from matplotlib import rc, rcParams, colors

sys.path.append(".")
sys.path.append("..")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the wavesim iteration
from wavesim.utilities import preprocess, normalize, relative_error
from __init__ import random_refractive_index, construct_source

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 13}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_data', exist_ok=True)
    os.makedirs('paper_figures', exist_ok=True)
    current_dir = 'paper_data/'
    figname = (f'paper_figures/fig6_dd_truncation.pdf')
else:
    try:
        os.makedirs('examples/paper_data', exist_ok=True)
        os.makedirs('examples/paper_figures', exist_ok=True)
        current_dir = 'examples/paper_data/'
        figname = (f'examples/paper_figures/fig6_dd_truncation.pdf')
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'paper_code' directory.")

sim_size = 100 * np.array([1, 1, 1])  # Simulation size in micrometers (excluding boundaries)

wavelength = 1.  # Wavelength in micrometers
pixel_size = wavelength/4  # Pixel size in wavelength units
boundary_wavelengths = 5  # Boundary width in wavelengths
boundary_widths = [round(boundary_wavelengths * wavelength / pixel_size), 0, 0]  # Boundary width in pixels
# Periodic boundaries True (no wrapping correction) if boundary width is 0, else False (wrapping correction)
periodic = tuple(np.where(np.array(boundary_widths) == 0, True, False))
n_dims = np.count_nonzero(sim_size != 1)  # Number of dimensions

# Size of the simulation domain
n_size = np.ones_like(sim_size, dtype=int)
n_size[:n_dims] = sim_size[:n_dims] * wavelength / pixel_size  # Size of the simulation domain in pixels
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

filename = os.path.join(current_dir, f'fig6_dd_truncation.npz')
if os.path.exists(filename):
    print(f"File {filename} already exists. Loading data...")
    data = np.load(filename)
    corrs = data['corrs']
    sim_time = data['sim_time']
    iterations = data['iterations']
    ure_list = data['ure_list']
    print(f"Loaded data from {filename}. Now plotting...")
else:
    n = random_refractive_index(n_size)  # Random refractive index

    # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
    n, boundary_array = preprocess((n**2), boundary_widths)  # permittivity is n², but uses the same variable n

    print(f"Size of n: {n_size}")
    print(f"Size of n in GB: {n.nbytes / (1024**3):.2f}")
    assert n.imag.min() >= 0, 'Imaginary part of n² is negative'
    assert (n.shape == np.asarray(n_size) + 2*boundary_array).all(), 'n and n_size do not match'
    assert n.dtype == np.complex64, f'n is not complex64, but {n.dtype}'

    source = construct_source(n_size, boundary_array)

    domain_ref =  HelmholtzDomain(permittivity=n, periodic=periodic, 
                                wavelength=wavelength, pixel_size=pixel_size)

    start_ref = time()
    # Field u and state object with information about the run
    u_ref, iterations_ref, residual_norm_ref = run_algorithm(domain_ref, source, 
                                                            max_iterations=10000)
    sim_time_ref = time() - start_ref
    print(f'\nTime {sim_time_ref:2.2f} s; Iterations {iterations_ref}; Residual norm {residual_norm_ref:.3e}')
    # crop the field to the region of interest
    u_ref = u_ref[*(slice(boundary_widths[i], 
                    u_ref.shape[i] - boundary_widths[i]) for i in range(3))].cpu().numpy()

    n_ext = np.array(n_size) + 2*boundary_array
    corrs = np.arange(n_ext[0] // 4 + 1)
    print(f"Number of correction points: {corrs[-1]}")

    ure_list = []
    iterations = []
    sim_time = []

    n_domains = (2, 1, 1)  # number of domains in each direction

    for n_boundary in corrs:
        print(f'n_boundary {n_boundary}/{corrs[-1]}', end='\r')
        domain_n = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, 
                            pixel_size=pixel_size, n_domains=n_domains, n_boundary=n_boundary)

        start_n = time()
        u_n, iterations_n, residual_norm_n = run_algorithm(domain_n, source, 
                                                        max_iterations=10000)
        sim_time_n = time() - start_n
        print(f'\nTime {sim_time_n:2.2f} s; Iterations {iterations_n}; Residual norm {residual_norm_n:.3e}')
        # crop the field to the region of interest
        u_n = u_n[*(slice(boundary_widths[i], 
                      u_n.shape[i] - boundary_widths[i]) for i in range(3))].cpu().numpy()

        ure_list.append(relative_error(u_n, u_ref))
        iterations.append(iterations_n)
        sim_time.append(sim_time_n)

    sim_time = np.array(sim_time)
    iterations = np.array(iterations)
    ure_list = np.array(ure_list)
    np.savez_compressed(filename, corrs=corrs,
                        sim_time=sim_time,
                        iterations=iterations,
                        ure_list=ure_list)
    print(f'Saved: {filename}. Now plotting...')


# Plot
length = int(len(ure_list) * 2/3)
x = np.arange(length)
ncols = 3
figsize = (12, 3)

fig, axs = plt.subplots(1, ncols, figsize=figsize, gridspec_kw={'hspace': 0., 'wspace': 0.3})

ax0 = axs[0]
ax0.semilogy(x, ure_list[:length], 'r', lw=1., marker='x', markersize=3)
ax0.set_xlabel('Number of correction points')
ax0.set_ylabel('Relative Error')
ax0.set_xticks(np.arange(0, length, 10))
ax0.set_xlim([-2 if n_dims == 3 else -10, length + 1 if n_dims == 3 else length + 9])
ax0.grid(True, which='major', linestyle='--', linewidth=0.5)

ax1 = axs[1]
ax1.plot(x, iterations[:length], 'g', lw=1., marker='+', markersize=3)
ax1.set_xlabel('Number of correction points')
ax1.set_ylabel('Iterations')
ax1.set_xticks(np.arange(0, length, 10))
ax1.set_xlim([-2 if n_dims == 3 else -10, length + 1 if n_dims == 3 else length + 9])
ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

ax2 = axs[2]
ax2.plot(x, sim_time[:length], 'b', lw=1., marker='*', markersize=3)
ax2.set_xlabel('Number of correction points')
ax2.set_ylabel('Time (s)')
ax2.set_xticks(np.arange(0, length, 10))
ax2.set_xlim([-2 if n_dims == 3 else -10, length + 1 if n_dims == 3 else length + 9])
ax2.grid(True, which='major', linestyle='--', linewidth=0.5)

# Add text boxes with labels (a), (b), (c), ...
labels = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axs.flat):
    ax.text(0.5, -0.3, labels[i], transform=ax.transAxes, ha='center')

plt.savefig(figname, bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')
