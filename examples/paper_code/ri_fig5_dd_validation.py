import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, colors
from matplotlib.ticker import MultipleLocator

from __init__ import sim_3d_random

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 13}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_data', exist_ok=True)
    os.makedirs('paper_figures', exist_ok=True)
    filename = 'paper_data/ri_fig5_dd_validation_'
    figname = 'paper_figures/ri_fig5_dd_validation.pdf'
else:
    try:
        os.makedirs('examples/paper_data', exist_ok=True)
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_data/ri_fig5_dd_validation_'
        figname = 'examples/paper_figures/ri_fig5_dd_validation.pdf'
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'paper_code' directory.")

sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers (excluding boundaries)

n_range = np.round(
    np.concatenate(
        (np.arange(1.0, 2.0, 0.1), 
         np.arange(2.0, 5.1, 0.5)
         ), 
         0), 
    1)
k_range = np.round(
    np.concatenate(
        (np.arange(0.0, 0.1, 0.02), 
         np.arange(0.1, 1.0, 0.1), 
         np.arange(1.0, 10., 2.0)
         ), 
        0), 2)
print('n_range', n_range)
print('k_range', k_range)
print(len(list(product(n_range, k_range))))

if os.path.exists(f'{filename}.txt'):
    print(f"File {filename}.txt already exists. Loading data and plotting...")
else:
    # Run the simulations
    for n_medium, k_medium in product(n_range, k_range):
        fn = filename + f'n{n_medium}_k{k_medium}_'
        sim_ref = sim_3d_random(fn, sim_size, n_domains=None, n_boundary=0, 
                                n_medium=n_medium, k_medium=k_medium, r=12)

        data = (f'n+ik: {n_medium}+{k_medium}j; Time {sim_ref['sim_time']:2.2f}; Iterations {sim_ref['iterations']}; Residual norm {sim_ref['residual_norm']:.3e}\n')

        with open(f'{filename}.txt', 'a') as file:
            file.write(data)

data = np.loadtxt(f'{filename}.txt', dtype=str, delimiter=';')

times = [data[i, 1] for i in range(len(data))]
times = np.array([float(times[i].split(' ')[2]) for i in range(len(times))])
times = np.reshape(times, (len(n_range), len(k_range)), order='F')

iterations = [data[i, 2] for i in range(len(data))]
iterations = np.array([int(iterations[i].split(' ')[2]) for i in range(len(iterations))])
iterations = np.reshape(iterations, (len(n_range), len(k_range)), order='F')

print('iterations\n', iterations)
print('times\n', times)

fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=2, sharey=True, gridspec_kw={'wspace': 0.25})
cmap = 'inferno'
gamma = .4

im0 = ax[0].imshow(np.flipud(iterations), cmap=cmap, norm=colors.LogNorm(), aspect='auto', extent=[0.95, n_range[-1]+0.05, -0.005, k_range[-1]+0.005])
ax[0].set_xlabel(r'$n$')
ax[0].set_ylabel(r'$k$')
cbar0 = plt.colorbar(im0, label='Iterations', fraction=0.046, pad=0.04)
# cbar0.set_ticks([200, 300, 400, 500, 1000, 1500])
# cbar0.set_ticklabels([200, 300, 400, 500, 1000, 1500])
ax[0].set_title('Iterations vs Refractive index')
ax[0].text(0.5, -0.27, '(a)', color='k', ha='center', va='center', transform=ax[0].transAxes)
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
# ax[0].set_xticks(np.arange(n_range[0], n_range[-1]+0.05, 0.1))
# ax[0].set_yticks(np.arange(k_range[0], k_range[-1]+0.005, 0.01))

im1 = ax[1].imshow(np.flipud(times), cmap=cmap, norm=colors.LogNorm(), aspect='auto', extent=[0.95, n_range[-1]+0.05, -0.005, k_range[-1]+0.005])
ax[1].set_xlabel(r'$n$')
cbar1 = plt.colorbar(im1, label='Time (s)', fraction=0.046, pad=0.04)
# cbar1.set_ticks([10, 20, 40, 60, 80])
# cbar1.set_ticklabels([10, 20, 40, 60, 80])
ax[1].set_title('Time vs Refractive index')
ax[1].text(0.5, -0.27, '(b)', color='k', ha='center', va='center', transform=ax[1].transAxes)
ax[1].xaxis.set_minor_locator(MultipleLocator(1))
ax[1].yaxis.set_minor_locator(MultipleLocator(1))
# ax[1].set_xticks(np.arange(n_range[0], n_range[-1]+0.05, 0.1))
# ax[1].set_yticks(np.arange(k_range[0], k_range[-1]+0.005, 0.01))

plt.savefig(figname, bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')
print(f'Saved: {figname}')
