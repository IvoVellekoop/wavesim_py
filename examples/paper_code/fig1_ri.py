"""
Refractive indices
==================
Repeat the experiment for 1 domain in figure 6 (domain decomposition 
validation), but for different refractive index values.
The real part (n) varies from 1 to 5, and the imaginary part (k) varies
from 0 to 9 (nonuniform spacing for both n and k, starting with small 
increments for smaller values, and much larger for larger values).
"""

import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, colors
from matplotlib.image import NonUniformImage

from __init__ import sim_3d_random

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 13}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_data', exist_ok=True)
    os.makedirs('paper_figures', exist_ok=True)
    filename = 'paper_data/fig1_ri_'
    figname = 'paper_figures/fig1_ri'
else:
    try:
        os.makedirs('examples/paper_data', exist_ok=True)
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_data/fig1_ri_'
        figname = 'examples/paper_figures/fig1_ri'
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
times = np.reshape(times, (len(k_range), len(n_range)), order='F')

iterations = [data[i, 2] for i in range(len(data))]
iterations = np.array([int(iterations[i].split(' ')[2]) for i in range(len(iterations))])
iterations = np.reshape(iterations, (len(k_range), len(n_range)), order='F')

# Plot the results
xtick_label = [1.0, 2.0, 3.0, 4.0, 5.0]
xtick_idx = [np.where(n_range == i)[0][0] for i in xtick_label]
ytick_label = [0., 0.1, 1., 5., 9.]
ytick_idx = [np.where(k_range == i)[0][0] for i in ytick_label]

xtick_label_ = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.5, 3.5, 4.5]
xtick_idx_ = [np.where(n_range == i)[0][0] for i in xtick_label_]
ytick_label_ = [0.02, 0.04, 0.06, 0.08, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 3.0, 7.0]
ytick_idx_ = [np.where(k_range == i)[0][0] for i in ytick_label_]

fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=2, gridspec_kw={'wspace': 0.25}, sharex=True, sharey=True)
cmap = 'inferno'

ax0 = ax[0]
im0 = ax0.imshow(iterations, cmap=cmap, norm=colors.LogNorm(), aspect='auto')
ax0.set_xlabel(r'$n$')
ax0.set_ylabel(r'$k$')
cbar0 = plt.colorbar(im0, label='Iterations', fraction=0.046, pad=0.02)
ax0.set_title('Iterations vs Refractive index')
ax0.text(0.5, -0.27, '(a)', color='k', ha='center', va='center', transform=ax0.transAxes)
ax0.set_xticks(xtick_idx, xtick_label)
ax0.set_xticks(xtick_idx_, xtick_label_, minor=True, fontdict={'fontsize':5})
ax0.set_yticks(ytick_idx, ytick_label)
ax0.set_yticks(ytick_idx_, ytick_label_, minor=True, fontdict={'fontsize':5})
ax0.invert_yaxis()

ax1 = ax[1]
im1 = ax1.imshow(times, cmap=cmap, norm=colors.LogNorm(), aspect='auto')
ax1.set_xlabel(r'$n$')
cbar1 = plt.colorbar(im1, label='Time (s)', fraction=0.046, pad=0.02)
ax1.set_title('Time vs Refractive index')
ax1.text(0.5, -0.27, '(b)', color='k', ha='center', va='center', transform=ax1.transAxes)
ax1.set_xticks(xtick_idx, xtick_label)
ax1.set_xticks(xtick_idx_, xtick_label_, minor=True, fontdict={'fontsize':5})
ax1.invert_yaxis()

plt.savefig(f'{figname}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')


xticks = [1.0, 2.0, 3.0, 4.0, 5.0]
yticks = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]

fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=2, gridspec_kw={'wspace': 0.25}, sharex=True, sharey=True)
cmap = 'inferno'
x0 = n_range[0] - 0.1/2
x1 = n_range[-1] + 0.5/2
y0 = k_range[0] - 0.02/2
y1 = k_range[-1] + 2.0/2

ax0 = ax[0]
im0 = NonUniformImage(ax0, interpolation='nearest', cmap=cmap, extent=(x0, x1, y0, y1), norm=colors.LogNorm())
im0.set_data(n_range, k_range, iterations)
ax0.add_image(im0)
ax0.set_xlim(x0, x1)
ax0.set_ylim(y0, y1)
ax0.set_xlabel(r'$n$')
ax0.set_ylabel(r'$k$')
cbar0 = plt.colorbar(im0, label='Iterations', fraction=0.046, pad=0.02)
ax0.set_title('Iterations vs Refractive index')
ax0.text(0.5, -0.27, '(a)', color='k', ha='center', va='center', transform=ax0.transAxes)
ax0.set_xticks(xticks)
ax0.set_yticks(yticks)

ax1 = ax[1]
im1 = NonUniformImage(ax1, interpolation='nearest', cmap=cmap, extent=(x0, x1, y0, y1), norm=colors.LogNorm())
im1.set_data(n_range, k_range, times)
ax1.add_image(im1)
ax1.set_xlabel(r'$n$')
cbar1 = plt.colorbar(im1, label='Time (s)', fraction=0.046, pad=0.02)
ax1.set_title('Time vs Refractive index')
ax1.text(0.5, -0.27, '(b)', color='k', ha='center', va='center', transform=ax1.transAxes)

plt.savefig(f'{figname}_interp.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')

print(f'Saved: {figname}')
