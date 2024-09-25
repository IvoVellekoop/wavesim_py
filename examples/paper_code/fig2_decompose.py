"""
This script demonstrates the domain decomposition of operator A, L, and V.
It visualizes the matrices in a figure with three subplots:
- Subplot (a): Matrix A
- Subplot (b): Matrix L
- Subplot (c): Matrix V

The figure is saved as a PDF file.
"""

# import packages
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, colors

sys.path.append(".")
sys.path.append("..")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.utilities import full_matrix, normalize, preprocess
from wavesim.iteration import domain_operator

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 13}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_figures', exist_ok=True)
    filename = 'paper_figures/fig2_decompose.pdf'
else:
    try:
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_figures/fig2_decompose.pdf'
    except FileNotFoundError:
        filename = 'fig2_decompose.pdf'

# Define problem parameters
boundary_widths = 0
n_size = (40, 1, 1)
# Random refractive index distribution
torch.manual_seed(0)  # Set the random seed for reproducibility
n = (torch.normal(mean=1.3, std=0.1, size=n_size, dtype=torch.float32)
     + 1j * abs(torch.normal(mean=0.05, std=0.02, size=n_size, dtype=torch.float32)))**2
assert n.imag.min() >= 0, 'Imaginary part of n is negative'

wavelength = 1.
pixel_size = wavelength / 4
periodic = (True, True, True)

# Get matrices of A, L, and V for large domain without wraparound artifacts
boundary_widths = 100
n_o, boundary_array = preprocess(n.numpy(), boundary_widths)
assert n_o.imag.min() >= 0, 'Imaginary part of n_o is negative'

domain_o = HelmholtzDomain(permittivity=n_o, periodic=periodic, wavelength=wavelength, 
                           pixel_size=pixel_size, debug=True)

crop2roi = (slice(boundary_array[0], -boundary_array[0]), 
            slice(boundary_array[0], -boundary_array[0]))
b_o = full_matrix(domain_operator(domain_o, 'medium'))[crop2roi].cpu().numpy()  # B = I - V
l_plus1_o = full_matrix(domain_operator(domain_o, 'inverse_propagator'))[crop2roi].cpu().numpy()  # L + I

I = np.eye(np.prod(n_size), dtype=np.complex64)

v_o = (I - b_o) / domain_o.scale + domain_o.shift.item()*I  # V = (I - B) / scaling
l_o = (l_plus1_o - I) / domain_o.scale - domain_o.shift.item()*I  # L = (L + I - I) / scaling
a_o = l_o + v_o  # A = L + V

# Get matrices of A, L, and V operators decomposed into two domains
domain_2 = MultiDomain(permittivity=n, periodic=(True, True, True), wavelength=wavelength, 
                       pixel_size=pixel_size, n_domains=(2,1,1), debug=True, n_boundary=0)

b_2 = full_matrix(domain_operator(domain_2, 'medium')).cpu().numpy()  # B = I - V
l_plus1_2 = full_matrix(domain_operator(domain_2, 'inverse_propagator')).cpu().numpy()  # L + I

v_2 = (I - b_2) / domain_2.scale + domain_2.shift.item()*I  # V = (I - B) / scaling
l_2 = (l_plus1_2 - I) / domain_2.scale - domain_2.shift.item()*I  # L = (L + I - I) / scaling
l_diff = l_o - l_2
v_l_diff = v_2 + l_diff
a_2 = l_2 + v_l_diff  # A = L + V


# Normalize matrices for visualization
l_2 = l_2.real
# l_diff = l_diff.real
# v_2 = v_2.real
v_l_diff = v_l_diff.real
a_2 = a_2.real

# print(np.max(l_2), np.max(l_diff), np.max(v_2), np.max(v_l_diff), np.max(a_2))
# print(np.min(l_2), np.min(l_diff), np.min(v_2), np.min(v_l_diff), np.min(a_2))
max_val = max(np.max(l_2), np.max(v_l_diff), np.max(a_2))
min_val = min(np.min(l_2), np.min(v_l_diff), np.min(a_2))
extremum = max(abs(min_val), abs(max_val))
vmin = -1
vmax = 1

l_2 = normalize(l_2, -extremum, extremum, vmin, vmax)
# l_diff = normalize(l_diff, -extremum, extremum, vmin, vmax)
# v_2 = normalize(v_2, -extremum, extremum, vmin, vmax)
v_l_diff = normalize(v_l_diff, -extremum, extremum, vmin, vmax)
a_2 = normalize(a_2, -extremum, extremum, vmin, vmax)

# Create a figure with three subplots in one row
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, 
                        gridspec_kw={'wspace': 0.15, 'width_ratios': [1, 1, 1.094]})
fraction = 0.046
pad = 0.04
extent = np.array([0, n_size[0], n_size[0], 0])  # * base.pixel_size
cmap = 'seismic'

ax0 = axs[0]
im0 = ax0.imshow(a_2, cmap=cmap, extent=extent, 
                 norm=colors.SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
ax0.set_title('$A$')
kwargs0 = dict(transform=ax0.transAxes, ha='center', va='center', 
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', 
                        alpha=0.5, linewidth=0.0))
# ax0.text(0.12, 0.6, '$A_{11}$', **kwargs0)
# ax0.text(0.62, 0.6, '$A_{12}$', **kwargs0)
# ax0.text(0.12, 0.1, '$A_{21}$', **kwargs0)
# ax0.text(0.62, 0.1, '$A_{22}$', **kwargs0)

# ax0.text(0.1, 0.58, '$A_{11}$', **kwargs0)
# ax0.text(0.77, 0.79, '$A_{12}$', **kwargs0)
# ax0.text(0.23, 0.21, '$A_{21}$', **kwargs0)
# ax0.text(0.6, 0.08, '$A_{22}$', **kwargs0)

ax0.text(0.09, 0.57, '$A_{11}$', **kwargs0)
ax0.text(0.9, 0.92, '$A_{12}$', **kwargs0)
ax0.text(0.09, 0.07, '$A_{21}$', **kwargs0)
ax0.text(0.9, 0.42, '$A_{22}$', **kwargs0)

# ax0.text(0.18, 0.64, '$A_{11}$', **kwargs0)
# ax0.text(0.84, 0.84, '$A_{12}$', **kwargs0)
# ax0.text(0.18, 0.15, '$A_{21}$', **kwargs0)
# ax0.text(0.84, 0.34, '$A_{22}$', **kwargs0)

ax1 = axs[1]
im1 = ax1.imshow(l_2, cmap=cmap, extent=extent, 
                 norm=colors.SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
ax1.set_title('$L$')

ax2 = axs[2]
im2 = ax2.imshow(v_l_diff, cmap=cmap, extent=extent, 
                 norm=colors.SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
ax2.set_title('$V$')
kwargs2 = dict(transform=ax2.transAxes, ha='center', va='center', 
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', 
                        alpha=0.5, linewidth=0.0))
ax2.text(0.18, 0.64, '$C_{11}$', **kwargs2)
ax2.text(0.84, 0.84, '$A_{12}$', **kwargs2)
ax2.text(0.18, 0.15, '$A_{21}$', **kwargs2)
ax2.text(0.84, 0.34, '$C_{22}$', **kwargs2)
fig.colorbar(im2, ax=ax2, fraction=fraction, pad=pad, ticks=[-1, -0.1, 0, 0.1, 1], format='%.1f')

# Add text boxes with labels (a), (b), (c), ...
labels = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axs.flat):
    ax.text(0.5, -0.23, labels[i], transform=ax.transAxes, ha='center')
    ax.axhline(20, color='gray', linestyle='--', alpha=0.5)  # subdomain demarcation
    ax.axvline(20, color='gray', linestyle='--', alpha=0.5)  # subdomain demarcation
    ax.set_yticks(np.arange(0, 41, 10))
    ax.set_xticks(np.arange(0, 41, 10))

plt.savefig(filename, bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')
