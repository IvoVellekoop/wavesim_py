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
    filename = 'paper_figures/fig3_decompose.pdf'
else:
    try:
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_figures/fig3_decompose.pdf'
    except FileNotFoundError:
        filename = 'fig3_decompose.pdf'

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

I = np.eye(np.prod(n_size), dtype=np.complex64)

l_plus1_o = full_matrix(domain_operator(domain_o, 'inverse_propagator'))[crop2roi].cpu().numpy()  # L + I
l_o = (l_plus1_o - I) / domain_o.scale - domain_o.shift.item()*I  # L = (L + I - I) / scaling

# Get matrices of A, L, and V operators decomposed into two domains
domain_2 = MultiDomain(permittivity=n, periodic=(True, True, True), wavelength=wavelength, 
                       pixel_size=pixel_size, n_domains=(2,1,1), debug=True, n_boundary=0)

b_2 = full_matrix(domain_operator(domain_2, 'medium')).cpu().numpy()  # B = I - V
l_plus1_2 = full_matrix(domain_operator(domain_2, 'inverse_propagator')).cpu().numpy()  # L + I

v_2 = I - b_2
l_2 = l_plus1_2 - I

l_o = (l_o + domain_2.shift.item()*I) * domain_2.scale

l_diff = l_o - l_2
v_l_diff = v_2 + l_diff
a_2 = l_2 + v_l_diff  # A = L + V

# Normalize matrices for visualization
l_2 = l_2.imag
v_l_diff = v_l_diff.imag
a_2 = a_2.imag

max_val = max(np.max(l_2), np.max(v_l_diff), np.max(a_2))
min_val = min(np.min(l_2), np.min(v_l_diff), np.min(a_2))
extremum = max(abs(min_val), abs(max_val))
vmin = -1
vmax = 1

l_2 = normalize(l_2, -extremum, extremum, vmin, vmax)
v_l_diff = normalize(v_l_diff, -extremum, extremum, vmin, vmax)
a_2 = normalize(a_2, -extremum, extremum, vmin, vmax)

# Create a figure with three subplots in one row
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, 
                        gridspec_kw={'wspace': 0.15, 'width_ratios': [1, 1, 1.094]})
fraction = 0.046
pad = 0.04
extent = np.array([0, n_size[0], n_size[0], 0])
cmap = 'seismic'

ax0 = axs[0]
im0 = ax0.imshow(a_2, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax0.set_title('$A$')
kwargs0 = dict(transform=ax0.transAxes, ha='center', va='center', 
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', 
                        alpha=0.5, linewidth=0.0))
ax0.text(0.13, 0.605, '$A_{11}$', **kwargs0)
ax0.text(0.75, 0.75, '$A_{12}$', **kwargs0)
ax0.text(0.25, 0.25, '$A_{21}$', **kwargs0)
ax0.text(0.87, 0.395, '$A_{22}$', **kwargs0)

ax1 = axs[1]
im1 = ax1.imshow(l_2, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax1.set_title('$L$')
kwargs1 = dict(transform=ax1.transAxes, ha='center', va='center', 
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', 
                        alpha=0.5, linewidth=0.0))
ax1.text(0.13, 0.605, '$L_{11}$', **kwargs1)
ax1.text(0.87, 0.395, '$L_{22}$', **kwargs1)

ax2 = axs[2]
im2 = ax2.imshow(v_l_diff, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax2.set_title('$V$')
kwargs2 = dict(transform=ax2.transAxes, ha='center', va='center', 
              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', 
                        alpha=0.5, linewidth=0.0))
ax2.text(0.13, 0.605, '$V_{11}$', **kwargs2)
ax2.text(0.75, 0.75, '$A_{12}$', **kwargs2)
ax2.text(0.25, 0.25, '$A_{21}$', **kwargs2)
ax2.text(0.87, 0.395, '$V_{22}$', **kwargs2)
fig.colorbar(im2, ax=ax2, fraction=fraction, pad=pad)

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
print(f'Saved: {filename}')
