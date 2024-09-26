"""
This script generates a figure demonstrating the fast convolution of the Laplacian with a point source
for a 1-domain and 2-domain case, and the construction of the correction matrix A_{12}.

The figure consists of three subplots:
- Subplot (a): Fast convolution with a point source for 1-domain case
- Subplot (b): Fast convolution with a point source for 2-domain case, where the fast convolution is performed over 1
                subdomain. The subplot also shows the difference between the unwrapped and wrapped fields.
- Subplot (c): Correction matrix A_{12}, a non-cyclic convolution matrix that computes the wrapping artifacts.

The figure is saved as a PDF file.
"""

# import packages
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc, rcParams, colors

sys.path.append(".")
sys.path.append("..")
from wavesim.domain import Domain
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.utilities import normalize, pad_boundaries, preprocess

font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_figures', exist_ok=True)
    filename = 'paper_figures/fig3_correction_matrix.pdf'
else:
    try:
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_figures/fig3_correction_matrix.pdf'
    except FileNotFoundError:
        filename = 'fig3_correction_matrix.pdf'


def coordinates_f_sq(n_, pixel_size=0.25):
    """ Calculate the coordinates in the frequency domain and returns squared values
    :param n_: Number of points
    :param pixel_size: Pixel size. Defaults to 0.25.
    :return: Tensor containing the coordinates in the frequency domain """
    return (2 * torch.pi * torch.fft.fftfreq(n_, pixel_size))**2

# Define problem parameters
boundary_widths = 0.
n_size = 40

# Fast convolution of Laplacian with point source

# Case 1: 1 domain
d = int(n_size/2 - 1)  # 1 point before the center
# Point source
side = torch.zeros(n_size)
side[d] = 1.0
# Fast convolution result
fc1 = (torch.fft.ifftn(coordinates_f_sq(n_size) 
                       * torch.fft.fftn(side))).real.cpu().numpy()  # discard tiny imaginary part due to numerical errors

# Case 2: 2 domains. Fast convolution of Laplacian with point source over 1 subdomain
# Point source
side2 = torch.zeros(n_size // 2)
side2[d] = 1.0
# discard tiny imaginary part due to numerical errors in the fast conv result
fc2 = (torch.fft.ifftn(coordinates_f_sq(n_size // 2)
                       * torch.fft.fftn(side2))).real.cpu().numpy()
fc2 = np.concatenate((fc2, np.zeros_like(fc2)))  # zero padding for the 2nd domain
diff_ = fc1 - fc2  # difference between unwrapped and wrapped fields

# construct a non-cyclic convolution matrix that computes the wrapping artifacts only
t = 8  # number of correction points for constructing correction matrix
a_12 = np.zeros((t, t), dtype=np.complex64)
for r in range(t):
    a_12[r, :] = diff_[r:r + t]
a_12 = np.flip(a_12.real, axis=0)  # flip the matrix

# Normalize matrices for visualization
min_val = min(fc1.min(), fc2.min(), diff_.min(), a_12.min())
max_val = max(fc1.max(), fc2.max(), diff_.max(), a_12.max())
extremum = max(abs(min_val), abs(max_val))
vmin = -1
vmax = 1

fc1 = normalize(fc1, -extremum, extremum, vmin, vmax)
fc2 = normalize(fc2, -extremum, extremum, vmin, vmax)
diff_ = normalize(diff_, -extremum, extremum, vmin, vmax)
a_12 = normalize(a_12, -extremum, extremum, vmin, vmax)

# Plot limits
c = n_size//2  # center of the domain

# Plot
fig = plt.figure(figsize=(12, 6), layout='constrained')
gs = GridSpec(2, 2, figure=fig)

# 1-domain case
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fc1, 'k')  # 1-domain, no wrapping field
ax1.set_title(r'$\nabla^2$ kernel in infinite domain')
ax1.set_xlim([-1, n_size])
ax1.set_xticks(np.arange(0, n_size + 1, 5))
ax1.set_xticklabels([])
ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
ax1.text(0.5, -0.15, '(a)', transform=ax1.transAxes, ha='center')

# 2-domain case
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(fc2, 'k', label='Field')  # 2-domain, with wrapping field
ax2.axvspan(c, n_size - 1, color='gray', alpha=0.3)  # Patch to demarcate 2nd subdomain

# difference between unwrapped and wrapped fields
ax2.plot(diff_[:t], 'r--', label='Corrections')  # wrapping correction
ax2.text(0.14, 0.5, '$C_{11}$', color='r', transform=ax2.transAxes, ha='center')

ax2.plot(np.arange(c, c + t), diff_[c:c + t], 'r--')  # transfer correction
ax2.text(0.62, 0.23, '$A_{12}$', color='r', transform=ax2.transAxes, ha='center')

ax2.set_xlim([-1, n_size])
ax2.set_xticks(np.arange(0, n_size + 1, 5))
ax2.set_xlabel('x')
ax2.text(0.26, 0.09, 'Subdomain 1', transform=ax2.transAxes, ha='center')
ax2.text(0.76, 0.09, 'Subdomain 2', transform=ax2.transAxes, ha='center')
ax2.set_title(r'$\nabla^2$ kernel in periodic subdomains')
ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
ax2.legend()
ax2.text(0.5, -0.36, '(b)', transform=ax2.transAxes, ha='center')

# Correction matrix
ax3 = fig.add_subplot(gs[:, 1])
im3 = ax3.imshow(a_12, cmap='seismic', extent=[0, t, 0, t], norm=colors.CenteredNorm())
cb3 = plt.colorbar(mappable=im3, fraction=0.05, pad=0.01, ax=ax3)
ax3.set_title('$A_{12}$')
ax3.text(0.5, -0.15, '(c)', transform=ax3.transAxes, ha='center')

plt.savefig(filename, bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')

# Print min and max a_12 values. min/max should be <<1% (for truncation to make sense)
min_a_12 = a_12[0, -1]
max_a_12 = a_12[-1, 0]
percent = (min_a_12/max_a_12) * 100
print(f'Wrapping artifact amplitudes: Min {min_a_12:.3f}, Max {max_a_12:.3f}')
print(f'Min/Max of A_{12} = {percent:.2f} %')
assert percent < 1, f"Min/Max of A_{12} ratio exceeds 1%: {percent:.2f} %"
