"""
This script generates a figure to demonstrate the truncationg of wrapping and transfer
corrections. The figure consists of three subpltos:

- Subplot (a): Matrix A with truncated corrections
- Subplot (b): Matrix L
- Subplot (c): Matrix V with truncated corrections

The figure is saved as a PDF file.
"""

# import packages
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, colors

sys.path.append(".")
sys.path.append("..")
from wavesim.helmholtzdomain import Helmholtz  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.utilities import normalize, add_absorbing_boundaries
from tests import domain_operator, full_matrix, relative_error

current_dir = os.path.dirname(__file__)
font = {"family": "serif", "serif": ["Times New Roman"], "size": 13}
rc("font", **font)
rcParams["mathtext.fontset"] = "cm"

if os.path.basename(os.getcwd()) == "paper_code":
    os.chdir("..")
    os.makedirs("paper_figures", exist_ok=True)
    filename = "paper_figures/fig4_truncate.pdf"
else:
    try:
        os.makedirs("examples/paper_figures", exist_ok=True)
        filename = "examples/paper_figures/fig4_truncate.pdf"
    except FileNotFoundError:
        filename = "fig4_truncate.pdf"

# Define problem parameters
n_size = (40, 1, 1)
# Random refractive index distribution
torch.manual_seed(0)  # Set the random seed for reproducibility
n = (
    torch.normal(mean=1.3, std=0.1, size=n_size, dtype=torch.float32)
    + 1j * abs(torch.normal(mean=0.05, std=0.02, size=n_size, dtype=torch.float32))
) ** 2
assert n.imag.min() >= 0, "Imaginary part of n is negative"

wavelength = 1.0
pixel_size = wavelength / 4
periodic = (True, True, True)

I = np.eye(np.prod(n_size), dtype=np.complex64)

# Get matrices of A, L, and V operators (with truncated corrections) decomposed into two domains
domain_c = MultiDomain(
    permittivity=n,
    periodic=(False, True, True),
    wavelength=wavelength,
    pixel_size=pixel_size,
    n_domains=(2, 1, 1),
    n_boundary=6,
    debug=True,
)

b_c = full_matrix(domain_operator(domain_c, "medium")).cpu().numpy()  # B = I - V
l_plus1_c = full_matrix(domain_operator(domain_c, "inverse_propagator")).cpu().numpy()  # L + I

v_c = I - b_c
l_c = l_plus1_c - I
a_c = l_c + v_c  # A = L + V

# Normalize matrices for visualization
a_c = a_c.imag
l_c = l_c.imag
v_c = v_c.imag

max_val = max(np.max(a_c), np.max(l_c), np.max(v_c))
min_val = min(np.min(a_c), np.min(l_c), np.min(v_c))
extremum = max(abs(min_val), abs(max_val))
vmin = -1
vmax = 1

a_c = normalize(a_c, -extremum, extremum, vmin, vmax)
v_c = normalize(v_c, -extremum, extremum, vmin, vmax)
l_c = normalize(l_c, -extremum, extremum, vmin, vmax)

# Plot (a) A, (b) L, and (c) V with truncated corrections

# Create a figure with three subplots in one row
fig, axs = plt.subplots(
    1, 3, figsize=(9, 3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.15, "width_ratios": [1, 1, 1.094]}
)
fraction = 0.046
pad = 0.04
extent = np.array([0, n_size[0], n_size[0], 0])
cmap = "seismic"

ax0 = axs[0]
im0 = ax0.imshow(a_c, cmap=cmap, extent=extent, norm=colors.SymLogNorm(linthresh=0.7, vmin=vmin, vmax=vmax))
ax0.set_title("$A$ (truncated corrections)")

ax1 = axs[1]
im1 = ax1.imshow(l_c, cmap=cmap, extent=extent, norm=colors.SymLogNorm(linthresh=0.7, vmin=vmin, vmax=vmax))
ax1.set_title("$L$")

ax2 = axs[2]
im2 = ax2.imshow(v_c, cmap=cmap, extent=extent, norm=colors.SymLogNorm(linthresh=0.7, vmin=vmin, vmax=vmax))
ax2.set_title("$V$ (truncated corrections)")
fig.colorbar(im2, ax=ax2, fraction=fraction, pad=pad, ticks=[-1, -0.5, 0, 0.5, 1], format="%.1f")

# Add text boxes with labels (a), (b), (c), ...
labels = ["(a)", "(b)", "(c)", "(d)"]
for i, ax in enumerate(axs.flat):
    ax.text(0.5, -0.23, labels[i], transform=ax.transAxes, ha="center")
    ax.axhline(20, color="gray", linestyle="--", alpha=0.5)  # subdomain demarcation
    ax.axvline(20, color="gray", linestyle="--", alpha=0.5)  # subdomain demarcation
    ax.set_yticks(np.arange(0, 41, 10))
    ax.set_xticks(np.arange(0, 41, 10))

plt.savefig(filename, bbox_inches="tight", pad_inches=0.03, dpi=300)
plt.close("all")
print(f"Saved: {filename}")
