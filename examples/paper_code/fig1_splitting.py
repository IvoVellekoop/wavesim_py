"""
This script generates a figure showing the splitting of matrices A, L, and V.

It sets up the problem parameters, computes the matrices A, L, and V,
and visualizes them in a figure with four subplots:
- Subplot (a): Matrix A
- Subplot (b): Matrix L without wraparound artifacts
- Subplot (c): Matrix V
- Subplot (d): Matrix L with wraparound artifacts

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
from wavesim.utilities import normalize, add_absorbing_boundaries
from tests import domain_operator, full_matrix

font = {"family": "serif", "serif": ["Times New Roman"], "size": 13}
rc("font", **font)
rcParams["mathtext.fontset"] = "cm"

if os.path.basename(os.getcwd()) == "paper_code":
    os.chdir("..")
    os.makedirs("paper_figures", exist_ok=True)
    filename = "paper_figures/fig1_splitting.pdf"
else:
    try:
        os.makedirs("examples/paper_figures", exist_ok=True)
        filename = "examples/paper_figures/fig1_splitting.pdf"
    except FileNotFoundError:
        filename = "fig1_splitting.pdf"

# Define problem parameters
boundary_widths = 0
n_size = (40, 1, 1)
# Random refractive index distribution
torch.manual_seed(0)  # Set the random seed for reproducibility
n = torch.normal(mean=1.3, std=0.1, size=n_size, dtype=torch.float32) + 1j * abs(
    torch.normal(mean=0.05, std=0.02, size=n_size, dtype=torch.float32)
)
assert n.imag.min() >= 0, "Imaginary part of n is negative"

wavelength = 1.0
pixel_size = wavelength / 4
periodic = (True, True, True)

# Get matrices of A, L (with wrapping artifacts), and V operators
domain = Helmholtz(permittivity=n**2, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size, debug=True)
domain.set_source(0)
b = full_matrix(domain_operator(domain, "medium")).cpu().numpy()  # B = I - V
l_plus1 = full_matrix(domain_operator(domain, "inverse_propagator")).cpu().numpy()  # L + I

I = np.eye(np.prod(n_size), dtype=np.complex64)
v = I - b
l = l_plus1 - I

# Get matrices of A, L, and V for large domain without wraparound artifacts
boundary_widths = 100
n, boundary_array, crop2roi = add_absorbing_boundaries((n**2).numpy(), boundary_widths)
assert n.imag.min() >= 0, "Imaginary part of n is negative"
domain_o = Helmholtz(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size, debug=True)

crop2roi = (slice(boundary_array[0], -boundary_array[0]), slice(boundary_array[0], -boundary_array[0]))
b_o = full_matrix(domain_operator(domain_o, "medium"))[crop2roi].cpu().numpy()  # B = I - V
l_plus1_o = full_matrix(domain_operator(domain_o, "inverse_propagator"))[crop2roi].cpu().numpy()  # L + I

v_o = (I - b_o) / domain_o.scale + domain_o.shift.item() * I  # V = (I - B) / scaling
l_o = (l_plus1_o - I) / domain_o.scale - domain_o.shift.item() * I  # L = (L + I - I) / scaling

v_o = (v_o - domain.shift.item() * I) * domain.scale
l_o = (l_o + domain.shift.item() * I) * domain.scale

a_o = l_o + v_o  # A = L + V

# Normalize matrices for visualization
l = l.imag
v = v.imag
a_o = a_o.imag
l_o = l_o.imag

max_val = max(np.max(l), np.max(a_o), np.max(l_o), np.max(v))
min_val = min(np.min(l), np.min(a_o), np.min(l_o), np.min(v))
extremum = max(abs(min_val), abs(max_val))
vmin = -1
vmax = 1

l = normalize(l, -extremum, extremum, vmin, vmax)
v = normalize(v, -extremum, extremum, vmin, vmax)
a_o = normalize(a_o, -extremum, extremum, vmin, vmax)
l_o = normalize(l_o, -extremum, extremum, vmin, vmax)

# Create a figure with four subplots in one row
fig, axs = plt.subplots(
    1, 4, figsize=(12, 3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.15, "width_ratios": [1, 1, 1, 1.094]}
)
fraction = 0.046
pad = 0.04
extent = np.array([0, n_size[0], n_size[0], 0])  # * base.pixel_size
cmap = "seismic"

ax0 = axs[0]
im0 = ax0.imshow(a_o, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax0.set_title("$A$")

ax1 = axs[1]
im1 = ax1.imshow(l_o, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax1.set_title("$L$")

ax2 = axs[2]
im2 = ax2.imshow(v, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax2.set_title("$V$")

ax3 = axs[3]
im3 = ax3.imshow(l, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax3.set_title("$L$ with wraparound artifacts")
fig.colorbar(im3, ax=ax3, fraction=fraction, pad=pad)

# Add text boxes with labels (a), (b), (c), ...
labels = ["(a)", "(b)", "(c)", "(d)"]
for i, ax in enumerate(axs.flat):
    ax.text(0.5, -0.23, labels[i], transform=ax.transAxes, ha="center")
    ax.set_xticks(np.arange(0, 41, 10))
    ax.set_yticks(np.arange(0, 41, 10))

plt.savefig(filename, bbox_inches="tight", pad_inches=0.03, dpi=300)
plt.close("all")
print(f"Saved: {filename}")
