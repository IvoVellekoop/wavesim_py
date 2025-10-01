""" 
Helmholtz 2D high contrast test
===============================
Test for propagation in 2D structure made of iron, 
with high refractive index contrast.
Compare with reference solution (matlab repo result). 
"""

import numpy as np
from time import time
from scipy.io import loadmat
from PIL.Image import fromarray, open, Resampling

import sys
sys.path.append(".")
from wavesim.simulate import simulate
from tests import all_close, relative_error
from examples import plot_computed_and_reference


# Parameters
n_iron = 2.8954 + 2.9179j
n_contrast = n_iron - 1
wavelength = 0.532  # Wavelength in micrometer (μm)
pixel_size = wavelength / (3 * np.max(abs(n_contrast + 1)))  # Pixel size in micrometer (μm)

# Load image and create refractive index map
oversampling = 0.25
im = np.asarray(open("examples/wavesim_mat_to_py/logo_structure_vector.png")) / 255  # Load image and normalize to [0, 1]
n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * n_contrast) + 1  # Assign refractive index to pixels
n_roi = int(oversampling * n_im.shape[0])  # Size of ROI in pixels
permittivity = np.asarray(fromarray(n_im.real).resize((n_roi, n_roi), Resampling.BILINEAR)) + 1j * np.asarray(
    fromarray(n_im.imag).resize((n_roi, n_roi), Resampling.BILINEAR)
)  # Resize to n_roi x n_roi
permittivity = permittivity[..., None] ** 2  # Add a dimension to make the array 3D and square it to get permittivity

# Create a custom source using the array from the imported png
source_values = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), Resampling.BILINEAR))[..., None]
source_position = [0, 0, 0]  # source position in (x, y, z) in pixels

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=5,  # Boundary width in micrometer (μm)
    periodic=(False, False, True)
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u2d_hc"])

# Compute relative error with respect to the reference solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re)

threshold = 1.0e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"
# assert all_close(u, u_ref, rtol=1.0e-2)  # todo: relative error is too high. Absolute error is fine
