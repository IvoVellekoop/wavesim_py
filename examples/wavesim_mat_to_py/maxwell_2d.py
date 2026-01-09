""" 
Maxwell's equations for 2D propagation.
=======================================
Test for Maxwell's equations for 2D propagation. 
Compare with reference solution (matlab repo result).
"""

import numpy as np
from time import time
from scipy.io import loadmat

from wavesim.utilities.create_source import gaussian_beam
from wavesim.simulate import simulate
from wavesim.utilities import relative_error, plot_computed_and_reference


# Parameters
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = wavelength / 8  # Pixel size in micrometer (μm)

# Size of the simulation domain
sim_size = np.array([16, 32])  # Simulation size in micrometer in micrometer (μm)
n_size = sim_size / pixel_size  # Size of the simulation domain in pixels
n_size = tuple(n_size.astype(int)) + (1,)  # Add 3rd dimension for z-axis

# Generate a refractive index map
epsilon1 = 1.0
epsilon2 = 2.0 ** 2
permittivity = np.full(n_size, epsilon1, dtype=np.complex64)  # medium 1
permittivity[n_size[0] // 2 :, ...] = epsilon2  # medium 2

# Create a plane wave source with Gaussian intensity profile with incident angle theta
source_values, source_position = gaussian_beam(
    shape=(sim_size[1]//2),  # source shape, 1D in Y direction, in micrometer (μm)
    origin='center',  # source position is defined with respect to this origin
    position=[2, 0, 8, 0],  # source center position in [polarization axis, x , y, z]. 2 for z-polarization. x, y, z in micrometer (μm)
    source_plane='y',
    pixel_size=pixel_size,
    theta=np.pi / 4, 
    wavelength=wavelength,  # wavelength must be defined for angled source, i.e. when theta or phi is not None
    alpha=3, 
)

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
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["maxwell_2d"])
u_ref = np.moveaxis(u_ref, -1, 0)  # polarization is in the last axis in MATLAB
u_ref = u_ref[(1, 0, 2), ...]  # x and y polarization are switched in MATLAB

# Compute relative error with respect to the reference solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u[2, ...], u_ref[2, ...], pixel_size, re)  # Plot the z-polarization component

threshold = 1.0e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"
