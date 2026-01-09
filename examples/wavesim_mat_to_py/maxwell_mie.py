"""
Maxwell's equations for Mie scattering.
=======================================
Test for Maxwell's equations for Mie scattering. 
Compare with reference solution (matlab repo result).
"""

import numpy as np
from time import time
from scipy.io import loadmat
from scipy.signal.windows import tukey

from wavesim.simulate import simulate
from wavesim.utilities.create_medium import sphere_permittivity
from wavesim.utilities import relative_error, plot_computed_and_reference

# Parameters
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = wavelength / 5  # Pixel size in micrometer (μm)
boundary_width = 4  # Boundary width in micrometer (μm)
periodic = (False, False, False)
n_size = (120, 120, 120)  # Size of the simulation domain in pixels

# Generate a refractive index map
sphere_radius = 6.0
sphere_epsilon = 1.2**2  # Permittivity of the sphere
bg_epsilon = 1**2  # Permittivity of the background
permittivity, x_r, y_r, z_r = sphere_permittivity(n_size, pixel_size, sphere_radius, sphere_epsilon, bg_epsilon)

# Create a custom source
# calculate source prefactor
k = np.sqrt(bg_epsilon) * 2 * np.pi / wavelength
prefactor = 2 * k / (1.0j * pixel_size)

# Linearly-polarized apodized plane wave
# (filter edges to reduce diffraction and the source extends into the absorbing boundaries)
src0 = tukey(n_size[0], 0.5).reshape((1, -1, 1, 1))
src1 = tukey(n_size[1], 0.5).reshape((1, 1, -1, 1))
source_values = (prefactor * np.exp(1.0j * k * z_r[0, 0, 0]) * src0 * src1).astype(np.complex64)
source_position = [1, 0, 0, 0]  # 1 for y-polarization

# Run the wavesim iteration and get the computed field
start = time()
u_sphere, iterations, residual_norm = simulate(
    permittivity=permittivity,
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=boundary_width, 
    periodic=periodic
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# Run similar simulation, but without the medium (to get the background field)
permittivity_bg = np.full(n_size, bg_epsilon, dtype=np.complex64)

start = time()
u_bg, iterations, residual_norm = simulate(
    permittivity=permittivity_bg,
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength,
    pixel_size=pixel_size,
    boundary_width=boundary_width,
    periodic=periodic
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# Subtract background to get the scattered field
u = (u_sphere - u_bg)[1, n_size[0]//2, ...]  # 2D section of the y-polarization component to compare with analytical solution

# load results from matlab wavesim for comparison and validation
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["maxwell_mie"])

# Compute relative error with respect to the reference solution
re = relative_error(u, u_ref)
print(f"Relative error: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re)

threshold = 2.0e-2  # In this simulation, the maximum accuracy is limited by the discretization of the Mie sphere, and can be further improved by performing the simulation on a finer grid. (Osnabrugge et al., 2021)
assert re < threshold, f"Relative error {re} higher than {threshold}"