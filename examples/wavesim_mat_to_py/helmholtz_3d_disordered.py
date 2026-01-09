""" 
Helmholtz 3D disordered medium test
===================================
Test for propagation in a 3D disordered medium. 
Compare with reference solution (matlab repo result). 
"""

import numpy as np
from time import time
from scipy.io import loadmat

from wavesim.utilities.create_source import point_source
from wavesim.simulate import simulate
from wavesim.utilities import relative_error, plot_computed_and_reference

# Parameters
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = wavelength / 4  # Pixel size in micrometer (μm)

# Load the refractive index map from a .mat file and square it to get permittivity
permittivity = np.ascontiguousarray(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["n3d_disordered"]) ** 2
sim_size = np.asarray(permittivity.shape) * pixel_size

# Create a point source at the center of the domain
source_values, source_position = point_source(
    position=sim_size / 2 - pixel_size,  # Source center position in the center of the domain in micrometer (μm)
    pixel_size=pixel_size
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=2,  # Boundary width in micrometer (μm)
    periodic=(False, False, False)
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u3d_disordered"])

# Compute relative error with respect to the reference solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re)

threshold = 1.0e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"
# assert all_close(u, u_ref, rtol=5.0e-3)  # todo: relative error is too high. Absolute error is fine
