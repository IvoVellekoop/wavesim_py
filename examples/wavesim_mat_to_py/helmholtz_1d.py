"""
Helmholtz 1D example with glass plate
=====================================
Test for 1D propagation through glass plate.
Compare with reference solution (matlab repo result).
"""

import numpy as np
from time import time
from scipy.io import loadmat

from wavesim.utilities.create_source import point_source
from wavesim.simulate import simulate
from wavesim.utilities import all_close, relative_error, plot_computed_and_reference

# Parameters
wavelength = 1.0  # wavelength in micrometer (μm)
pixel_size = wavelength / 4  # pixel size in micrometer (μm)

# Create refractive index map
n_size = (256, 1, 1)  # size of simulation domain in the x, y, and z direction in pixels. We want to set up a 1D simulation, so y and z are 1.
refractive_index = np.ones(n_size, dtype=np.complex64)  # background refractive index of 1
refractive_index[99:130] = 1.5  # glass plate with refractive index of 1.5 in the defined region

# Create a point source
source_values, source_position = point_source(
    position=[0, 0, 0],  # source center position at the (starting) edge of the domain in micrometer (μm)
    pixel_size=pixel_size
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=refractive_index**2, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=5,  # boundary width in micrometer (μm)
    periodic=(False, True, True) 
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u_1d"])

# Compute relative error with respect to the reference solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re)

threshold = 1.0e-3
assert re < threshold, f"Relative error higher than {threshold}"
assert all_close(u, u_ref, rtol=4e-2)
