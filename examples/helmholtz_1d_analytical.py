"""
Helmholtz 1D analytical test
============================
Test to compare the result of Wavesim to analytical results. 
Compare 1D free-space propagation with analytic solution.
"""

import numpy as np
from time import time

from wavesim.utilities.create_source import point_source
from wavesim.simulate import simulate
from wavesim.utilities import all_close, analytical_solution, relative_error, plot_computed_and_reference

# Parameters
wavelength = 0.5  # wavelength in micrometer (μm)
pixel_size = wavelength / 10  # pixel size in micrometer (μm)

# Create a refractive index map
sim_size = 128  # size of simulation domain in x direction in micrometer (μm)
n_size = (int(sim_size / pixel_size), 1, 1)  # We want to set up a 1D simulation, so y and z are 1.
permittivity = np.ones(n_size, dtype=np.complex64)  # permittivity (refractive index squared) of 1

# Create a point source at the center of the domain
source_values, source_position = point_source(
    position=[sim_size//2, 0, 0],  # source center position in the center of the domain in micrometer (μm)
    pixel_size=pixel_size
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=5,  # Boundary width in micrometer (μm) 
    periodic=(False, True, True)  # Periodic boundary conditions in the y and z directions
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# Compute the analytical solution
c = np.arange(0, sim_size, pixel_size)
c = c - c[source_position[0]]
u_ref = analytical_solution(c, wavelength)

# Compute relative error with respect to the analytical solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re)

threshold = 1.0e-3
assert re < threshold, f"Relative error higher than {threshold}"
assert all_close(u, u_ref, rtol=4e-2)
