"""
Run Helmholtz example
=====================
Example script to run a simulation of a point source in a 
random refractive index map using the Helmholtz equation.
"""

import numpy as np
from time import time

from wavesim.utilities.create_medium import random_permittivity
from wavesim.utilities.create_source import point_source
from wavesim.simulate import simulate
from wavesim.utilities import plot_computed

# Parameters
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = wavelength / 4  # Pixel size in micrometer (μm)
periodic = (False, False, False)
boundary_width = 5  # Boundary width in micrometer (μm)

# Size of the simulation domain
sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometer (μm)

# Create permittivity map with a random medium
permittivity = random_permittivity(sim_size, pixel_size)

# Create a point source at the center of the domain
source_values, source_position = point_source(
    position=sim_size//2,  # source center position in the center of the domain in micrometer (μm)
    pixel_size=pixel_size
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    periodic=periodic, 
    boundary_width=boundary_width, 
    n_domains=None
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u, pixel_size)
