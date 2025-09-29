""" 
Maxwell simulate
====================
This example demonstrates all the arguments of the simulate wrapper 
function through an example solving Maxwell's equations.
"""

import numpy as np
from cupy import asnumpy
from time import time
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from wavesim.utilities.create_medium import random_permittivity
from wavesim.utilities.create_source import gaussian_beam
from wavesim.simulate import simulate
from examples import plot_computed


def simulation_callback(domain, iteration, x, residual_norm, **kwargs):
    """
    Callback function to log simulation progress and control loop execution.

    Args:
        domain: The simulation domain.
        iteration (int): The current iteration number.
        x: The current solution vector.
        residual_norm (float): The residual norm at the current iteration.
        **kwargs: Additional arguments passed to the callback.

    Returns:
        bool: True to continue the loop, False to stop.
    """
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Residual Norm = {residual_norm:.2e}")

    # Stop the loop if iteration exceeds 10000
    if iteration > 10000:
        return False

    return True


# Parameters
wavelength = 1  # Wavelength in micrometer (μm)
pixel_size = wavelength / 8  # Pixel size in micrometer (μm)

# Size of the simulation domain
sim_size = np.array([8, 5, 10])  # Simulation size in micrometer (μm)

# Create permittivity map
permittivity = random_permittivity(sim_size, pixel_size, seed=42)

# Create a plane wave source with Gaussian intensity profile
source_values, source_position = gaussian_beam(
    shape=(sim_size[1], sim_size[2]),  # source shape in micrometer (μm)
    origin='topleft',  # source position is defined with respect to this origin
    position=[0, 0, 0, 0],  # source top left position in (x, y, z) in micrometer (μm)
    source_plane='yz',  # The string is always sorted by alphabetical order, so 'xz' and 'zx' are both recognized as 'xz'.
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
    periodic=(False, False, False), 
    use_gpu=True,
    max_iterations=10000,
    threshold=1.e-7,  # tolerance for convergence. Default is 1.0e-6.
    alpha=0.9,  # relaxation factor for the preconditioned Richardson method. Default is 0.75.
    full_residuals=True,  # if true return full residuals, and if False, return only the final residual norm. Default is False.
    crop_boundaries=True,  # if true crop the boundaries of the field to remove the absorbing boundaries. Default is True.
    callback=simulation_callback
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm[-1]:.2e})")

# Plot the results (with absorbing boundaries)
plot_computed(u[0, ...], pixel_size, log=True)

# Plot the residual norm with iterations
plt.semilogy(np.arange(1, iterations+1), [asnumpy(r) for r in residual_norm])
plt.xlabel('Iterations')
plt.ylabel('Residual norm')
plt.show()
