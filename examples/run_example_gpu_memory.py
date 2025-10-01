"""
Run Helmholtz (GPU memory) example
==================================
Example script to run a simulation of a point source in a
random refractive index map using the Helmholtz equation,
and report the GPU memory usage.
"""

import cupy as cp
import numpy as np
from time import time

import sys

sys.path.append(".")
from wavesim.engine import BlockArray, CupyArray, NumpyArray, SparseArray
from wavesim.utilities.create_medium import random_permittivity
from wavesim.utilities import add_absorbing_boundaries
from wavesim.helmholtzdomain import Helmholtz
from wavesim.iteration import preconditioned_richardson
from examples import plot_computed

# Enable memory pool management
pool = cp.get_default_memory_pool()

# Parameters
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = wavelength / 4  # Pixel size in micrometer (μm)
periodic = (False, False, False)
boundary_width = 5  # Boundary width in micrometer (μm)
boundary_width = int(boundary_width / pixel_size)  # Boundary width in pixels
boundary_widths = np.full((3, 2), boundary_width, dtype=np.int32)
boundary_widths[np.array(periodic)] = 0

# Size of the simulation domain
sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometer (μm)

initial_memory = pool.total_bytes() / 1024**3
initial_free_memory = pool.free_bytes() / 1024**3

# Create a permittivity (refractive index squared) map
permittivity = random_permittivity(sim_size, pixel_size)
permittivity, roi = add_absorbing_boundaries(NumpyArray(permittivity), boundary_widths, strength=1.0)
factory = CupyArray  # NumpyArray for CPU, CupyArray for GPU acceleration
permittivity = factory(permittivity)

# Create a point source at the center of the domain
source = SparseArray.point(at=np.asarray(permittivity.shape) // 2, shape=permittivity.shape, dtype=np.complex64)
source = factory(source)

n_domains = None  # number of domains in each direction
if n_domains is not None:
    permittivity = BlockArray(permittivity, n_domains=n_domains, factories=factory)
else:
    n_domains = (1, 1, 1)  # Only used for the file name

domain = Helmholtz(permittivity=permittivity, pixel_size=pixel_size, wavelength=wavelength, periodic=periodic)

# Record memory after domain creation
memory_after_domain = pool.total_bytes() / 1024**3
free_memory_after_domain = pool.free_bytes() / 1024**3

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = preconditioned_richardson(domain, source, max_iterations=2000)
sim_time = time() - start

# Memory reporting
print("\nGPU Memory Usage:")
print(f"Initial Memory            : {initial_memory:5.2f} GB")
print(f"Initial Free Memory       : {initial_free_memory:5.2f} GB")
print(f"After Domain              : {memory_after_domain:5.2f} GB")
print(f"Free Memory After Domain  : {free_memory_after_domain:5.2f} GB")
print(f"Peak Memory               : {pool.total_bytes() / 1024**3:5.2f} GB")
print(f"Released Memory           : {pool.free_bytes() / 1024**3:5.2f} GB")


# Cleanup intermediate results
del permittivity
pool.free_all_blocks()
final_memory = pool.total_bytes() / 1024**3
final_free_memory = pool.free_bytes() / 1024**3
print(f"Final Memory              : {final_memory:5.2f} GB")
print(f"Final Free Memory         : {final_free_memory:5.2f} GB")

print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

roi = (slice(boundary_widths[i][0], u.shape[i] - boundary_widths[i][1]) for i in range(3))
u = u[*roi].squeeze()

plot_computed(u, pixel_size)
