"""
Maxwell 2D example to demonstrate utilities.create_medium
=========================================================
This example simulates the propagation of a wave through a 2D medium 
with rectangular block(s) of material(s) in a homogeneous background 
permittivity, by solving Maxwell's equations.
"""

import numpy as np
from time import time

import sys
sys.path.append(".")
from wavesim.utilities.create_source import gaussian_beam
from wavesim.utilities.create_medium import cuboids_permittivity
from wavesim.simulate import simulate
from examples import plot_computed

# Parameters
material_1 = 2.9277 + 3.8315j  # Iron
material_2 = 0.22769 + 6.4731j  # Gold 
wavelength = 1.0  # wavelength in micrometer (μm)
pixel_size = wavelength / (3 * max(abs(material_1), abs(material_2)))  # Pixel size in micrometer (μm)

# Create refractive index map
sim_size = np.array([6, 8])  # Simulation size in micrometer (μm)
permittivity = cuboids_permittivity(
    shape=sim_size, 
    pixel_size=pixel_size,
    origin='center',
    origin_size_material=[(sim_size//2-1, [2, 3], material_1), ([4, 6], [2, 2], material_2)]
)

# Create a plane wave source with Gaussian intensity profile
source_values, source_position = gaussian_beam(
    shape=(sim_size[0]),  # source shape, 1D in X direction, in micrometer (μm)
    origin='topleft',  # source position is defined with respect to this origin
    position=[2, 0, 0, 0],  # source top left position in (x, y, z) in micrometer (μm)
    source_plane='x',
    pixel_size=pixel_size,
    alpha=3,  # width factor for Gaussian window
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity[..., None], 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=10,  # Boundary width in micrometer (μm)
    periodic=(False, False, True)  # Periodic boundary conditions in the z direction
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u[2, ...], pixel_size, normalize_u=False)
