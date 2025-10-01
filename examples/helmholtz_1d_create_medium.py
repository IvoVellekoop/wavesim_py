"""
Helmholtz 1D example to demonstrate utilities.create_medium
===========================================================
This example simulates the propagation of a wave through a 1D medium 
with box-shaped block(s) of material(s) in a homogeneous background 
permittivity, by solving the Helmholtz equation.
"""

import numpy as np
from time import time

import sys
sys.path.append(".")
from wavesim.utilities.create_source import point_source
from wavesim.utilities.create_medium import cuboids_permittivity
from wavesim.simulate import simulate
from examples import plot_computed

#indicative materials. Optical refractive index at 1 μm (unless otherwise specified)
material_1 = 3.5750 + 0.00049020j  # Silicon
material_2 = 1.5184 + 0.0011681j  # PVP - Polyvinylpyrrolidone (a plastic)

# Parameters
wavelength = 1.0  # wavelength in micrometer (μm)
pixel_size = wavelength / (3 * max(abs(material_1), abs(material_2)))  # Pixel size in micrometer (μm)

# Create a refractive index map
sim_size = np.array([20])  # Simulation size in micrometer (μm)
permittivity = cuboids_permittivity(
    shape=sim_size, 
    pixel_size=pixel_size,
    origin='center',
    origin_size_material=[([4], [2], material_1), ([16], [3], material_2)]
)

# Create a point source at the center of the domain
source_values, source_position = point_source(
    position=[10, 0, 0],
    pixel_size=pixel_size
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity[..., None, None],  # Adding two axes to make permittivity 3D
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=10,  # Boundary width in micrometer (μm)
    periodic=(False, True, True)  # Periodic boundary conditions in the y and z directions
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u, pixel_size, normalize_u=False)
