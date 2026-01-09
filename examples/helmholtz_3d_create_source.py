
"""
Helmholtz 3D example to demonstrate utilities.create_source
===========================================================
This example simulates the propagation of a wave through a 3D medium 
with an interface of two media with different refractive indices, and 
can be simulated with one of the three primitive sources available as 
functions in utilities.create_source: point_source, plane_wave, or 
gaussian_beam.
"""

import numpy as np
from time import time

import sys
sys.path.append(".")
from wavesim.utilities import create_source, plot_computed
from wavesim.simulate import simulate

# Parameters
wavelength = 1.0  # wavelength in micrometer (μm)
pixel_size = wavelength / 8  # Pixel size in micrometer (μm)

# Create refractive index map
sim_size = np.array([10, 8, 5])
n_size = (sim_size / pixel_size).astype(int)  # Size of the simulation domain in pixels
refractive_index = np.ones(n_size, dtype=np.complex64)  # background refractive index of 1
refractive_index[:, refractive_index.shape[1] // 2:, :] = 2.0  # medium with refractive index of 2 in the lower half of the domain

# Generate source term (values and position) using one of the three create_source utility primitive sources. 
# Uncomment one of the sources below to use in the simulation

# # Option 1: Create a point source
# source_values, source_position = create_source.point_source(
#     position=[0, 0, 0],
#     pixel_size=pixel_size,
#     amplitude=3+0.5j
# )

# # Option 2: Create a plane wave source with incident angles theta and phi
# source_values, source_position = create_source.plane_wave(
#     shape=(sim_size[0]//2, sim_size[2]),  # source shape in micrometer (μm)
#     origin='center',  # source position is defined with respect to this origin
#     position=[0, 0, 0],  # source top left position in (x, y, z) in micrometer (μm)
#     source_plane='xz',  # The string is always sorted by alphabetical order, so 'xz' and 'zx' are both recognized as 'xz'.
#     pixel_size=pixel_size,
#     amplitude=2,
#     theta=np.pi / 4,  # angle of incidence of the source with respect to the source axis in radians
#     phi=np.pi / 6  # angle of incidence of the source in the plane orthogonal to the source axis in radians
#     wavelength=wavelength,  # wavelength must be defined for angled source, i.e. when theta or phi is not None
# )

# Option 3: Create a plane wave source with Gaussian intensity profile with incident angles theta and phi
source_values, source_position = create_source.gaussian_beam(
    shape=(sim_size[0]//2, sim_size[2]),  # source shape in micrometer (μm)
    origin='topleft',  # source position is defined with respect to this origin
    position=[0, 0, 0],  # source top left position in (x, y, z) in micrometer (μm)
    source_plane='xz',  # The string is always sorted by alphabetical order, so 'xz' and 'zx' are both recognized as 'xz'.
    pixel_size=pixel_size,
    amplitude=np.complex64(2.0, 0.0),
    theta=np.pi / 4,  # angle of incidence of the source with respect to the source axis in radians
    phi=np.pi / 6,  # angle of incidence of the source in the plane orthogonal to the source axis in radians
    wavelength=wavelength,  # wavelength must be defined for angled source, i.e. when theta or phi is not None
    alpha=3,  # width factor for Gaussian window
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=refractive_index**2, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=10,  # Boundary width in micrometer (μm) 
    periodic=(False, False, False)
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u, pixel_size, normalize_u=False, log=True)
