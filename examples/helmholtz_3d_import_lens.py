"""
Helmholtz 3D lens test
=======================
3D simulation for propagation through a lens, 
with a point source at one edge of the domain.
"""

import numpy as np
from time import time
from scipy.io import loadmat

import sys
sys.path.append(".")
from wavesim.utilities.create_source import gaussian_beam
from wavesim.simulate import simulate
from examples import plot_computed

# Parameters
wavelength = 1.  # Wavelength in micrometer (μm)
pixel_size = wavelength / 4  # Pixel size in micrometer (μm)

# Load refractive index map from a .mat file and square it to get permittivity.
# Contains a lens with a radius of 2 um and refractive index of 1.5. The background refractive index is 1.0.
permittivity = np.ascontiguousarray(loadmat('examples/test_lens.mat')['w']).astype(np.complex64) ** 2

sim_size = np.asarray(permittivity.shape) * pixel_size

# Create a plane wave source with Gaussian intensity profile
source_values, source_position = gaussian_beam(
    shape=(sim_size[1], sim_size[2]),  # Shape of the source, 2D in YZ plane, in micrometer (μm)
    origin='topleft',  # source position is defined with respect to this origin
    position=[0, 0, 0],  # source top left position in (x, y, z) in micrometer (μm)
    source_plane='yz',  # The string is always sorted by alphabetical order, so 'yz' and 'zy' are both recognized as 'yz'.
    pixel_size=pixel_size,
    alpha=3, # width factor for Gaussian window
)
print(source_position)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ],
    wavelength=wavelength, 
    pixel_size=pixel_size
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u, pixel_size)
