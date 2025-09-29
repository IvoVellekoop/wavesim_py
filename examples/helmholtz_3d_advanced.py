"""
Helmholtz 3D example to demonstrate simulate
============================================
This example demonstrates advanced funcationalities of the simulate
helper function, by simulating the propagation of multiple sources 
through a 3D medium with box-shaped block(s) of material(s) in a 
homogeneous background permittivity, by solving the Helmholtz equation.
The example also defines and used a simulation_callback.
"""

import numpy as np
from time import time

import sys
sys.path.append(".")
from wavesim.utilities import create_medium, create_source
from wavesim.simulate import simulate
from examples import plot_computed

#indicative materials. Optical refractive index at 1 μm (unless otherwise specified)
material_ITO = np.complex64(1.3061 + 0.012939j) # In2O3-SnO2 (Indium tin oxide, ITO)
material_PVP = np.complex64(1.5184 + 0.0011681j)  # PVP - Polyvinylpyrrolidone (a plastic)
material_Si  = np.complex64(3.5750 + 0.00049020j)  # Silicon
# 2.8954 + 2.9179j  # Iron (at wavelength = 0.532 μm)
material_Fe  = np.complex64(2.9277 + 3.8315j) # Iron
material_Au  = np.complex64(0.22769 + 6.4731j)  # Gold 


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
wavelength = 1.0  # Wavelength in micrometer (μm)
pixel_size = wavelength / 4  # Pixel size in micrometer (μm). Divided by (at least) 3 the max of the absolute of the highest refractive index. wavelength / 4 is recommended.

# Size of the simulation domain
sim_size = np.array([12, 10, 15])  # Simulation size in micrometer (μm)

# Create permittivity map with orthogonal shapes or a random medium, with all relevant inputs in micrometer (μm)

# Create a 3D permittivity map with a single block of a material
material = material_PVP
# pixel_size = wavelength / (3 * np.max(abs(material)))
permittivity = create_medium.cuboids_permittivity(
    sim_size, 
    pixel_size=pixel_size,
    origin='center',
    origin_size_material=[([6, 2, 7.5], [6, 2, 7.5], material)]
)

# # Create a 3D permittivity map with multiple 3D orthogonal blocks of different materials
# origin_size_material = []
# origin_size_material.append( ([6, 7, 7.5], [6, 2, 7.5], material_PVP) ) # PVP - Polyvinylpyrrolidone (a plastic)
# origin_size_material.append( ([6, 4, 7.5], [6, 2, 7.5], material_Si) ) # Silicon
# permittivity = create_medium.cuboids_permittivity(
#     shape=sim_size, 
#     pixel_size=pixel_size, 
#     origin='center',
#     origin_size_material=origin_size_material)

# # or just use a random medium
# permittivity = create_medium.random_permittivity(sim_size, pixel_size=pixel_size)

# Sources term. A list of different sample sources at different locations is prepared. One or more can be combined in the simulation.
sourceList =[]
point_source, ps_position = create_source.point_source(
    position=sim_size//2,  # source position in the center of the domain in micrometer (μm)
    pixel_size=pixel_size
)  # Point source
gaussian_source, gs_position = create_source.gaussian_beam(
    shape=(sim_size[0], sim_size[2]), 
    origin='topleft',  # source position is defined with respect to this origin
    position=[0, 0, 0],
    source_plane='xz',  # The string is always sorted by alphabetical order, so 'xz' and 'zx' are both recognized as 'xz'.
    pixel_size=pixel_size,
    alpha=3,  # width factor for Gaussian window
    amplitude=np.complex64(2.0, 0.0),
)  # Gaussian beam in XZ plane at Y start

# append the sources you want to use in the simulation
# sourceList.append( (point_source, ps_position) ) # Point source in the center of the domain
sourceList.append( (gaussian_source, gs_position) ) # Gaussian beam in XZ plane at Y start 
#sourceList.append( (gaussian_source, [0,permittivity.shape[1]//2,0]) ) # Gaussian beam in XZ plane at Y middle
#sourceList.append( (gaussian_source, [0,permittivity.shape[1],0]) ) # Gaussian beam in XZ plane at Y end

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=sourceList, 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    periodic=(False, False, False), 
    boundary_width=5,  # Boundary width in micrometer (μm)
    n_domains=None, 
    callback=simulation_callback
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

plot_computed(u, pixel_size)
