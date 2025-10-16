"""
Helmholtz 1D analytical test
============================
Test to compare the result of WaveSim to analytical results. 
Compare 1D free-space propagation with analytic solution.
"""

import torch
import numpy as np
from time import time
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the wavesim iteration
from wavesim.utilities import analytical_solution, preprocess, relative_error
from __init__ import plot


# Parameters
wavelength = 1.  # wavelength in micrometer (um)
n_size = (256, 1, 1)  # size of simulation domain (in pixels in x, y, and z direction)
n = np.ones(n_size, dtype=np.complex64)  # permittivity (refractive index²) map
boundary_widths = 16  # width of the boundary in pixels

# return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n**2, boundary_widths)  # permittivity is n², but uses the same variable n

# Source term. This way is more efficient than dense tensor
indices = torch.tensor([[0 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of the domain
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# Set up the domain operators (HelmholtzDomain() or MultiDomain() depending on number of domains)
# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, True, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=(3, 1, 1))

# Run the wavesim iteration and get the computed field
start = time()
u_computed, iterations, residual_norm = run_algorithm(domain, source, max_iterations=2000)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_computed = u_computed.squeeze().cpu().numpy()[boundary_widths:-boundary_widths]
u_ref = analytical_solution(n_size[0], domain.pixel_size, wavelength)

# Compute relative error with respect to the analytical solution
re = relative_error(u_computed, u_ref)
print(f'Relative error: {re:.2e}')
threshold = 1.e-3
assert re < threshold, f"Relative error higher than {threshold}"

# Plot the results
plot(u_computed, u_ref, re)
