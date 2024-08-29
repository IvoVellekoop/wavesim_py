""" 
Helmholtz 3D homogeneous medium test
====================================
Test for propagation in a 3D homogeneous medium. 
Compare with reference solution (matlab repo result). 
"""

import os
import torch
import numpy as np
from time import time
from scipy.io import loadmat
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the wavesim iteration
from wavesim.utilities import preprocess, relative_error
from __init__ import plot

if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')


# Parameters
wavelength = 1.  # Wavelength in micrometers
n_size = (128, 128, 128, 1)  # Size of the domain in pixels (x, y, z, 1)
n = np.ones(tuple(n_size), dtype=np.complex64)  # Refractive index map
boundary_widths = 50  # Width of the boundary in pixels

# add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n, boundary_widths)

# Source term. This way is more efficient than dense tensor
indices = torch.tensor([[int(v/2 - 1) + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of the domain
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# Set up the domain operators (HelmholtzDomain() or MultiDomain() depending on number of domains)
# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, False, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength,
#                      n_domains=(2, 1, 1))

# Run the wavesim iteration and get the computed field
start = time()
u_computed, iterations, residual_norm = run_algorithm(domain, source, max_iterations=2000)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_computed = u_computed.squeeze().cpu().numpy()[*([slice(boundary_widths, -boundary_widths)]*3)]

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('examples/matlab_results.mat')[f'u3d_{n_size[0]}_{n_size[1]}_{n_size[2]}_bw_20_24_32'])

re = relative_error(u_computed, u_ref)
print(f'Relative error: {re:.2e}')
threshold = 1.e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"

# Plot the results
plot(u_computed, u_ref, re)
