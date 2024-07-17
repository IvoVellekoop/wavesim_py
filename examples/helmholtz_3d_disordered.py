import os
import torch
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess, relative_error
from __init__ import plot

if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')

""" Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result). """

wavelength = 1.
n_size = (128, 48, 96)
n = np.ascontiguousarray(loadmat('matlab_results.mat')['n3d_disordered'])
boundary_widths = 50
# add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n, boundary_widths)

# Source: single point source in the center of the domain
indices = torch.tensor([[int(v/2 - 1) + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, False, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength,
#                      n_domains=(2, 1, 1))

u_computed = run_algorithm(domain, source, max_iterations=1000)[0]
u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*3)]

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('matlab_results.mat')['u3d_disordered'])

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed.cpu().numpy(), u_ref, re)

threshold = 1.e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"
