import torch
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(".")
from wavesim_iteration import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess, relative_error
from __init__ import plot

""" Test for 1D propagation through glass plate. Compare with reference solution (matlab repo result) """
wavelength = 1.
n_size = (256, 1, 1)
n = np.ones(n_size, dtype=np.complex64)
n[99:130] = 1.5
boundary_widths = 50
# add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n, boundary_widths)

indices = torch.tensor([[0 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of the domain
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, True, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=1., n_domains=(2, 1, 1))

u_computed = run_algorithm(domain, source, max_iterations=2000)[0]
u_computed = u_computed.squeeze()[boundary_widths:-boundary_widths]
# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('matlab_results.mat')['u'])

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
threshold = 1.e-3
assert re < threshold, f"Relative error higher than {threshold}"

plot(u_computed.cpu().numpy(), u_ref, re)
