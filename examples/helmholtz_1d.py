import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(".")
from anysim import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess, relative_error
from __init__ import plot

""" Test for 1D propagation through glass plate. Compare with reference solution (matlab repo result) """
n = np.ones((256, 1, 1), dtype=np.complex64)
n[99:130] = 1.5
source = np.zeros_like(n)
source[0] = 1.
boundary_widths = 50
n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source

wavelength = 1.

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)

# # to test domain decomposition
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
