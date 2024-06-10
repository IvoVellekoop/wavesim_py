import numpy as np
from scipy.io import loadmat
import sys
sys.path.append(".")
from anysim import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess, relative_error
from __init__ import plot

""" Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result) """

n_roi = (128, 128, 128)
n = np.ones(tuple(n_roi), dtype=np.complex64)
source = np.zeros_like(n)
source[tuple(int(d / 2 - 1) for d in n.shape)] = 1.
boundary_widths = 50
n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source
wavelength = 1.

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)

# # to test domain decomposition
# periodic = (False, False, False)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=(2, 1, 1))

u_computed = run_algorithm(domain, source, max_iterations=2000)
u_computed = u_computed.squeeze()[*([slice(boundary_widths,-boundary_widths)]*3)]

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('matlab_results.mat')[f'u3d_{n_roi[0]}_{n_roi[1]}_{n_roi[2]}_bw_20_24_32'])

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed.cpu().numpy(), u_ref, re)

threshold = 1.e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"
