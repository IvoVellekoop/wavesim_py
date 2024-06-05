import numpy as np
from scipy.io import loadmat
from PIL.Image import BILINEAR, fromarray, open
import sys
sys.path.append(".")
from anysim import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess, relative_error
from __init__ import plot

""" Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result) """

n_roi = (128, 48, 96)
n = np.ascontiguousarray(loadmat('matlab_results.mat')['n3d_disordered'])
source = np.zeros_like(n, dtype=np.complex64)
source[tuple(int(d / 2 - 1) for d in n.shape)] = 1.
boundary_widths = 50
n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source

wavelength = 1.
n_domains = (1, 1, 1)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
# periodic = (False, True, True)  # wrapping correction (here and beyond)
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)

u_computed = run_algorithm(domain, source)
u_computed = u_computed.squeeze()[*([slice(boundary_widths,-boundary_widths)]*3)]

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('matlab_results.mat')['u3d_disordered'])

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed.cpu().numpy(), u_ref, re)
