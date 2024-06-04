import numpy as np
from scipy.special import exp1

import sys
sys.path.append(".")
sys.path.append("..")

from anysim import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess, relative_error
from __init__ import plot

def u_ref_1d_h(n_size0, pixel_size, wavelength=None):
    """ Compute analytic solution for 1D case """
    x = np.arange(0, n_size0 * pixel_size, pixel_size, dtype=np.float32)
    x = np.pad(x, (n_size0, n_size0), mode='constant', constant_values=np.nan)
    h = pixel_size
    if wavelength is None:
        k = 1. * 2. * np.pi * pixel_size  # wavenumber
    else:
        k = 1. * 2. * np.pi / wavelength # wavenumber
    phi = k * x
    u_theory = (1.0j * h / (2 * k) * np.exp(1.0j * phi)  # propagating plane wave
                - h / (4 * np.pi * k) * (
                        np.exp(1.0j * phi) * (exp1(1.0j * (k - np.pi / h) * x) - exp1(1.0j * (k + np.pi / h) * x)) -
                        np.exp(-1.0j * phi) * (-exp1(-1.0j * (k - np.pi / h) * x) + exp1(-1.0j * (k + np.pi / h) * x)))
                )
    small = np.abs(k * x) < 1.e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory[n_size0:-n_size0]


""" 1D free-space propagation. Compare with analytic solution """
n_size = (1000, 1, 1)
n = np.ones(n_size, dtype=np.complex64)
source = np.zeros_like(n)
source[0] = 1.
boundary_widths = 50
n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source

wavelength = None
n_domains = (1, 1, 1)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
# periodic = (False, True, True)  # wrapping correction (here and beyond)
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=1., n_domains=n_domains)
u_computed = run_algorithm(domain, source, max_iterations=10000)
u_computed = u_computed.squeeze()[boundary_widths:-boundary_widths]
u_ref = u_ref_1d_h(n_size[0], domain.pixel_size, wavelength)

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed.cpu().numpy(), u_ref, re)