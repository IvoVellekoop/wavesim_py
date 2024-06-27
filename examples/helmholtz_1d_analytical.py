import torch
import numpy as np
from scipy.special import exp1
import sys
sys.path.append(".")
from wavesim_iteration import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess, relative_error
from __init__ import plot


def u_ref_1d_h(n_size0, pixel_size, wavelength=None):
    """ Compute analytic solution for 1D case """
    x = np.arange(0, n_size0 * pixel_size, pixel_size, dtype=np.float32)
    x = np.pad(x, (n_size0, n_size0), mode='constant', constant_values=np.nan)
    h = pixel_size
    # wavenumber (k)
    if wavelength is None:
        k = 1. * 2. * np.pi * pixel_size
    else:
        k = 1. * 2. * np.pi / wavelength
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
wavelength = 1.
n_size = (256, 1, 1)
n = np.ones(n_size, dtype=np.complex64)
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
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=(3, 1, 1))

u_computed = run_algorithm(domain, source, max_iterations=2000)[0]
u_computed = u_computed.squeeze()[boundary_widths:-boundary_widths]
u_ref = u_ref_1d_h(n_size[0], domain.pixel_size, wavelength)

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
threshold = 1.e-3
assert re < threshold, f"Relative error higher than {threshold}"

plot(u_computed.cpu().numpy(), u_ref, re)
