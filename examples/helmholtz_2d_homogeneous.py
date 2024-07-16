import torch
import numpy as np
import sys
import time
sys.path.append(".")
from wavesim_iteration import run_algorithm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from utilities import preprocess

""" Test for propagation in 2D homogeneous medium """

pixel_size = 0.25
wavelength = 1.
n_size = tuple((wavelength/pixel_size * np.array([50, 50, 1])).astype(int))
n = np.ones(n_size, dtype=np.complex64)
source = np.zeros_like(n)
source[n_size[0]//2, n_size[1]//2] = 1
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

start = time.time()
u_computed = run_algorithm(domain, source, max_iterations=2000)[0]
u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*2)]
print(f'Elapsed time: {time.time() - start:.2f} s')

# plot_one(u_computed.cpu().numpy())
