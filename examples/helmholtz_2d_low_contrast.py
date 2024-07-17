import os
import torch
import numpy as np
from scipy.io import loadmat
from PIL.Image import BILINEAR, fromarray, open
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import pad_boundaries, preprocess, relative_error
from __init__ import plot

if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')

""" Test for propagation in 2D structure with low refractive index contrast (made of fat and water to mimic biological 
    tissue). Compare with reference solution (matlab repo result). """

oversampling = 0.25
im = np.asarray(open('logo_structure_vector.png')) / 255
n_water = 1.33
n_fat = 1.46
n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water
n_roi = int(oversampling * n_im.shape[0])
n = np.asarray(fromarray(n_im).resize((n_roi, n_roi), BILINEAR))
boundary_widths = 50
# add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n, boundary_widths)

source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
source = pad_boundaries(source, boundary_array)
source = torch.tensor(source, dtype=torch.complex64)

wavelength = 0.532
pixel_size = wavelength / (3 * abs(n_fat))

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, False, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength,
#                      n_domains=(2, 2, 1))

u_computed = run_algorithm(domain, source, max_iterations=10000)[0]
u_computed = u_computed.squeeze()[*([slice(boundary_widths, -boundary_widths)]*2)]
# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('matlab_results.mat')['u2d_lc'])

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed.cpu().numpy(), u_ref, re)

threshold = 1.e-3
assert re < threshold, f"Relative error higher than {threshold}"
