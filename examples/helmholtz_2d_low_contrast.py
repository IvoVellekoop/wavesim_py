""" 
Helmholtz 2D low contrast test
===============================
Test for propagation in 2D structure with low refractive index contrast 
(made of fat and water to mimic biological tissue). 
Compare with reference solution (matlab repo result). 
"""

import os
import torch
import numpy as np
from time import time
from scipy.io import loadmat
from PIL.Image import BILINEAR, fromarray, open
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the wavesim iteration
from wavesim.utilities import pad_boundaries, preprocess, relative_error
from __init__ import plot

if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')


# Parameters
n_water = 1.33
n_fat = 1.46
wavelength = 0.532  # Wavelength in micrometers
pixel_size = wavelength / (3 * abs(n_fat))  # Pixel size in wavelength units

# Load image and create refractive index map
oversampling = 0.25
im = np.asarray(open('examples/logo_structure_vector.png')) / 255
n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water
n_roi = int(oversampling * n_im.shape[0])  # Size of ROI in pixels
n = np.asarray(fromarray(n_im).resize((n_roi, n_roi), BILINEAR))  # Refractive index map
boundary_widths = 40  # Width of the boundary in pixels

# return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n**2, boundary_widths)  # permittivity is n², but uses the same variable n

# Source term
source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
source = pad_boundaries(source, boundary_array)
source = torch.tensor(source, dtype=torch.complex64)

# Set up the domain operators (HelmholtzDomain() or MultiDomain() depending on number of domains)
# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, True, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength,
#                      n_domains=(3, 1, 1))

# Run the wavesim iteration and get the computed field
start = time()
u_computed, iterations, residual_norm = run_algorithm(domain, source, max_iterations=10000)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_computed = u_computed.squeeze().cpu().numpy()[*([slice(boundary_widths, -boundary_widths)]*2)]

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('examples/matlab_results.mat')['u2d_lc'])

# Compute relative error with respect to the reference solution
re = relative_error(u_computed, u_ref)
print(f'Relative error: {re:.2e}')
threshold = 1.e-3
assert re < threshold, f"Relative error higher than {threshold}"

# Plot the results
plot(u_computed, u_ref, re)
