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

""" Test for propagation in 2D structure made of iron, with high refractive index contrast.
    Compare with reference solution (matlab repo result) """

oversampling = 0.25
im = np.asarray(open('logo_structure_vector.png')) / 255
n_iron = 2.8954 + 2.9179j
n_contrast = n_iron - 1
n_im = ((np.where(im[:, :, 2] > 0.25, 1, 0) * n_contrast) + 1)
n_roi = int(oversampling * n_im.shape[0])
n = np.asarray(fromarray(n_im.real).resize((n_roi, n_roi), BILINEAR)) + 1j * np.asarray(
    fromarray(n_im.imag).resize((n_roi, n_roi), BILINEAR))
# # load dictionary of results from matlab wavesim/anysim for comparison and validation
# n = loadmat('matlab_results.mat')['n2d_hc']
source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
boundary_widths = 50
n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source

wavelength = 0.532
pixel_size = wavelength / (3 * np.max(abs(n_contrast + 1)))

# # 1-domain, periodic boundaries (without wrapping correction)
# periodic = (True, True, True)  # periodic boundaries, wrapped field.
# domain = HelmholtzDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength)

# to test domain decomposition
periodic = (False, False, True)  # wrapping correction
domain = MultiDomain(permittivity=n, periodic=periodic, pixel_size=pixel_size, wavelength=wavelength, 
                     n_domains=(2, 2, 1))
print(domain.scale)

u_computed = run_algorithm(domain, source, max_iterations=int(1.e+5))[0]
u_computed = u_computed.squeeze()[*([slice(boundary_widths,-boundary_widths)]*2)]

# load dictionary of results from matlab wavesim/anysim for comparison and validation
u_ref = np.squeeze(loadmat('matlab_results.mat')['u2d_hc'])

re = relative_error(u_computed.cpu().numpy(), u_ref)
print(f'Relative error: {re:.2e}')
plot(u_computed.cpu().numpy(), u_ref, re)

threshold = 1.e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"