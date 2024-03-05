import os
if os.path.basename(os.getcwd()) == 'tests':
    os.chdir('..')

import pytest
import numpy as np
from scipy.io import loadmat
from PIL.Image import open, BILINEAR, fromarray  # needed for 2D tests

from helmholtzbase import HelmholtzBase
from anysim import run_algorithm
from save_details import LogPlot
from utilities import max_abs_error, relative_error

# load dictionary of results from matlab wavesim/anysim for comparison and validation
matlab_results = loadmat('matlab_results.mat')


def compare(base: HelmholtzBase, u_computed, u_reference, threshold=1.e-3):
    """ Compute, Print, and Assert relative and maximum absolute errors between computed and reference field """
    if u_reference.shape[0] != base.n_roi[0]:
        u_computed = u_computed[tuple([slice(0, base.n_roi[i]) for i in range(base.n_dims)])]
    rel_err = relative_error(u_computed, u_reference)
    mae = max_abs_error(u_computed, u_reference)
    print(f'Relative error ({rel_err:.2e})')
    print(f'Max absolute error (Normalized) ({mae:.2e})')
    assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
    assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'


def u_ref_1d_h(n):
    """ Compute analytic solution for 1D case """
    base_ = HelmholtzBase(n=n, setup_operators=False)

    x = np.arange(0, base_.n_roi[0] * base_.pixel_size, base_.pixel_size, dtype=np.complex64)
    x = np.pad(x, (64, 64), mode='constant')
    h = base_.pixel_size
    k = (1. * 2. * np.pi) / 1.
    phi = k * x
    u_theory = 1.0j * h / (2 * k) * np.exp(1.0j * phi) - h / (4 * np.pi * k) * (
               np.exp(1.0j * phi) * (np.exp(1.0j * (k - np.pi / h) * x) - np.exp(1.0j * (k + np.pi / h) * x)) - np.exp(
                -1.0j * phi) * (-np.exp(-1.0j * (k - np.pi / h) * x) + np.exp(-1.0j * (k + np.pi / h) * x)))
    small = np.abs(k * x) < 1.e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory[64:-64]


@pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'), (1, 'L_omega'), 
                                                        (2, 'wrap_corr'), (3, 'wrap_corr'), (4, 'wrap_corr')])
def test_1d_homogeneous(n_domains, wrap_correction):
    """ Test for 1D free-space propagation. Compare with analytic solution """
    n_size = (256, 1, 1)
    n = np.ones(n_size, dtype=np.complex64)
    u_ref = u_ref_1d_h(n)
    source = np.zeros_like(n)
    source[0] = 1.
    base = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction=wrap_correction)
    u_computed, state = run_algorithm(base)
    LogPlot(base, state, u_computed, u_ref).log_and_plot()
    compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)


@pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'), (1, 'L_omega'),
                                                        (2, 'wrap_corr'), (3, 'wrap_corr'), (4, 'wrap_corr')])
def test_1d_glass_plate(n_domains, wrap_correction):
    """ Test for 1D propagation through glass plate. Compare with reference solution (matlab repo result) """
    n = np.ones((256, 1, 1), dtype=np.complex64)
    n[99:130] = 1.5
    source = np.zeros_like(n)
    source[0] = 1.
    base = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction=wrap_correction)
    u_computed, state = run_algorithm(base)
    u_ref = np.squeeze(matlab_results['u'])
    LogPlot(base, state, u_computed, u_ref).log_and_plot()
    compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)


@pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'),
                                                        (2, 'wrap_corr'), (3, 'wrap_corr')])
def test_2d_high_contrast(n_domains, wrap_correction):
    """ Test for propagation in 2D structure made of iron, with high refractive index contrast.
        Compare with reference solution (matlab repo result) """
    oversampling = 0.25
    im = np.asarray(open('logo_structure_vector.png')) / 255
    n_iron = 2.8954 + 2.9179j
    n_contrast = n_iron - 1
    n_im = ((np.where(im[:, :, 2] > 0.25, 1, 0) * n_contrast) + 1)
    n_roi = int(oversampling * n_im.shape[0])
    # n2d = np.asarray(fromarray(n_im).resize((n_roi,n_roi), BILINEAR)) # resize cannot work with complex values?
    if os.path.basename(os.getcwd()) == 'tests':
        os.chdir('..')
    n = matlab_results['n2d_hc']
    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    base = HelmholtzBase(n=n, source=source, wavelength=0.532, ppw=3*np.max(abs(n_contrast + 1)), 
                         n_domains=n_domains, wrap_correction=wrap_correction, 
                         max_iterations=int(1.e+5))
    u_computed, state = run_algorithm(base)
    u_ref = matlab_results['u2d_hc']
    LogPlot(base, state, u_computed, u_ref).log_and_plot()
    compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)


@pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'),
                                                        (2, 'wrap_corr'), (3, 'wrap_corr')])
def test_2d_low_contrast(n_domains, wrap_correction):
    """ Test for propagation in 2D structure with low refractive index contrast. 
        Compare with reference solution (matlab repo result) """
    oversampling = 0.25
    im = np.asarray(open('logo_structure_vector.png')) / 255
    n_water = 1.33
    n_fat = 1.46
    n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water
    n_roi = int(oversampling * n_im.shape[0])
    n = np.asarray(fromarray(n_im).resize((n_roi, n_roi), BILINEAR))
    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    base = HelmholtzBase(n=n, source=source, wavelength=0.532, ppw=3*abs(n_fat), 
                         n_domains=n_domains, wrap_correction=wrap_correction)
    u_computed, state = run_algorithm(base)
    u_ref = matlab_results['u2d_lc']
    LogPlot(base, state, u_computed, u_ref).log_and_plot()
    compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)


# @pytest.mark.parametrize("n_roi", [np.array([128, 128, 128]), np.array([128, 48, 96])])
# @pytest.mark.parametrize("boundary_widths", [np.array([24, 24, 24]), np.array([20, 24, 32])])
# @pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr'])
# def test_3d_homogeneous(n_roi, boundary_widths, wrap_correction):
@pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'),
                                                        (2, 'wrap_corr'), (3, 'wrap_corr')])
def test_3d_homogeneous(n_domains, wrap_correction):
    """ Test for propagation in a 3D homogeneous medium. Compare with reference solution (matlab repo result).
        Testing with same and varying sizes and boundary widths in each dimension. """
    n_roi = np.array([128, 128, 128])
    n_sample = np.ones(tuple(n_roi), dtype=np.complex64)
    source = np.zeros_like(n_sample)
    source[int(n_roi[0] / 2 - 1), int(n_roi[1] / 2 - 1), int(n_roi[2] / 2 - 1)] = 1.

    base = HelmholtzBase(n=n_sample, source=source, n_domains=n_domains, wrap_correction=wrap_correction)
    u_computed, state = run_algorithm(base)
    u_ref = matlab_results[f'u3d_{n_roi[0]}_{n_roi[1]}_{n_roi[2]}_bw_20_24_32']
    LogPlot(base, state, u_computed, u_ref).log_and_plot()
    compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)


@pytest.mark.parametrize("n_domains, wrap_correction", [(1, None), (1, 'wrap_corr'),
                                                        (2, 'wrap_corr'), (3, 'wrap_corr')])
def test_3d_disordered(n_domains, wrap_correction):
    """ Test for propagation in a 3D disordered medium. Compare with reference solution (matlab repo result) """
    n_roi = (128, 48, 96)
    n_sample = matlab_results['n3d_disordered']
    source = np.zeros_like(n_sample, dtype=np.complex64)
    source[int(n_roi[0] / 2 - 1), int(n_roi[1] / 2 - 1), int(n_roi[2] / 2 - 1)] = 1.

    base = HelmholtzBase(n=n_sample, source=source, n_domains=n_domains, wrap_correction=wrap_correction)
    u_computed, state = run_algorithm(base)
    u_ref = matlab_results['u3d_disordered']
    LogPlot(base, state, u_computed, u_ref).log_and_plot()
    compare(base, u_computed.cpu().numpy(), u_ref, threshold=1.e-3)
