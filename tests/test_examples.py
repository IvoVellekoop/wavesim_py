import pytest
import numpy as np
from scipy.io import loadmat
from PIL.Image import open, BILINEAR, fromarray  # needed for 2D tests

from anysim_main import AnySim
from save_details import print_details, compare, LogPlot


@pytest.fixture
def setup_1d_homogeneous():
    n = np.ones((256, 1, 1))
    anysim_1d_h_setup = AnySim(n=n, n_domains=1, boundary_widths=20)
    print('anysim_1d_h_setup.n_roi', anysim_1d_h_setup.n_roi)

    # Compare with the analytic solution
    x = np.arange(0, anysim_1d_h_setup.n_roi[0] * anysim_1d_h_setup.pixel_size, anysim_1d_h_setup.pixel_size)
    x = np.pad(x, (64, 64), mode='constant')
    h = anysim_1d_h_setup.pixel_size
    k = anysim_1d_h_setup.k0
    phi = k * x

    e_theory = 1.0j * h / (2 * k) * np.exp(1.0j * phi) - h / (4 * np.pi * k) * (
            np.exp(1.0j * phi) * (np.exp(1.0j * (k - np.pi / h) * x) - np.exp(1.0j * (k + np.pi / h) * x)) - np.exp(
        -1.0j * phi) * (-np.exp(-1.0j * (k - np.pi / h) * x) + np.exp(-1.0j * (k + np.pi / h) * x)))
    # special case for values close to 0
    small = np.abs(k * x) < 1.e-10
    e_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    yield e_theory[64:-64]


@pytest.mark.parametrize("n_domains", [(i, 1, 1) for i in range(1, 11)])
def test_1d_homogeneous(setup_1d_homogeneous, n_domains):
    n = np.ones((256, 1, 1))
    # n = np.random.rand(256, 1, 1)
    source = np.zeros_like(n, dtype='complex_')
    source[0] = 1.
    anysim_1d_h = AnySim(n=n, n_domains=n_domains, boundary_widths=20, source=source, overlap=20)
    print_details(anysim_1d_h)
    anysim_1d_h.setup_operators_n_initialize()
    anysim_1d_h.iterate()
    rel_err_1d_h = compare(anysim_1d_h, setup_1d_homogeneous)
    lp_1d_h = LogPlot(anysim_1d_h, setup_1d_homogeneous, rel_err_1d_h)
    lp_1d_h.log_and_plot()

    assert rel_err_1d_h <= 1.e-3


@pytest.mark.parametrize("n_domains", [(i, 1, 1) for i in range(1, 11)])
def test_1d_glass_plate(n_domains):
    n = np.ones((256, 1, 1))
    n[99:130] = 1.5
    source = np.zeros_like(n, dtype='complex_')
    source[0] = 1.
    anysim_1d_gp = AnySim(n=n, n_domains=n_domains, boundary_widths=20, source=source, overlap=20)
    print_details(anysim_1d_gp)
    anysim_1d_gp.setup_operators_n_initialize()
    anysim_1d_gp.iterate()

    u_true_1d_gp = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])
    rel_err_1d_gp = compare(anysim_1d_gp, u_true_1d_gp)
    lp_1d_gp = LogPlot(anysim_1d_gp, u_true_1d_gp, rel_err_1d_gp)
    lp_1d_gp.log_and_plot()

    assert rel_err_1d_gp <= 1.e-3


@pytest.mark.parametrize("n_domains", [1])
def test_2d_high_contrast(n_domains):
    oversampling = 0.25
    im = np.asarray(open('anysim_matlab/logo_structure_vector.png')) / 255
    n_iron = 2.8954 + 2.9179j
    n_contrast = n_iron - 1
    n_im = ((np.where(im[:, :, 2] > (0.25), 1, 0) * n_contrast) + 1)
    n_roi = int(oversampling * n_im.shape[0])
    # n2d = np.asarray(fromarray(n_im).resize((n_roi,n_roi), BILINEAR)) # resize cannot work with complex values?
    n = loadmat('anysim_matlab/n2d.mat')['n']

    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    boundary_widths = (31.5, 31.5)
    max_iters = int(1.e+4)  # 1.e+4 iterations gives rel_error 1.65e-4 with matlab result, but takes really long
    wavelength = 0.532
    ppw = 3 * np.max(abs(n_contrast + 1))

    anysim_2d_hc = AnySim(wavelength=wavelength, ppw=ppw, boundary_widths=boundary_widths, n=n, source=source,
                          n_domains=n_domains, overlap=boundary_widths, max_iterations=max_iters)
    print_details(anysim_2d_hc)
    anysim_2d_hc.setup_operators_n_initialize()
    anysim_2d_hc.iterate()

    u_true_2d_hc = loadmat('anysim_matlab/u2d.mat')['u2d']
    rel_err_2d_hc = compare(anysim_2d_hc, u_true_2d_hc)
    lp_2d_hc = LogPlot(anysim_2d_hc, u_true_2d_hc, rel_err_2d_hc)
    lp_2d_hc.log_and_plot()

    assert rel_err_2d_hc <= 1.e-3


@pytest.mark.parametrize("n_domains", [1])
def test_2d_low_contrast(n_domains):
    oversampling = 0.25
    im = np.asarray(open('anysim_matlab/logo_structure_vector.png')) / 255
    n_water = 1.33
    n_fat = 1.46

    n_im = (np.where(im[:, :, 2] > (0.25), 1, 0) * (n_fat - n_water)) + n_water
    n_roi = int(oversampling * n_im.shape[0])
    n = np.asarray(fromarray(n_im).resize((n_roi, n_roi), BILINEAR))

    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    boundary_widths = (75, 75, 0)
    wavelength = 0.532
    ppw = 3 * abs(n_fat)

    anysim_2d_lc = AnySim(wavelength=wavelength, ppw=ppw, boundary_widths=boundary_widths, n=n, source=source,
                          n_domains=n_domains, overlap=(20, 20, 0))
    print_details(anysim_2d_lc)
    anysim_2d_lc.setup_operators_n_initialize()
    anysim_2d_lc.iterate()

    u_true_2d_lc = loadmat('anysim_matlab/u2d_lc.mat')['u2d']
    rel_err_2d_lc = compare(anysim_2d_lc, u_true_2d_lc)
    lp_2d_lc = LogPlot(anysim_2d_lc, u_true_2d_lc, rel_err_2d_lc)
    lp_2d_lc.log_and_plot()

    assert rel_err_2d_lc <= 1.e-3


@pytest.mark.parametrize("n_roi", [np.array([128, 128, 128]), np.array([128, 48, 96])])
@pytest.mark.parametrize("boundary_widths", [np.array([24, 24, 24]), np.array([20, 24, 32])])
def test_3d_homogeneous(n_roi, boundary_widths):
    n_sample = np.ones(tuple(n_roi))
    source = np.zeros_like(n_sample, dtype='complex_')
    source[int(n_roi[0] / 2 - 1), int(n_roi[1] / 2 - 1), int(n_roi[2] / 2 - 1)] = 1.

    anysim_3d_h = AnySim(boundary_widths=boundary_widths, n=n_sample, source=source, n_domains=np.array([1, 1, 1]),
                         overlap=boundary_widths)
    print_details(anysim_3d_h)
    anysim_3d_h.setup_operators_n_initialize()
    anysim_3d_h.iterate()

    u_true_3d_h = loadmat(
        f'anysim_matlab/u3d_{n_roi[0]}_{n_roi[1]}_{n_roi[2]}_bw_{boundary_widths[0]}_{boundary_widths[1]}_{boundary_widths[2]}.mat')[
        'u']
    rel_err_3d_h = compare(anysim_3d_h, u_true_3d_h)
    lp_3d_h = LogPlot(anysim_3d_h, u_true_3d_h, rel_err_3d_h)
    lp_3d_h.log_and_plot()

    assert rel_err_3d_h <= 1.e-3


def test_3d_disordered():
    boundary_widths = np.array([20., 20., 20.])
    n_roi = np.array([128, 128, 128])
    n_sample = loadmat(f'anysim_matlab/n3d_disordered.mat')['n_sample']
    source = np.zeros_like(n_sample, dtype='complex_')
    source[int(n_roi[0] / 2 - 1), int(n_roi[1] / 2 - 1), int(n_roi[2] / 2 - 1)] = 1.

    anysim_3d_d = AnySim(boundary_widths=boundary_widths, n=n_sample, source=source, 
                         n_domains=np.array([1, 1, 1]), overlap=boundary_widths)
    print_details(anysim_3d_d)
    anysim_3d_d.setup_operators_n_initialize()
    anysim_3d_d.iterate()

    u_true_3d_d = loadmat(f'anysim_matlab/u3d_disordered.mat')['u']
    rel_err_3d_d = compare(anysim_3d_d, u_true_3d_d)
    lp_3d_d = LogPlot(anysim_3d_d, u_true_3d_d, rel_err_3d_d)
    lp_3d_d.log_and_plot()

    assert rel_err_3d_d <= 1.e-3
