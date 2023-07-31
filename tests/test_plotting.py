import pytest

import numpy as np
from anysim_main import AnySim
from save_details import LogPlot
from scipy.io import loadmat
from PIL.Image import open, BILINEAR, fromarray ## needed for 2D tests

## Relative error
def relative_error(E_, E_true):
    return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )


@pytest.fixture
def setup_1DFreeSpace():
    n = np.ones((256,1,1))
    anysim1D_FS_setup = AnySim(n=n, N_domains=1, boundary_widths=20)
    print('anysim1D_FS_setup.N_roi', anysim1D_FS_setup.N_roi)

    ## Compare with the analytic solution
    x = np.arange(0,anysim1D_FS_setup.N_roi[0]*anysim1D_FS_setup.pixel_size,anysim1D_FS_setup.pixel_size)
    x = np.pad(x, (64,64), mode='constant')
    h = anysim1D_FS_setup.pixel_size
    k = anysim1D_FS_setup.k0
    phi = k * x

    E_theory = 1.0j*h/(2*k) * np.exp(1.0j*phi) - h/(4*np.pi*k) * (np.exp(1.0j * phi) * ( np.exp(1.0j * (k-np.pi/h) * x) - np.exp(1.0j * (k+np.pi/h) * x)) - np.exp(-1.0j * phi) * ( -np.exp(-1.0j * (k-np.pi/h) * x) + np.exp(-1.0j * (k+np.pi/h) * x)))
    # special case for values close to 0
    small = np.abs(k*x) < 1.e-10
    E_theory[small] = 1.0j * h/(2*k) * (1 + 2j * np.arctanh(h*k/np.pi)/np.pi); # exact value at 0.
    yield E_theory[64:-64]

@pytest.mark.parametrize("N_domains", [(1,1,1), (2,1,1), (5,1,1), (10,1,1)])
def test_1DFreeSpace(setup_1DFreeSpace, N_domains):
    n = np.ones((256,1,1))
    source = np.zeros_like(n, dtype='complex_')
    source[0] = 1.
    anysim1D_FS = AnySim(n=n, N_domains=N_domains, boundary_widths=20, source=source, overlap=20)
    anysim1D_FS.setup_operators_n_init_variables()
    anysim1D_FS.iterate()
    rel_err = anysim1D_FS.compare(setup_1DFreeSpace)
    LogPlot()

    assert rel_err <= 1.e-3

@pytest.mark.parametrize("N_domains", [(1,1,1), (2,1,1)])
def test_1DGlassPlate(N_domains):
    n = np.ones((256,1,1))
    n[99:130] = 1.5
    source = np.zeros_like(n, dtype='complex_')
    source[0] = 1.
    anysim1D_GP = AnySim(n=n, N_domains=N_domains, boundary_widths=20, source=source, overlap=20)
    anysim1D_GP.setup_operators_n_init_variables()
    anysim1D_GP.iterate()
    lp = LogPlot()
    assert lp.plotting_done


@pytest.mark.parametrize("N_domains", [1])
def test_2DHighContrast(N_domains):
    oversampling = 0.25
    im = np.asarray(open('anysim_matlab/logo_structure_vector.png'))/255
    n_iron = 2.8954 + 2.9179j
    n_contrast = n_iron - 1
    n_im = ((np.where(im[:,:,2]>(0.25),1,0) * n_contrast)+1)
    N_roi = int(oversampling*n_im.shape[0])
    # n2d = np.asarray(fromarray(n_im).resize((N_roi,N_roi), BILINEAR)) # resize cannot work with complex values?
    n = loadmat('anysim_matlab/n2d.mat')['n']

    source = np.asarray(fromarray(im[:,:,1]).resize((N_roi,N_roi), BILINEAR))
    boundary_widths = (31.5, 31.5)
    max_iters = int(1.e+4)  # 1.e+4 iterations gives relative error 1.65e-4 with the matlab test result, but takes ~140s
    wavelength = 0.532
    ppw = 3*np.max(abs(n_contrast+1))

    anysim2D_HC = AnySim(wavelength=wavelength, ppw=ppw, boundary_widths=boundary_widths, n=n, source=source, N_domains=N_domains, overlap=boundary_widths, max_iters=max_iters)
    anysim2D_HC.setup_operators_n_init_variables()
    anysim2D_HC.iterate()
    lp = LogPlot()
    assert lp.plotting_done


@pytest.mark.parametrize("N_domains", [1])
def test_2DLowContrast(N_domains):
    oversampling = 0.25
    im = np.asarray(open('anysim_matlab/logo_structure_vector.png'))/255
    n_water = 1.33
    n_fat = 1.46

    n_im = (np.where(im[:,:,2]>(0.25),1,0) * (n_fat-n_water)) + n_water
    N_roi = int(oversampling*n_im.shape[0])
    n = np.asarray(fromarray(n_im).resize((N_roi,N_roi), BILINEAR))

    source = np.asarray(fromarray(im[:,:,1]).resize((N_roi,N_roi), BILINEAR))
    boundary_widths = (75,75,0)
    wavelength = 0.532
    ppw = 3*abs(n_fat)

    anysim2D_LC = AnySim(wavelength=wavelength, ppw=ppw, boundary_widths=boundary_widths, n=n, source=source, N_domains=N_domains, overlap=(20,20,0))
    anysim2D_LC.setup_operators_n_init_variables()
    anysim2D_LC.iterate()
    lp = LogPlot()
    assert lp.plotting_done


@pytest.mark.parametrize("N_roi", [np.array([128, 128, 128]), np.array([128, 48, 96])])
@pytest.mark.parametrize("boundary_widths", [np.array([24, 24, 24]), np.array([20, 24, 32])])
def test_3DHomogeneous(N_roi, boundary_widths):
    n_sample = np.ones(tuple(N_roi))
    source = np.zeros_like(n_sample, dtype='complex_')
    source[int(N_roi[0]/2-1),int(N_roi[1]/2-1),int(N_roi[2]/2-1)] = 1.

    anysim3D_H = AnySim(boundary_widths=boundary_widths, n=n_sample, source=source, N_domains=np.array([1,1,1]), overlap=boundary_widths)

    anysim3D_H.setup_operators_n_init_variables()
    anysim3D_H.iterate()
    lp = LogPlot()
    assert lp.plotting_done


def test_3DDisordered():
    boundary_widths = np.array([20., 20., 20.])
    N_roi = np.array([128, 128, 128])
    n_sample = loadmat(f'anysim_matlab/n3d_disordered.mat')['n_sample']
    source = np.zeros_like(n_sample, dtype='complex_')
    source[int(N_roi[0]/2-1),int(N_roi[1]/2-1),int(N_roi[2]/2-1)] = 1.

    anysim3D = AnySim(boundary_widths=boundary_widths, n=n_sample, source=source, N_domains=np.array([1,1,1]), overlap=boundary_widths)

    anysim3D.setup_operators_n_init_variables()
    anysim3D.iterate()
    lp = LogPlot()
    assert lp.plotting_done
