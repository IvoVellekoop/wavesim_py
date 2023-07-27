import pytest 
import numpy as np
from anysim_combined import AnySim

@pytest.fixture
def setup_inputs():#N_roi):
    N_roi = 256
    n = np.ones((N_roi,N_roi))
    anysim = AnySim(test='Test_1DFreeSpace', N_roi=N_roi, n=n)
    yield anysim

def test_N(setup_inputs):
    assert np.array_equal( setup_inputs.N_roi, 256*np.ones(setup_inputs.N_dim))

def test_boundary_widths(setup_inputs):
    assert np.array_equal( setup_inputs.boundary_widths, 20.*np.ones(setup_inputs.N_dim))

def test_N_domains(setup_inputs):
    assert np.array_equal( setup_inputs.N_domains, np.ones(setup_inputs.N_dim))

def test_overlap(setup_inputs):
    assert np.array_equal( setup_inputs.overlap, 20*np.ones(setup_inputs.N_dim))

## Should work irrespective of input type of parameters (e.g. here, boundary_widths)
@pytest.mark.parametrize('boundary_widths', [10, (10,10), [10,10], np.array([10,10]) ])
def test_input_boundary_widths(boundary_widths):
    N_roi = np.array([256, 256])
    n = np.ones(tuple(N_roi))
    anysim = AnySim(test='Test_2DLowContrast', N_roi=N_roi, n=n, boundary_widths=boundary_widths)
    assert np.array_equal( anysim.boundary_widths, 10*np.ones(anysim.N_dim))
