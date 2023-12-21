import pytest
import numpy as np
from helmholtzbase import HelmholtzBase


@pytest.fixture
def setup_inputs():
    """ Setup base problem with dummy input """
    yield HelmholtzBase(n=np.ones((256, 256, 1)), setup_operators=False)


def test_n(setup_inputs):
    """ Check that n_roi from base is an 3-element numpy array. 
        Elements beyond the dimensions of the problem should be either 1 or 0"""
    assert np.array_equal(setup_inputs.n_roi, np.array([256, 256, 1]))


def test_n_domains(setup_inputs):
    """ Check that n_domains from base is an 3-element numpy array of ones """
    assert np.array_equal(setup_inputs.n_domains, np.ones(3))


@pytest.mark.parametrize('boundary_widths', [10, (10,), [10], np.array([10]), (10, 10), [10, 10], np.array([10, 10])])
def test_input_boundary_widths(boundary_widths):
    """ Input parameters (here, boundary_widths) should be 3-element numpy arrays irrespective of input type and shape.
        Elements beyond the dimensions of the problem should be either 1 or 0"""
    anysim_bw = HelmholtzBase(n=np.ones((256, 256, 1)), boundary_widths=boundary_widths, setup_operators=False)
    assert np.array_equal(anysim_bw.boundary_widths, np.array([10, 10, 0]))
