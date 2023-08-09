import pytest
import numpy as np
from helmholtz_base import Helmholtz_Base


@pytest.fixture
def setup_inputs():
    n = np.ones((256, 256, 1))
    anysim = Helmholtz_Base(n=n)
    yield anysim


def test_n(setup_inputs):
    assert np.array_equal(setup_inputs.n_roi, np.array([256, 256, 1]))


def test_n_domains(setup_inputs):
    assert np.array_equal(setup_inputs.n_domains, np.ones(3))


# Should work irrespective of input type of parameters (e.g. here, boundary_widths)
@pytest.mark.parametrize('boundary_widths', [10, (10,), [10], np.array([10]), (10, 10), [10, 10], np.array([10, 10])])
def test_input_boundary_widths(boundary_widths):
    n = np.ones((256, 256, 1))
    anysim_bw = Helmholtz_Base(n=n, boundary_widths=boundary_widths)
    assert np.array_equal(anysim_bw.boundary_widths, np.array([10, 10, 0]))
