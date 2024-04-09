import pytest
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain
from utilities import max_abs_error, relative_error, squeeze_
from torch import complex64, rand


@pytest.fixture
def check_l_plus1_inv(n_size, boundary_widths, n_domains, wrap_correction):
    """ Check that (L+1)^(-1) (L+1) x = x """
    n = np.ones(n_size, dtype=np.complex64)
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, n_domains=n_domains, wrap_correction=wrap_correction,
                         scaling=1.)
    restrict, extend = domain_decomp_operators(base)

    # function that evaluates (L+1)^(-1) (L+1) x
    def l_inv_l(x):
        u_dict = defaultdict(list)
        for patch in base.domains_iterator:
            u_dict[patch] = map_domain(x.to(base.devices[patch]), restrict, patch)
        l_dict = base.l_plus1(u_dict, crop=False)
        l_dict = base.propagator(l_dict)
        x_ = 0.
        for patch in base.domains_iterator:
            x_ += map_domain(l_dict[patch], extend, patch).cpu()
        return x_
    
    x_in = rand(*base.s.shape, dtype=complex64, device=base.devices[(0, 0, 0)])
    x_out = l_inv_l(x_in)

    if boundary_widths != 0:
        # crop to n_roi, excluding boundaries
        crop2roi = tuple([slice(base.boundary_pre[0], -base.boundary_post[0]) for _ in range(base.n_dims)])
        x_in = x_in[crop2roi]
        x_out = x_out[crop2roi]
    x_in = squeeze_(x_in.cpu().numpy())
    x_out = squeeze_(x_out.cpu().numpy())

    rel_err = relative_error(x_out, x_in)
    mae = max_abs_error(x_out, x_in)
    print(f'Relative error ({rel_err:.2e})')
    print(f'Max absolute error (Normalized) ({mae:.2e})')
    return rel_err, mae


param_n_boundaries = [(236, 0), (236, 10),
                      ((30, 32), 0), ((30, 32), 10),
                      ((30, 31, 32), 0), ((30, 31, 32), 5)]


@pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [1])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_1domain_wrap_options(check_l_plus1_inv):
    """ Check that (L+1)^(-1) (L+1) x = x for 1-domain scenario for all wrapping correction options """
    rel_err, mae = check_l_plus1_inv
    threshold = 1.e-3
    assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
    assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'


@pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [2])
@pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
def test_ndomains(check_l_plus1_inv):
    """ Check that (L+1)^(-1) (L+1) x = x when number of domains > 1 
    (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
    rel_err, mae = check_l_plus1_inv
    threshold = 1.e-3
    assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
    assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'
