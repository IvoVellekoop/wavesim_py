import pytest
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain, precon_iteration
from utilities import full_matrix
import torch
torch.set_default_dtype(torch.float32)


@pytest.fixture
def spec_radius(n, boundary_widths, n_domains, wrap_correction):
    """ Check that preconditioned operator is a contraction,
        i.e., the operator norm || 1 - B[1 - (L+1)^(-1)B] || < 1 """
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         n_domains=n_domains, wrap_correction=wrap_correction)
    restrict, extend = domain_decomp_operators(base)
    
    # function that evaluates one preconditioned iteration
    # both input and output are arrays, so it can be evaluated by full_matrix() as an operator
    def op_(x):
        u_dict = defaultdict(list)
        ut_dict = u_dict.copy()
        for patch in base.domains_iterator:
            u_dict[patch] = map_domain(x, restrict, patch)
        t_dict = precon_iteration(base, u_dict, ut_dict)
        t = 0.
        for patch in base.domains_iterator:
            t += map_domain(t_dict[patch], extend, patch)
        return t

    n_ext = base.n_roi + base.boundary_pre + base.boundary_post
    mat_ = (torch.diag(torch.ones(np.prod(n_ext), dtype=torch.complex64, device=base.device))
            - base.alpha * full_matrix(op_, n_ext))
    # norm_ = np.linalg.norm(mat_.cpu().numpy(), 2)
    # print(f'norm_ {norm_:.4f}')
    sr = np.max(np.abs(np.linalg.eigvals(mat_.cpu().numpy())))
    print(f'spec_radius {sr:.4f}')
    return sr


param_n_boundaries = [(np.ones(256), 0), (np.ones(256), 10),
                      (np.ones((30, 32)), 0), (np.ones((30, 32)), 10),
                      (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)]


@pytest.mark.parametrize("n, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [2])
@pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
def test_ndomains(spec_radius):
    """ Check that spectral radius < 1 when number of domains > 1 
    (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
    assert spec_radius < 1, f'spectral radius >= 1 ({spec_radius})'


@pytest.mark.parametrize("n, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [1])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_1domain_wrap_options(spec_radius):
    """ Check that spectral radius < 1 for 1-domain scenario for all wrapping correction options """
    assert spec_radius < 1, f'spectral radius >= 1 ({spec_radius})'
