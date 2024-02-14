import pytest
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain, precon_iteration
from utilities import full_matrix
import torch
torch.set_default_dtype(torch.float32)


@pytest.fixture
def operator_checks(n, boundary_widths, n_domains, wrap_correction):
    """ Check that preconditioned operator Op = 1 - alpha* B[1 - (L+1)^(-1)B] is a contraction,
        i.e., the operator norm || Op || < 1 
        and spectral radius, i.e. max(abs(eigvals(Op))) < 1 """
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
    norm_ = np.linalg.norm(mat_.cpu().numpy(), 2)
    spec_radius = np.max(np.abs(np.linalg.eigvals(mat_.cpu().numpy())))
    return norm_, spec_radius


param_n_boundaries = [(np.ones(256), 0), (np.ones(256), 10),
                      (np.ones((30, 32)), 0), (np.ones((30, 32)), 10),
                      (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)]


@pytest.mark.parametrize("n, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [1])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_1domain_wrap_options(operator_checks):
    """ Check that spectral radius < 1 for 1-domain scenario for all wrapping correction options """
    norm_, spec_radius = operator_checks
    assert norm_ < 1, f'||Op|| not < 1, but {norm_}'
    assert spec_radius < 1, f'spectral radius not < 1, but {spec_radius}'


@pytest.mark.parametrize("n, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [2])
@pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
def test_ndomains(operator_checks):
    """ Check that spectral radius < 1 when number of domains > 1 
    (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
    norm_, spec_radius = operator_checks
    assert norm_ < 1, f'||Op|| not < 1, but {norm_}'
    assert spec_radius < 1, f'spectral radius not < 1, but {spec_radius}'
