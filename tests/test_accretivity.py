import pytest
from numpy import ones
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain
from utilities import full_matrix
from torch import min, real, round
from torch.linalg import eigvals


@pytest.fixture
def accretivity(n, boundary_widths, n_domains, wrap_correction):
    """ Check that operator A = L + V is accretive, 
        i.e., has a non-negative real part """
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         n_domains=n_domains, wrap_correction=wrap_correction)
    restrict, extend = domain_decomp_operators(base)

    # function that evaluates one forward iteration and gives operator A
    # both input and output are arrays, so it can be evaluated by full_matrix() as an operator
    def forward(x):
        u_dict = defaultdict(list)
        for patch in base.domains_iterator:
            u_dict[patch] = map_domain(x, restrict, patch)
        l_dict = base.l_plus1(u_dict)
        b_dict = base.medium(u_dict)
        a_dict = defaultdict(list)
        for patch in base.domains_iterator:
            a_dict[patch] = l_dict[patch] - b_dict[patch]
        a = 0.
        for patch in base.domains_iterator:
            a += map_domain(a_dict[patch], extend, patch)
        return a
    
    n_ext = base.n_roi + base.boundary_pre + base.boundary_post
    a = full_matrix(forward, n_ext)
    acc = min(real(eigvals(a + a.conj().t())))
    print(f'acc {acc:.2e}')
    return acc


param_n_boundaries = [(ones(256), 0), (ones(256), 10),
                      (ones((30, 32)), 0), (ones((30, 32)), 10),
                      (ones((5, 6, 7)), 0), (ones((5, 6, 7)), 1)]


@pytest.mark.parametrize("n, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [1])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_1domain_wrap_options(accretivity):
    """ Check that operator A is accretive for 1-domain scenario for all wrapping correction options """
    # round(., 12) with numpy works. 3 with torch??
    assert round(accretivity, decimals=3) >= 0, f'a is not accretive. {accretivity}'


@pytest.mark.parametrize("n, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [2])
@pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
def test_ndomains(accretivity):
    """ Check that operator A is accretive when number of domains > 1 
    (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
    # round(., 12) with numpy works. 3 with torch??
    assert round(accretivity, decimals=3) >= 0, f'a is not accretive. {accretivity}'
