import pytest
import numpy as np
from collections import defaultdict
from wavesim.multidomain import MultiDomain
from anysim import domain_decomp_operators, map_domain
from utilities import full_matrix


# directly v = 1-b
@pytest.fixture
def v_contraction(n_size, boundary_widths, n_domains, wrap_correction):
    """ Check that V is a contraction, i.e., the operator norm || V || < 1 
        and spectral radius, i.e. max(abs(eigvals(Op))) < 1 """
    # n = np.ones(n_size, dtype=np.complex64)
    n = np.random.rand(*n_size).astype(np.float32)
    # n[tuple(i//2 for i in n_size)] = 1.5
    base = MultiDomain(refractive_index=n, boundary_widths=boundary_widths,
                       n_domains=n_domains, wrap_correction=wrap_correction)
    restrict, extend = domain_decomp_operators(base)

    # function that evaluates B = 1 - V
    # both input and output are arrays, so it can be evaluated by full_matrix() as an operator
    def b_(x):
        u_dict = defaultdict(list)
        for patch_ in base.domains_iterator:
            u_dict[patch_] = map_domain(x.to(base.devices[patch_]), restrict, patch_)
        b_dict = base.medium(u_dict)
        b = 0.
        for patch_ in base.domains_iterator:
            b += map_domain(b_dict[patch_], extend, patch_).cpu()
        return b

    # compute full_matrix(B) and then V = 1 - B
    n_ext = base.n_roi + base.boundary_pre + base.boundary_post
    v = np.eye(np.prod(n_ext), dtype=np.complex64) - full_matrix(b_, n_ext)

    norm_ = np.linalg.norm(v, 2)
    spec_radius = np.max(np.abs(np.linalg.eigvals(v)))
    print(f'Norm ({norm_:.4e})')
    print(f'Spectral radius ({spec_radius:.4e})')
    return norm_, spec_radius


def check_assertions(norm_, spec_radius):
    errors = []
    # replace assertions by conditions
    if not norm_ < 1:
        errors.append(f'||op|| not < 1, but {norm_}')
    if not spec_radius < 1:
        errors.append(f'spectral radius not < 1, but {spec_radius}')
    # assert no error message has been registered, else print messages
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


param_n_boundaries = [((236,), 0), ((236,), 10),
                      ((30, 32), 0), ((30, 32), 10),
                      ((5, 6, 7), 0), ((5, 6, 7), 1)]


@pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [1])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_1domain_wrap_options(v_contraction):
    """ Check that V is a contraction for 1-domain scenario for all wrapping correction options """
    norm_, spec_radius = v_contraction
    # assert norm_ < 1, f'||op|| not < 1, but {norm_}'
    # assert spec_radius < 1, f'spectral radius not < 1, but {spec_radius}'
    check_assertions(norm_, spec_radius)


@pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
@pytest.mark.parametrize("n_domains", [2])
@pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
def test_ndomains(v_contraction):
    """ Check that V is a contraction when number of domains > 1 
    (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
    norm_, spec_radius = v_contraction
    # assert norm_ < 1, f'||op|| not < 1, but {norm_}'
    # assert spec_radius < 1, f'spectral radius not < 1, but {spec_radius}'
    check_assertions(norm_, spec_radius)