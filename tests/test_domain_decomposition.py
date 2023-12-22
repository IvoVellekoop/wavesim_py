import pytest
import numpy as np
from collections import defaultdict

from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain, iterate1, iterate2
from utilities import relative_error

@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((16, 18)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
def test_iteration(n, boundary_widths):
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, 
                         n_domains=1, wrap_correction='wrap_corr', 
                         n_correction=n_correction)
    
    u = (np.random.rand(*base.s.shape) + 1j*np.random.rand(*base.s.shape)).astype(np.complex64)

    patch = (0, 0, 0)
    s_dict = defaultdict(list)
    u_dict = defaultdict(list)
    s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * base.s
    u_dict[patch] = u.copy()
    ut_dict = u_dict.copy()
    ut_dict[patch] = iterate1(base, u_dict, s_dict, patch)
    t1 = iterate2(base, ut_dict, patch)

    base2 = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, 
                          n_domains=2, wrap_correction='wrap_corr', 
                          n_correction=n_correction)
    restrict, extend = domain_decomp_operators(base2)
    s_dict2 = defaultdict(list)
    u_dict2 = defaultdict(list)
    for patch2 in base2.domains_iterator:
        s_dict2[patch2] = 1j * np.sqrt(base2.scaling[patch2]) * map_domain(base2.s, restrict, patch2)
        u_dict2[patch2] = map_domain(u, restrict, patch2)
    ut_dict2 = u_dict2.copy()

    for patch2 in base2.domains_iterator:
        ut_dict2[patch2] = iterate1(base2, u_dict2, s_dict2, patch2)
    t2 = 0.
    for patch2 in base2.domains_iterator:
        t2_patch = iterate2(base2, ut_dict2, patch2)
        t2 += map_domain(t2_patch, extend, patch2)

    if boundary_widths == 0:
        rel_err = relative_error(np.squeeze(t2), np.squeeze(t1))
    else:
        rel_err = relative_error(np.squeeze(t2[base2.crop2roi]), np.squeeze(t1[base.crop2roi]))
    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-2
