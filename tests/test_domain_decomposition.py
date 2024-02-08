import pytest
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain, precon_iteration
from utilities import pad_boundaries_torch, relative_error, squeeze_
import torch
torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("n", 
                         [np.ones(256), 
                          np.ones((220, 256)), 
                          np.ones((30, 26, 29))])
@pytest.mark.parametrize("n_domains", [2, 3])
def test_forward_iteration(n, n_domains):
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    # 1 domain problem
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction='wrap_corr')
    x = (torch.rand(*base.s.shape) + 1j*torch.rand(*base.s.shape)).to(base.device)
    patch = (0, 0, 0)  # 1 domain so only 1 patch
    x_dict = defaultdict(list)
    x_dict[patch] = x
    l_plus1_x = base.l_plus1_operators[patch](x_dict[patch])
    b_x = base.medium_operators[patch](x_dict[patch])
    a_x = (l_plus1_x - b_x)/base.scaling[patch]

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')
    x2 = pad_boundaries_torch(x, (0, 0, 0), tuple(np.array(base2.s.shape)-np.array(base.s.shape)),
                              mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    x_dict2 = defaultdict(list)
    for patch2 in base2.domains_iterator:
        x_dict2[patch2] = map_domain(x2, restrict, patch2)
    l_plus1_x2 = base2.l_plus1(x_dict2)
    b_x2 = base2.medium(x_dict2)
    a_x2 = 0.
    for patch2 in base2.domains_iterator:
        a_x2_patch = (l_plus1_x2[patch2] - b_x2[patch2])/base2.scaling[patch2]
        a_x2 += map_domain(a_x2_patch, extend, patch2)

    if (base.boundary_post != 0).any():
        a_x = a_x[base.crop2roi]
    if (base2.boundary_post != 0).any():
        a_x2 = a_x2[base2.crop2roi]
    a_x = squeeze_(a_x.cpu().numpy())
    a_x2 = squeeze_(a_x2.cpu().numpy())
    rel_err = relative_error(a_x2, a_x)

    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3


@pytest.mark.parametrize("n", 
                         [np.ones(256), 
                          np.ones((40, 42)), 
                          np.ones((10, 10, 10))])
@pytest.mark.parametrize("n_domains", [2])
def test_precon_iteration(n, n_domains):
    iterations = 5000
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    # 1 domain problem
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction=None)
    u = (torch.rand(*base.s.shape) + 1j*torch.rand(*base.s.shape)).to(base.device)

    _, extend = domain_decomp_operators(base)
    patch = (0, 0, 0)  # 1 domain so only 1 patch
    s_dict = defaultdict(list)
    u_dict = s_dict.copy()
    ut_dict = s_dict.copy()
    s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * base.s
    u_dict[patch] = u

    for _ in range(iterations):
        t_dict = precon_iteration(base, u_dict, ut_dict, s_dict)
        for patch in base.domains_iterator:
            u_dict[patch] = u_dict[patch] - (base.alpha * t_dict[patch])
    t1 = 0.
    for patch in base.domains_iterator:
        t1_patch = np.sqrt(base.scaling[patch]) * u_dict[patch]
        t1 += map_domain(t1_patch, extend, patch)

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')
    u2 = pad_boundaries_torch(u, (0, 0, 0), tuple(np.array(base2.s.shape)-np.array(base.s.shape)),
                              mode="constant")
    restrict2, extend2 = domain_decomp_operators(base2)
    s_dict2 = defaultdict(list)
    u_dict2 = s_dict2.copy()
    ut_dict2 = s_dict2.copy()
    for patch2 in base2.domains_iterator:
        s_dict2[patch2] = 1j * np.sqrt(base2.scaling[patch2]) * map_domain(base2.s, restrict2, patch2)
        u_dict2[patch2] = map_domain(u2, restrict2, patch2)

    for _ in range(iterations):
        t_dict2 = precon_iteration(base2, u_dict2, ut_dict2, s_dict2)
        for patch2 in base2.domains_iterator:
            u_dict2[patch2] = u_dict2[patch2] - (base2.alpha * t_dict2[patch2])
    t2 = 0.
    for patch2 in base2.domains_iterator:
        t2_patch = np.sqrt(base2.scaling[patch2]) * u_dict2[patch2]
        t2 += map_domain(t2_patch, extend2, patch2)

    t1 = squeeze_(t1.cpu().numpy())
    t2 = squeeze_(t2.cpu().numpy())
    if (base2.boundary_post != 0).any():
        t2 = t2[base2.crop2roi]
    if (base.boundary_post != 0).any():
        t1 = t1[base.crop2roi]

    rel_err = relative_error(t2, t1)
    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3
