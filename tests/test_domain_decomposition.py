import pytest
import numpy as np
from collections import defaultdict
from scipy.sparse import diags as spdiags
from scipy.sparse.linalg import norm as spnorm

import torch
torch.set_default_dtype(torch.float32)

from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain
from utilities import full_matrix, pad_boundaries_torch, relative_error

@pytest.mark.parametrize("n", 
                         [np.ones(256), 
                          np.ones((256, 256)), 
                          np.ones((64, 52, 58))])
@pytest.mark.parametrize("n_domains", [2, 3])
def test_forward_iteration(n, n_domains):
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    # 1 domain problem
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction='wrap_corr')
    # x = (np.random.rand(*base.s.shape) + 1j*np.random.rand(*base.s.shape)).astype(np.complex64)
    x = (torch.rand(*base.s.shape) + 1j*torch.rand(*base.s.shape)).to(base.device)

    patch = (0, 0, 0)  # 1 domain so only 1 patch
    x_dict = defaultdict(list)
    x_dict[patch] = x.clone()
    l_plus1_x = base.l_plus1_operators[patch](x_dict[patch])
    b_x = base.medium_operators[patch](x_dict[patch])
    a_x = (l_plus1_x - b_x)/base.scaling[patch]

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')

    # x2 = pad_boundaries(x, (0,0,0), np.array(base2.s.shape)-np.array(base.s.shape), mode="constant")
    x2 = pad_boundaries_torch(x, (0,0,0), tuple(np.array(base2.s.shape)-np.array(base.s.shape)), mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    x_dict2 = defaultdict(list)
    l_plus1_x2 = defaultdict(list)
    b_x2 = defaultdict(list)
    # z = np.zeros_like(x2, dtype=np.complex64)
    z = torch.zeros_like(x2, dtype=torch.complex64, device=base.device)
    for patch2 in base2.domains_iterator:
        x_dict2[patch2] = map_domain(x2, restrict, patch2)
        b_x2[patch2] = map_domain(z, restrict, patch2)

    l_plus1_x2 = base2.l_plus1(x_dict2)
    b_x2 = base2.medium(x_dict2)
    b_x2 = base2.transfer(x_dict2, b_x2)

    a_x2 = 0.
    for patch2 in base2.domains_iterator:
        a_x2_patch = (l_plus1_x2[patch2] - b_x2[patch2])/base2.scaling[patch2]
        a_x2 += map_domain(a_x2_patch, extend, patch2)

    if (base.boundary_post != 0).any():
        a_x = a_x[base.crop2roi]
    if (base2.boundary_post != 0).any():
        a_x2 = a_x2[base2.crop2roi]
    a_x = torch.squeeze(a_x).cpu().numpy()
    a_x2 = torch.squeeze(a_x2).cpu().numpy()
    rel_err = relative_error(a_x2, a_x)

    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 10), 
                                                (np.ones((30, 32)), 0), (np.ones((30, 32)), 10),
                                                (np.ones((10, 10, 10)), 0), (np.ones((10, 10, 10)), 10)])
@pytest.mark.parametrize("n_domains", [2])
def test_op_contraction(n, boundary_widths, n_domains):
    """ Check that preconditioned operator is a contraction,
        i.e., the operator norm || 1 - B[1 - (L+1)^(-1)B] || < 1 """
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         wrap_correction='wrap_corr', n_domains=n_domains)
    restrict, extend = domain_decomp_operators(base)
    def op_(x):
        u_ = defaultdict(list)
        for patch in base.domains_iterator:
            u_[patch] = map_domain(x, restrict, patch)
        t_dict = base.medium(u_)
        t_dict = base.transfer(u_, t_dict)
        t_dict = base.propagator(t_dict)
        ut_dict = defaultdict(list)
        for patch in base.domains_iterator:
            ut_dict[patch] = u_[patch] - t_dict[patch]
        t_dict = base.medium(ut_dict)
        t_dict = base.transfer(ut_dict, t_dict)
        t = 0.
        for patch in base.domains_iterator:
            t += map_domain(t_dict[patch], extend, patch)
        return t

    n_ext = base.n_roi + base.boundary_pre + base.boundary_post
    # mat_ = spdiags(np.ones(np.prod(n_ext)), dtype=np.complex64) - base.alpha * full_matrix(op_, n_ext)
    mat_ = torch.diag(torch.ones(np.prod(n_ext), dtype=torch.complex64, device=base.device)) - base.alpha * full_matrix(op_, n_ext)
    # norm_ = spnorm(mat_, 2)
    norm_ = np.linalg.norm(mat_.cpu().numpy(), 2)

    print(f'norm_ {norm_}')
    assert norm_ < 1, f'||op|| not < 1, but {norm_}'


@pytest.mark.parametrize("n", 
                         [np.ones(256), 
                          np.ones((40, 42)), 
                          np.ones((10, 10, 10))])
@pytest.mark.parametrize("n_domains", [2])
def test_precon_iteration(n, n_domains):
    iterations = 10000
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    # 1 domain problem
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction=None)
    # u = (np.random.rand(*base.s.shape) + 1j*np.random.rand(*base.s.shape)).astype(np.complex64)
    u = (torch.rand(*base.s.shape) + 1j*torch.rand(*base.s.shape)).to(base.device)

    restrict, extend = domain_decomp_operators(base)
    patch = (0, 0, 0)  # 1 domain so only 1 patch
    s_dict = defaultdict(list)
    u_dict = defaultdict(list)
    ut_dict = defaultdict(list)
    s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * base.s
    u_dict[patch] = u.clone()

    for _ in range(iterations):
        t_dict = base.medium(u_dict, s_dict)
        t_dict = base.transfer(u_dict, t_dict)
        t_dict = base.propagator(t_dict)
        for patch in base.domains_iterator:
            ut_dict[patch] = u_dict[patch] - t_dict[patch]
        t_dict = base.medium(ut_dict)
        t_dict = base.transfer(ut_dict, t_dict)
        for patch in base.domains_iterator:
            u_dict[patch] = u_dict[patch] - (base.alpha * t_dict[patch])
    t1 = 0.
    for patch in base.domains_iterator:
        t1_patch = np.sqrt(base.scaling[patch]) * u_dict[patch]
        t1 += map_domain(t1_patch, extend, patch)
    if (base.boundary_post != 0).any():
        t1 = t1[base.crop2roi]
    t1 = torch.squeeze(t1).cpu().numpy()

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')
    # u2 = pad_boundaries(u, (0,0,0), np.array(base2.s.shape)-np.array(base.s.shape), mode="constant")
    u2 = pad_boundaries_torch(u, (0,0,0), tuple(np.array(base2.s.shape)-np.array(base.s.shape)), mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    s_dict2 = defaultdict(list)
    u_dict2 = defaultdict(list)
    ut_dict2 = defaultdict(list)
    for patch2 in base2.domains_iterator:
        s_dict2[patch2] = 1j * np.sqrt(base2.scaling[patch2]) * map_domain(base2.s, restrict, patch2)
        u_dict2[patch2] = map_domain(u2, restrict, patch2)

    for _ in range(iterations):
        t_dict2 = base2.medium(u_dict2, s_dict2)
        t_dict2 = base2.transfer(u_dict2, t_dict2)
        t_dict2 = base2.propagator(t_dict2)
        for patch2 in base2.domains_iterator:
            ut_dict2[patch2] = u_dict2[patch2] - t_dict2[patch2]
        t_dict2 = base2.medium(ut_dict2)
        t_dict2 = base2.transfer(ut_dict2, t_dict2)
        for patch2 in base2.domains_iterator:
            u_dict2[patch2] = u_dict2[patch2] - (base2.alpha * t_dict2[patch2])
    t2 = 0.
    for patch2 in base2.domains_iterator:
        t2_patch = np.sqrt(base2.scaling[patch2]) * u_dict2[patch2]
        t2 += map_domain(t2_patch, extend, patch2)
    if (base2.boundary_post != 0).any():
        t2 = t2[base2.crop2roi]
    t2 = torch.squeeze(t2).cpu().numpy()

    rel_err = relative_error(t2, t1)

    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3
