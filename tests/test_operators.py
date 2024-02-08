import pytest
import numpy as np
from collections import defaultdict
# from scipy.sparse import diags as spdiags
# from scipy.sparse.linalg import norm as spnorm

import torch
torch.set_default_dtype(torch.float32)

from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain
from utilities import full_matrix, pad_boundaries_torch, relative_error

@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((20, 21)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_accretive(n, boundary_widths, wrap_correction):
    """ Check that operator A = L + V (sum of propagator and medium operators) is accretive
        , i.e., has a non-negative real part """
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    patch = (0, 0, 0)
    l_plus1 = full_matrix(base.l_plus1_operators[patch], base.domain_size)
    b = full_matrix(base.medium_operators[patch], base.domain_size)
    # a = (l_plus1 - b).toarray()
    a = l_plus1 - b

    # acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))
    acc = torch.min(torch.real(torch.linalg.eigvals(a + a.conj().t())))
    print(f'acc {acc:.2e}')
    # assert np.round(acc, 12) >= 0, f'a is not accretive. {acc}'
    assert torch.round(acc, decimals=3) >= 0, f'a is not accretive. {acc}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 10), 
                                                (np.ones((20, 21)), 0), (np.ones((20, 21)), 10),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_v_contraction(n, boundary_widths, wrap_correction):
    """ Check that potential V is a contraction,
        i.e., the operator norm ||V|| < 1 """
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    patch = (0, 0, 0)
    if wrap_correction == 'wrap_corr':
        scaling = base.scaling[patch]
        # v_corr = spdiags(base.v.ravel(), dtype=np.complex64) - scaling * full_matrix(base.wrap_corr, base.domain_size)
        v_corr = torch.diag(torch.tensor(base.v.ravel(), dtype=torch.complex64, device=base.device)) - scaling * full_matrix(base.wrap_corr, base.domain_size)
    else:
        # # vc = np.max(np.abs(base.v))
        # v_corr = (full_matrix(base.medium_operators[patch], base.domain_size) 
        #           - spdiags(np.ones(np.prod(base.domain_size)), dtype=np.complex64))
        v_corr = (full_matrix(base.medium_operators[patch], base.domain_size) 
                  - torch.diag(torch.ones(np.prod(base.domain_size), dtype=torch.complex64, device=base.device)))
    # vc = spnorm(v_corr, 2)
    vc = np.linalg.norm(v_corr.cpu(), 2)
    print(f'vc {vc:.2f}')
    assert vc < 1, f'||V|| not < 1, but {vc}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 10), 
                                                (np.ones((30, 32)), 0), (np.ones((30, 32)), 10),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_op_contraction(n, boundary_widths, wrap_correction):
    """ Check that preconditioned operator is a contraction,
        i.e., the operator norm || 1 - B[1 - (L+1)^(-1)B] || < 1 """
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    patch = (0, 0, 0)
    op_ = lambda x: base.medium_operators[patch](x - base.propagator_operators[patch](base.medium_operators[patch](x)))

    n_ext = base.n_roi + base.boundary_pre + base.boundary_post
    # mat_ = spdiags(np.ones(np.prod(n_ext)), dtype=np.complex64) - base.alpha * full_matrix(op_, n_ext)
    mat_ = torch.diag(torch.ones(np.prod(n_ext), dtype=torch.complex64, device=base.device)) - base.alpha * full_matrix(op_, n_ext)
    # norm_ = spnorm(mat_, 2)
    norm_ = np.linalg.norm(mat_.cpu(), 2)

    print(f'norm_ {norm_}')
    assert norm_ < 1, f'||op|| not < 1, but {norm_}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 4), 
                                                (np.ones((256, 256)), 0), (np.ones((256, 256)), 4),
                                                (np.ones((48, 48, 48)), 0), (np.ones((40, 40, 40)), 4)])
def test_compare_A(n, boundary_widths):
    """ Check that the operator (A) is the same for wrap_correction = ['wrap_corr', 'L_omega'] """
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    base_w = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='wrap_corr')
    # x = (np.random.rand(*base_w.s.shape) + 1j*np.random.rand(*base_w.s.shape)).astype(np.complex64)
    x = (torch.rand(*base_w.s.shape) + 1j*torch.rand(*base_w.s.shape)).to(base_w.device)
    patch = (0, 0, 0)
    scaling_w = base_w.scaling[patch]
    l_w_plus1 = base_w.l_plus1_operators[patch](torch.tensor(x))
    b_w = base_w.medium_operators[patch](x)
    a_w = (l_w_plus1 - b_w)/scaling_w

    base_o = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='L_omega')    
    # x2 = pad_boundaries(x, (0,0,0), np.array(base_o.s.shape)-np.array(base_w.s.shape), mode="constant")
    x2 = pad_boundaries_torch(x, (0,0,0), tuple(np.array(base_o.s.shape)-np.array(base_w.s.shape)), mode="constant")
    scaling_o = base_o.scaling[patch]
    l_o_plus1 = base_o.l_plus1_operators[patch](x2)
    b_o = base_o.medium_operators[patch](x2)
    a_o = (l_o_plus1 - b_o)/scaling_o

    # base = HelmholtzBase(n=n, boundary_widths=boundary_widths, wrap_correction=None)
    # l_plus1 = full_matrix(base.l_plus1_operator[patch], base.domain_size)
    # b = full_matrix(base.medium_operators[patch], base.domain_size)
    # a = (l_plus1 - b)/base.scaling[patch]

    a_w = np.squeeze(a_w.cpu().numpy())
    a_o = np.squeeze(a_o.cpu().numpy())
    if boundary_widths != 0:
        crop2roi = tuple([slice(base_w.boundary_pre[0], -base_w.boundary_post[0]) 
                          for _ in range(base_w.n_dims)])  # crop to n_roi, excluding boundaries
        a_w = a_w[crop2roi]
        a_o = a_o[crop2roi]

    rel_err = relative_error(a_w, a_o)
    print(f'Relative error \t\t {rel_err:.2e}')
    # print(f'{rel_err:.2e}, {relative_error(a, a_o):.2e}, {relative_error(a, a_w):.2e}')
    assert rel_err <= 1.e-4, f'Operator A (wrap_corr case) != A (L_omega case). relative error {rel_err:.2e}'
