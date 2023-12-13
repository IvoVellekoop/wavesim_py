import pytest
import numpy as np
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse import diags as spdiags

from helmholtzbase import HelmholtzBase
from utilities import full_matrix, relative_error


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((15, 16)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_accretive(n, boundary_widths, wrap_correction):
    """ Check that operator A = L + V (sum of propagator and medium operators) is accretive
        , i.e., has a non-negative real part """
    source = np.zeros_like(n)
    source[0] = 1.
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    d = base.domain_size[:base.n_dims]
    patch = (0, 0, 0)
    if wrap_correction == 'L_omega':
        l_plus_1_operator = lambda x: (np.fft.ifftn((base.scaling[patch] * base.l_p + 1) *
                                       np.fft.fftn(x, d*10)))[
                                       tuple([slice(0, d[i]) for i in range(base.n_dims)])]
    else:
        l_plus_1_operator = lambda x: np.fft.ifftn((base.scaling[patch] * base.l_p + 1) * np.fft.fftn(x))
    l_plus_1 = full_matrix(l_plus_1_operator, d)
    b = full_matrix(base.medium_operators[patch], d)
    a = (l_plus_1 - b).todense()

    acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))
    print(f'acc {acc:.2e}')
    assert np.round(acc, 5) >= 0, f'a is not accretive. {acc}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((15, 16)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_contraction(n, boundary_widths, wrap_correction):
    """ Check that potential V is a contraction,
        i.e., the operator norm ||V|| < 1 """
    # n = np.ones((256, 1, 1))
    source = np.zeros_like(n)
    source[0] = 1.
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    d = base.domain_size[:base.n_dims]
    patch = (0, 0, 0)
    if wrap_correction == 'wrap_corr':
        scaling = base.scaling[patch]
        v_corr = spdiags(base.v.ravel(), dtype=np.complex64) - scaling * full_matrix(base.wrap_corr, d)
    else:
        # vc = np.max(np.abs(base.v))
        v_corr = (full_matrix(base.medium_operators[patch], d) 
                  - spdiags(np.ones(np.prod(d)), dtype=np.complex64))
    vc = spnorm(v_corr, 2)
    print(f'vc {vc:.2f}')
    assert vc < 1, f'||V|| not < 1, but {vc}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((15, 16)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
def test_compare_A(n, boundary_widths):
    """ Check that the operator (A) is the same for wrap_correction = ['wrap_corr', 'L_omega'] """
    source = np.zeros_like(n)
    source[0] = 1.
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base_w = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, 
                           wrap_correction='wrap_corr', n_correction=n_correction)
    d = base_w.domain_size[:base_w.n_dims]
    patch = (0, 0, 0)
    scaling_w = base_w.scaling[patch]
    l_w_operator = lambda x: np.fft.ifftn((scaling_w * base_w.l_p) * np.fft.fftn(x))
    l_w = full_matrix(l_w_operator, d)
    v_w = spdiags(base_w.v.ravel(), dtype=np.complex64) - scaling_w * full_matrix(base_w.wrap_corr, d)
    a_w = (l_w + v_w)/scaling_w

    base_o = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='L_omega')    
    n_ = (base_o.domain_size[:base_o.n_dims]).astype(int)
    scaling_o = base_o.scaling[patch]
    l_o_operator = lambda x: (np.fft.ifftn((scaling_o * base_o.l_p) * np.fft.fftn(x, n_*10)))[
                                           tuple([slice(0, n_[i]) for i in range(base_o.n_dims)])]
    l_o = full_matrix(l_o_operator, d)
    v_o = np.diag(base_o.v.ravel())
    a_o = (l_o + v_o)/scaling_o

    # base = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction=None)
    # l_plus_1_operator = lambda x: np.fft.ifftn((base.scaling[patch] * base.l_p + 1) * np.fft.fftn(x))
    # l_plus_1 = full_matrix(l_plus_1_operator, d)
    # b = full_matrix(base.medium_operators[patch], d)
    # a = (l_plus_1 - b)/base.scaling[patch]

    rel_err = relative_error(a_w, a_o)
    print(f'{rel_err:.2e}')
    # print(f'{rel_err:.2e}, {relative_error(a, a_o):.2e}, {relative_error(a, a_w):.2e}')
    assert rel_err <= 1.e-3, f'Operator A (wrap_corr case) != A (L_omega case). relative error {rel_err:.2e}'


# def test_subdomain_op_reconstruction():
#     """ Check splitting of operators into subdomains still gives the original operator after reconstruction """
#     n = np.ones((256, 1, 1), dtype=np.float32)
#     source = np.zeros_like(n)
#     source[0] = 1.

#     # Get the operator A = (L+1)-B = (L+1)-(1-V) = L+V for the full-domain problem (baseline to compare against)
#     base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction='wrap_corr', cp=296)
#     x = np.eye(base.domain_size[0], dtype=np.float32)
#     l_plus_1_inv = base.propagator(x, base.scaling[base.domains_iterator[0]])
#     l_plus_1 = np.linalg.inv(l_plus_1_inv)
#     b = base.medium_operators[base.domains_iterator[0]](x)
#     a = l_plus_1 - b

#     # Get the subdomain operators and transfer corrections (2 subdomains) and reconstruct A
#     base2 = HelmholtzBase(n=n, source=source, n_domains=2, wrap_correction='wrap_corr', cp=296)
#     sub_n = base2.domain_size[0]
#     x_ = np.eye(sub_n, dtype=np.float32)

#     # (L+1) and B for subdomain 1
#     l_plus_1_inv_1 = base2.propagator(x_, base2.scaling[base2.domains_iterator[0]])
#     l_plus_1_1 = np.linalg.inv(l_plus_1_inv_1)
#     b1 = base2.medium_operators[base2.domains_iterator[0]](x_)

#     # (L+1) and B for subdomain 2
#     l_plus_1_inv_2 = base2.propagator(x_, base2.scaling[base2.domains_iterator[1]])
#     l_plus_1_2 = np.linalg.inv(l_plus_1_inv_2)
#     b2 = base2.medium_operators[base2.domains_iterator[1]](x_)

#     # Transfer corrections
#     b1_corr = base2.transfer(x_, base2.scaling[base2.domains_iterator[0]], +1)
#     b2_corr = base2.transfer(x_, base2.scaling[base2.domains_iterator[1]], -1)

#     # Reconstruct A using above subdomain operators and transfer corrections
#     a_reconstructed = np.zeros_like(a, dtype=np.complex64)
#     a_reconstructed[:sub_n, :sub_n] = l_plus_1_1 - b1  # Subdomain 1
#     a_reconstructed[sub_n:, sub_n:] = l_plus_1_2 - b2  # Subdomain 2
#     a_reconstructed[:sub_n, sub_n:] = b1_corr  # Transfer correction
#     a_reconstructed[sub_n:, :sub_n] = b2_corr  # Transfer correction
#     rel_err = relative_error(a_reconstructed, a)
#     print(f'Relative error between A reconstructed and A: {rel_err:.2e}')
#     assert rel_err <= 1.e-6, f'operator A not reconstructed properly. relative error high {rel_err:.2e}'
