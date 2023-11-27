import pytest
import numpy as np
from helmholtzbase import HelmholtzBase
from preprocess import full_matrix, relative_error


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 20)), 0), (np.ones((10, 10)), 5),
                                                (np.ones((5, 5, 5)), 0), (np.ones((5, 5, 5)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_accretive(n, boundary_widths, wrap_correction):
    """ Check that operator A = L + V (sum of propagator and medium operators) is accretive
        , i.e., has a non-negative real part """
    source = np.zeros_like(n)
    source[0] = 1.
    base = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction=wrap_correction)
    d = base.domain_size[:base.n_dims]
    patch = (0, 0, 0)
    if wrap_correction == 'L_omega':
        l_plus_1_operator = lambda x: (np.fft.ifftn((base.l_p + 1) *
                                       np.fft.fftn(np.pad(x, (0, (d*base.omega)[0] - d[0])))))[
                                       tuple([slice(0, d[i]) for i in range(base.n_dims)])]
    else:
        l_plus_1_operator = lambda x: np.fft.ifftn((base.l_p + 1) * np.fft.fftn(x))
    l_plus_1 = full_matrix(l_plus_1_operator, d)
    b = full_matrix(base.medium_operators[patch], d)
    a = (l_plus_1 - b).todense()

    acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))
    print(f'acc {acc:.2e}')
    assert np.round(acc, 5) >= 0, f'a is not accretive. {acc}'


@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_contraction(wrap_correction):
    """ Check that potential V is a contraction,
        i.e., the operator norm ||V|| < 1 """
    n = np.ones((256, 1, 1), dtype=np.float32)
    source = np.zeros_like(n)
    source[0] = 1.
    base = HelmholtzBase(n=n, source=source, wrap_correction=wrap_correction)
    if wrap_correction == 'wrap_corr':
        wrap_operator = lambda x: base.scaling[(0, 0, 0)] * base.wrap_corr(x)
        vc = np.linalg.norm(np.diag(base.v) + full_matrix(wrap_operator, base.domain_size[:base.n_dims]), 2)
    else:
        vc = np.max(np.abs(base.v))
    print(f'vc {vc:.2e}')
    assert vc < 1, f'||V|| not < 1, but {vc}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 20)), 0), (np.ones((10, 10)), 5),
                                                (np.ones((5, 5, 5)), 0), (np.ones((5, 5, 5)), 1)])
def test_compare_A(n, boundary_widths):
    """ Check that the operator (A) is the same for wrap_correction = ['wrap_corr', 'L_omega'] """
    source = np.zeros_like(n)
    source[0] = 1.

    base_w = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='wrap_corr')
    d = base_w.domain_size[:base_w.n_dims]
    patch = (0, 0, 0)
    l_w_operator = lambda x: np.fft.ifftn((base_w.l_p) * np.fft.fftn(x))
    l_w = full_matrix(l_w_operator, d)
    # b_w = full_matrix(base_w.medium_operators[patch], d)
    wrap_operator = lambda x: base_w.scaling[patch] * base_w.wrap_corr(x)
    v_w = np.diag(base_w.v.ravel()) + full_matrix(wrap_operator, base_w.domain_size[:base_w.n_dims])
    a_w = (l_w + v_w)/base_w.scaling[patch]

    base_o = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='L_omega')    
    n_ = (base_o.domain_size[:base_o.n_dims]).astype(int)
    l_o_operator = lambda x: (np.fft.ifftn((base_o.l_p) *
                                     np.fft.fftn(np.pad(x, (0, (n_*base_o.omega)[0] - n_[0])))))[
                                     tuple([slice(0, n_[i]) for i in range(base_o.n_dims)])]
    l_o = full_matrix(l_o_operator, d)
    # b_o = full_matrix(base_o.medium_operators[patch], d)
    v_o = np.diag(base_o.v.ravel())
    a_o = (l_o + v_o)/base_o.scaling[patch]

    # base = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction=None)
    # l_plus_1_operator = lambda x: np.fft.ifftn((base.l_p + 1) * np.fft.fftn(x))
    # l_plus_1 = full_matrix(l_plus_1_operator, d)
    # b = full_matrix(base.medium_operators[patch], d)
    # a = (l_plus_1 - b)/base.scaling[patch]

    rel_err = relative_error(a_w, a_o)
    print(f'{rel_err:.2e}')
    # print(f'{rel_err:.2e}, {relative_error(a, a_o):.2e}, {relative_error(a, a_w):.2e}')
    assert rel_err <= 1.e-12, f'Operator A (wrap_corr case) != A (L_omega case). relative error {rel_err:.2e}'


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
