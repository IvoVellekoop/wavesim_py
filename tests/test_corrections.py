import pytest
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain
from utilities import full_matrix, max_abs_error, pad_boundaries_torch, relative_error, squeeze_
from torch import rand, complex64, zeros_like


@pytest.mark.parametrize("n_size, boundary_widths", [(236, 0), (236, 10),
                                                     ((90, 100), 0), ((90, 100), 10),
                                                     ((30, 26, 29), 0), ((30, 26, 29), 10)])
def test_compare_a(n_size, boundary_widths):
    """ Check that the operator (A) is the same for wrap_correction = ['wrap_corr', 'L_omega'] """
    n = np.ones(n_size, dtype=np.complex64)
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    base_w = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='wrap_corr', scaling=1.)
    x = rand(*base_w.s.shape, dtype=complex64, device=base_w.device)
    patch = (0, 0, 0)
    x_dict = defaultdict(list)
    x_dict[patch] = x
    l_w_plus1 = base_w.l_plus1(x_dict)[patch]
    b_w = base_w.medium(x_dict)[patch]
    a_w = l_w_plus1 - b_w

    base_o = HelmholtzBase(n=n, source=source, boundary_widths=boundary_widths, wrap_correction='L_omega', scaling=1.)
    x2 = pad_boundaries_torch(x, (0, 0, 0), tuple(np.array(base_o.s.shape)-np.array(base_w.s.shape)),
                              mode="constant")
    x2_dict = defaultdict(list)
    x2_dict[patch] = x2.clone()
    l_o_plus1 = base_o.l_plus1(x2_dict)[patch]
    b_o = base_o.medium(x2_dict)[patch]
    a_o = l_o_plus1 - b_o

    if boundary_widths != 0:
        # crop to n_roi, excluding boundaries
        crop2roi = tuple([slice(base_w.boundary_pre[0], -base_w.boundary_post[0]) for _ in range(base_w.n_dims)])
        a_w = a_w[crop2roi]
        a_o = a_o[crop2roi]
    a_w = squeeze_(a_w.cpu().numpy())
    a_o = squeeze_(a_o.cpu().numpy())

    rel_err = relative_error(a_w, a_o)
    mae = max_abs_error(a_w, a_o)
    print(f'Relative error ({rel_err:.2e})')
    print(f'Max absolute error (Normalized) ({mae:.2e})')
    assert rel_err <= 1.e-3, f'Operator A (wrap_corr case) != A (L_omega case). Relative Error {rel_err:.2e}'
    assert mae <= 1.e-3, f'Operator A (wrap_corr case) != A (L_omega case). Max absolute error (Normalized) {mae:.2e}'


@pytest.mark.parametrize("n_size, boundary_widths", [(236, 0), (236, 10),
                                                     ((30, 32), 0), ((30, 32), 10),
                                                     ((5, 6, 7), 0), ((5, 6, 7), 1)])
@pytest.mark.parametrize("n_domains", [1, 2])
@pytest.mark.parametrize("corr_type", ['wrapping', 'transfer'])
def test_symmetry(n_size, boundary_widths, n_domains, corr_type):
    n = np.ones(n_size, dtype=np.complex64)
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, n_domains=n_domains, wrap_correction='wrap_corr',
                         scaling=1.)
    restrict, extend = domain_decomp_operators(base)

    def corr(x):
        u_dict = defaultdict(list)
        for patch in base.domains_iterator:
            u_dict[patch] = map_domain(x, restrict, patch)

        c_dict = defaultdict(list)
        y = zeros_like(x, dtype=complex64, device=base.device)
        for patch in base.domains_iterator:
            c_dict[patch] = map_domain(y, restrict, patch)

        c_dict = base.apply_corrections(u_dict, c_dict, corr_type, im=False)
        c_ = 0.
        for patch in base.domains_iterator:
            c_ += map_domain(c_dict[patch], extend, patch)
        return c_

    n_ext = base.n_roi + base.boundary_pre + base.boundary_post
    c = full_matrix(corr, n_ext)
    acc = np.min(np.real(np.linalg.eigvals(1j*c + (1j*c).conj().T)))
    print(f'acc {acc:.2e}')

    assert np.allclose(c, c.T)
    assert np.round(acc, 3) >= 0, f'{corr_type} correction is not accretive. {acc}'
