import pytest
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain, precon_iteration
from utilities import pad_boundaries_torch, max_abs_error, relative_error, squeeze_, max_relative_error
import torch

torch.set_default_dtype(torch.float32)


def forward_operator(x, n, n_domains, n_correction):
    """Construct forward operator for given number of subdomains and compute its action on x
    The result should be the same regardless of n_domains"""
    base = HelmholtzBase(n=n, n_domains=n_domains, wrap_correction='wrap_corr', boundary_widths=0,
                         n_correction=n_correction)

    # Split x into subdomains
    restrict, extend = domain_decomp_operators(base)
    x_dict = defaultdict(list)
    for patch in base.domains_iterator:
        x_dict[patch] = map_domain(x, restrict, patch)

    # Compute action of A on x
    x_l_plus1 = base.l_plus1(x_dict)
    x_medium = base.medium(x_dict)

    # merge subdomains
    y = torch.zeros_like(x)
    for patch in base.domains_iterator:
        y_domain = x_l_plus1[patch] - x_medium[patch]
        y += map_domain(y_domain / base.scaling[patch], extend, patch).cpu()

    return y.cpu().numpy()


def test_simple_decomposition():
    """Test decomposition of the forward operator A"""
    # shape = (128, 192, 256)
    # n_domains = (2, 3, 4)
    shape = (300, 300, 300)
    n_domains = (3, 3, 3)

    n = torch.ones(size=shape)  # refractive index
    x = torch.zeros(size=shape, dtype=torch.complex64)
    x[0, 0, 0] = 1.0
    x[300 // 2, 300 // 2, 300 // 2] = 1.0
    x[300 // 2, 1, 1] = 1.0
    y_full = forward_operator(x, n, n_domains=(1, 1, 1), n_correction=40)
    y_split = forward_operator(x, n, n_domains=n_domains, n_correction=40)
    # s1 = np.squeeze(y_full[300 // 2, :, :])
    # s2 = np.squeeze(y_split[300 // 2, :, :])

    print(f"relative error: {relative_error(y_full, y_split)}")
    print(f"absolute error: {max_abs_error(y_full, y_split)}")
    print(f"maximum_relative error: {max_relative_error(y_full, y_split)}")

    # import matplotlib.pyplot as plt
    # plt.imshow(abs(s1 - s2))
    # plt.colorbar()
    # plt.show()
    assert np.allclose(y_full, y_split)


param_n = [1004, (90, 100), (30, 26, 29)]


@pytest.mark.parametrize("n_size", param_n)
@pytest.mark.parametrize("n_domains", [2, 3])
def test_forward_iteration(n_size, n_domains):
    n = np.ones(n_size, dtype=np.complex64)

    # 1 domain problem
    base = HelmholtzBase(n=n, n_domains=1, wrap_correction='wrap_corr')
    x = torch.rand(*base.s.shape, dtype=torch.complex64)
    patch = (0, 0, 0)  # 1 domain so only 1 patch
    x_dict = defaultdict(list)
    x_dict[patch] = x
    l_plus1_x = base.l_plus1_operators[patch](x_dict[patch])
    b_x = base.medium_operators[patch](x_dict[patch])
    a_x = (l_plus1_x - b_x) / base.scaling[patch]

    # n_domains
    base2 = HelmholtzBase(n=n, n_domains=n_domains, wrap_correction='wrap_corr')
    x2 = pad_boundaries_torch(x, (0, 0, 0), tuple(np.array(base2.s.shape) - np.array(base.s.shape)),
                              mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    x_dict2 = defaultdict(list)
    for patch2 in base2.domains_iterator:
        x_dict2[patch2] = map_domain(x2, restrict, patch2)
    l_plus1_x2 = base2.l_plus1(x_dict2)
    b_x2 = base2.medium(x_dict2)
    a_x2 = 0.
    for patch2 in base2.domains_iterator:
        a_x2_patch = (l_plus1_x2[patch2] - b_x2[patch2]) / base2.scaling[patch2]
        a_x2 += map_domain(a_x2_patch, extend, patch2).cpu()

    if (base.boundary_post != 0).any():
        a_x = a_x[base.crop2roi]
    if (base2.boundary_post != 0).any():
        a_x2 = a_x2[base2.crop2roi]
    a_x = squeeze_(a_x.cpu().numpy())
    a_x2 = squeeze_(a_x2.cpu().numpy())

    rel_err = relative_error(a_x2, a_x)
    mae = max_abs_error(a_x2, a_x)
    print(f'Relative error ({rel_err:.2e})')
    print(f'Max absolute error (Normalized) ({mae:.2e})')
    threshold = 1.e-3
    assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
    assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'


@pytest.mark.parametrize("n_size", param_n)
@pytest.mark.parametrize("n_domains", [2])
def test_precon_iteration(n_size, n_domains):
    iterations = 1000
    n = np.ones(n_size, dtype=np.complex64)
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    # 1 domain problem
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction=None)
    u = torch.rand(*base.s.shape, dtype=torch.complex64)

    _, extend = domain_decomp_operators(base)
    patch = (0, 0, 0)  # 1 domain so only 1 patch
    s_dict = defaultdict(list)
    u_dict = s_dict.copy()
    ut_dict = s_dict.copy()
    s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * base.s.to(base.devices[patch])
    u_dict[patch] = u.to(base.devices[patch])

    for _ in range(iterations):
        t_dict = precon_iteration(base, u_dict, ut_dict, s_dict)
        for patch in base.domains_iterator:
            u_dict[patch] = u_dict[patch] - (base.alpha * t_dict[patch])
    t1 = 0.
    for patch in base.domains_iterator:
        t1_patch = np.sqrt(base.scaling[patch]) * u_dict[patch]
        t1 += map_domain(t1_patch, extend, patch).cpu()

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')
    u2 = pad_boundaries_torch(u, (0, 0, 0), tuple(np.array(base2.s.shape) - np.array(base.s.shape)),
                              mode="constant")
    restrict2, extend2 = domain_decomp_operators(base2)
    s_dict2 = defaultdict(list)
    u_dict2 = s_dict2.copy()
    ut_dict2 = s_dict2.copy()
    for patch2 in base2.domains_iterator:
        s_dict2[patch2] = 1j * np.sqrt(base2.scaling[patch2]) * map_domain(base2.s.to(base2.devices[patch2]), restrict2, patch2)
        u_dict2[patch2] = map_domain(u2.to(base2.devices[patch2]), restrict2, patch2)

    for _ in range(iterations):
        t_dict2 = precon_iteration(base2, u_dict2, ut_dict2, s_dict2)
        for patch2 in base2.domains_iterator:
            u_dict2[patch2] = u_dict2[patch2] - (base2.alpha * t_dict2[patch2])
    t2 = 0.
    for patch2 in base2.domains_iterator:
        t2_patch = np.sqrt(base2.scaling[patch2]) * u_dict2[patch2]
        t2 += map_domain(t2_patch, extend2, patch2).cpu()

    if (base2.boundary_post != 0).any():
        t2 = t2[base2.crop2roi]
    if (base.boundary_post != 0).any():
        t1 = t1[base.crop2roi]
    t1 = squeeze_(t1.cpu().numpy())
    t2 = squeeze_(t2.cpu().numpy())

    rel_err = relative_error(t2, t1)
    mae = max_abs_error(t2, t1)
    print(f'Relative error ({rel_err:.2e})')
    print(f'Max absolute error (Normalized) ({mae:.2e})')
    threshold = 1.e-3
    assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
    assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'
