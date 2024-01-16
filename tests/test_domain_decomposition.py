import pytest
import numpy as np
from collections import defaultdict
from PIL.Image import open, BILINEAR, fromarray  # needed for 2D tests

from helmholtzbase import HelmholtzBase
from anysim import domain_decomp_operators, map_domain
from utilities import pad_boundaries, relative_error

@pytest.mark.parametrize("n", 
                         [np.ones(256), 
                          np.ones((256, 256)), 
                          np.ones((64, 52, 58))])
@pytest.mark.parametrize("n_domains", [2, 3])
def test_precon_iteration(n, n_domains):
    source = np.zeros_like(n, dtype=np.complex64)
    source[0] = 1.

    # 1 domain problem
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction='wrap_corr')
    u = (np.random.rand(*base.s.shape) + 1j*np.random.rand(*base.s.shape)).astype(np.complex64)

    restrict, extend = domain_decomp_operators(base)
    patch = (0, 0, 0)  # 1 domain so only 1 patch
    s_dict = defaultdict(list)
    u_dict = defaultdict(list)
    ut_dict = defaultdict(list)
    s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * base.s
    u_dict[patch] = u.copy()

    t_dict = base.medium(u_dict, s_dict)
    t_dict = base.transfer(u_dict, t_dict)
    t_dict = base.propagator(t_dict)
    for patch in base.domains_iterator:
        ut_dict[patch] = u_dict[patch] - t_dict[patch]
    t_dict = base.medium(ut_dict)
    t_dict = base.transfer(ut_dict, t_dict)
    t1 = 0.
    for patch in base.domains_iterator:
        t1_patch = np.sqrt(base.scaling[patch]) * t_dict[patch]
        t1 += map_domain(t1_patch, extend, patch)

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')
    u2 = pad_boundaries(u, (0,0,0), np.array(base2.s.shape)-np.array(base.s.shape), mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    s_dict2 = defaultdict(list)
    u_dict2 = defaultdict(list)
    ut_dict2 = defaultdict(list)
    for patch2 in base2.domains_iterator:
        s_dict2[patch2] = 1j * np.sqrt(base2.scaling[patch2]) * map_domain(base2.s, restrict, patch2)
        u_dict2[patch2] = map_domain(u2, restrict, patch2)

    t_dict2 = base2.medium(u_dict2, s_dict2)
    t_dict2 = base2.transfer(u_dict2, t_dict2)
    t_dict2 = base2.propagator(t_dict2)
    for patch2 in base2.domains_iterator:
        ut_dict2[patch2] = u_dict2[patch2] - t_dict2[patch2]
    t_dict2 = base2.medium(ut_dict2)
    t_dict2 = base2.transfer(ut_dict2, t_dict2)
    t2 = 0.
    for patch2 in base2.domains_iterator:
        t2_patch = np.sqrt(base2.scaling[patch2]) * t_dict2[patch2]
        t2 += map_domain(t2_patch, extend, patch2)

    if (base.boundary_post != 0).any():
        t1 = t1[base.crop2roi]
    if (base2.boundary_post != 0).any():
        t2 = t2[base2.crop2roi]
    t1 = np.squeeze(t1)
    t2 = np.squeeze(t2)
    rel_err = relative_error(t2, t1)

    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3


@pytest.mark.parametrize("n_domains", [2, 3, 4, 5, 6, 7])
def test_iteration_2DLowContrast(n_domains):
    oversampling = 0.25
    im = np.asarray(open('logo_structure_vector.png')) / 255
    n_water = 1.33
    n_fat = 1.46
    n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water
    n_roi = int(oversampling * n_im.shape[0])
    n = np.asarray(fromarray(n_im).resize((n_roi, n_roi), BILINEAR))
    source = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), BILINEAR))
    wavelength = 0.532
    ppw = 3 * abs(n_fat)

    base = HelmholtzBase(n=n, source=source, wavelength=wavelength, ppw=ppw, 
                         n_domains=1, wrap_correction='wrap_corr')
    u = (np.random.rand(*base.s.shape) + 1j*np.random.rand(*base.s.shape)).astype(np.complex64)

    restrict, extend = domain_decomp_operators(base)
    patch = (0, 0, 0)
    s_dict = defaultdict(list)
    u_dict = defaultdict(list)
    ut_dict = defaultdict(list)
    s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * base.s
    u_dict[patch] = u.copy()

    t_dict = base.medium(u_dict, s_dict)
    t_dict = base.transfer(u_dict, t_dict)
    t_dict = base.propagator(t_dict)
    for patch in base.domains_iterator:
        ut_dict[patch] = u_dict[patch] - t_dict[patch]
    t_dict = base.medium(ut_dict)
    t_dict = base.transfer(ut_dict, t_dict)
    t1 = 0.
    for patch in base.domains_iterator:
        t1_patch = np.sqrt(base.scaling[patch]) * t_dict[patch]
        t1 += map_domain(t1_patch, extend, patch)

    base2 = HelmholtzBase(n=n, source=source, wavelength=wavelength, ppw=ppw, 
                         n_domains=n_domains, wrap_correction='wrap_corr')
    u2 = pad_boundaries(u, (0,0,0), np.array(base2.s.shape)-np.array(base.s.shape), mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    s_dict2 = defaultdict(list)
    u_dict2 = defaultdict(list)
    ut_dict2 = defaultdict(list)
    for patch2 in base2.domains_iterator:
        s_dict2[patch2] = 1j * np.sqrt(base2.scaling[patch2]) * map_domain(base2.s, restrict, patch2)
        u_dict2[patch2] = map_domain(u2, restrict, patch2)

    t_dict2 = base2.medium(u_dict2, s_dict2)
    t_dict2 = base2.transfer(u_dict2, t_dict2)
    t_dict2 = base2.propagator(t_dict2)
    for patch2 in base2.domains_iterator:
        ut_dict2[patch2] = u_dict2[patch2] - t_dict2[patch2]
    t_dict2 = base2.medium(ut_dict2)
    t_dict2 = base2.transfer(ut_dict2, t_dict2)
    t2 = 0.
    for patch2 in base2.domains_iterator:
        t2_patch = np.sqrt(base2.scaling[patch2]) * t_dict2[patch2]
        t2 += map_domain(t2_patch, extend, patch2)

    if (base.boundary_post != 0).any():
        t1 = t1[base.crop2roi]
    if (base2.boundary_post != 0).any():
        t2 = t2[base2.crop2roi]
    t1 = np.squeeze(t1)
    t2 = np.squeeze(t2)
    rel_err = relative_error(t2, t1)

    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3


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
    x = (np.random.rand(*base.s.shape) + 1j*np.random.rand(*base.s.shape)).astype(np.complex64)

    patch = (0, 0, 0)  # 1 domain so only 1 patch
    x_dict = defaultdict(list)
    x_dict[patch] = x.copy()
    l_plus1_x = base.l_plus1_operators[patch](x_dict[patch])
    b_x = base.medium_operators[patch](x_dict[patch])
    a_x = (l_plus1_x - b_x)/base.scaling[patch]

    # n_domains
    base2 = HelmholtzBase(n=n, source=source, n_domains=n_domains, wrap_correction='wrap_corr')

    x2 = pad_boundaries(x, (0,0,0), np.array(base2.s.shape)-np.array(base.s.shape), mode="constant")
    restrict, extend = domain_decomp_operators(base2)
    x_dict2 = defaultdict(list)
    l_plus1_x2 = defaultdict(list)
    b_x2 = defaultdict(list)
    z = np.zeros_like(x2, dtype=np.complex64)
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
    a_x = np.squeeze(a_x)
    a_x2 = np.squeeze(a_x2)
    rel_err = relative_error(a_x2, a_x)

    print('Relative error: {:.2e}'.format(rel_err))
    assert rel_err <= 1.e-3
