import pytest
from helmholtzbase import HelmholtzBase
from wavesim.domain import Domain
import torch
from torch import tensor

""" Performs a set of basic consistency checks for the Domain class and the HelmholtzBase multi-domain class. """

device = "cuda:0"
dtype = torch.complex64


def construct_domain(n_size, n_domains, n_boundary):
    """ Construct a domain or multi-domain"""
    n = torch.rand(n_size, dtype=dtype, device=device) + 1.0
    if n_domains is None:  # single domain
        return Domain(refractive_index=n, pixel_size=0.25, periodic=(False, False, True), n_boundary=n_boundary)
    else:
        return HelmholtzBase(refractive_index=n, pixel_size=0.25, periodic=(False, False, True), n_boundary=n_boundary,
                             n_domains=n_domains)


def construct_source(n_size):
    """ Construct a sparse-matrix source with some points at the corners and in the center"""
    locations = tensor([
        [n_size[0] // 2, 0, n_size[0]],
        [n_size[1] // 2, 0, 0],
        [n_size[2] // 2, 0, 0]])

    return torch.sparse_coo_tensor(locations, tensor([1, 1, 1]), n_size, dtype=torch.complex64)


def allclose(a, b):
    if not torch.is_tensor(a):
        a = tensor(a, dtype=b.dtype)
    if not torch.is_tensor(b):
        b = tensor(b, dtype=a.dtype)
    if a.dtype != b.dtype:
        a = a.astype(b.dtype)
    if a.device != b.device:
        a = a.to('cpu')
        b = b.to('cpu')
    a = a.to_dense()
    b = b.to_dense()
    return torch.allclose(a, b)


@pytest.mark.parametrize("n_size", [(128, 100, 93), (50, 49, 1)])
@pytest.mark.parametrize("n_domains", [None, (1, 1, 1), (3, 2, 1)])
def test_domains(n_size: tuple[int, int, int], n_domains: tuple[int, int, int] | None):
    # construct the (multi-) domain operator
    domain = construct_domain(n_size, n_domains, n_boundary=8)

    # construct a random vector for testing operators
    x = torch.randn(n_size, device=device) + 1.0j * torch.randn(n_size, device=device)
    y = torch.randn(n_size, device=device) + 1.0j * torch.randn(n_size, device=device)

    # perform some very basic checks
    # mainly, this tests if the partitioning and composition works correctly
    assert domain.shape == n_size

    domain.set(0, x)
    domain.set(1, y)
    assert x.device == domain.device
    assert allclose(domain.get(0), x)
    assert allclose(domain.get(1), y)

    inp = domain.inner_product(0, 1)
    assert torch.isclose(inp, torch.vdot(x.flatten(), y.flatten()))

    # construct a source and test adding it
    domain.clear(0)
    assert allclose(domain.get(0), 0.0)
    source = construct_source(n_size)
    domain.set_source(source)
    domain.add_source(0)
    domain.add_source(0)
    assert allclose(domain.get(0), 2.0 * source)
#
# def check_l_plus1_inv(n_size, n_domains):
#     """ Check that (L+1)^(-1) (L+1) x = x """
#     n = np.ones(n_size, dtype=np.complex64)
#     base = HelmholtzBase(refractive_index=n, n_domains=n_domains)
#     restrict, extend = domain_decomp_operators(base)
#
#     # function that evaluates (L+1)^(-1) (L+1) x
#     def l_inv_l(x):
#         u_dict = defaultdict(list)
#         for patch in base.domains_iterator:
#             u_dict[patch] = map_domain(x.to(base.devices[patch]), restrict, patch)
#         l_dict = base.l_plus1(u_dict, crop=False)
#         l_dict = base.propagator(l_dict)
#         x_ = 0.
#         for patch in base.domains_iterator:
#             x_ += map_domain(l_dict[patch], extend, patch).cpu()
#         return x_
#
#     x_in = rand(*base.s.shape, dtype=complex64, device=base.devices[(0, 0, 0)])
#     x_out = l_inv_l(x_in)
#
#     if boundary_widths != 0:
#         # crop to n_roi, excluding boundaries
#         crop2roi = tuple([slice(base.boundary_pre[0], -base.boundary_post[0]) for _ in range(base.n_dims)])
#         x_in = x_in[crop2roi]
#         x_out = x_out[crop2roi]
#     x_in = squeeze_(x_in.cpu().numpy())
#     x_out = squeeze_(x_out.cpu().numpy())
#
#     rel_err = relative_error(x_out, x_in)
#     mae = max_abs_error(x_out, x_in)
#     print(f'Relative error ({rel_err:.2e})')
#     print(f'Max absolute error (Normalized) ({mae:.2e})')
#     return rel_err, mae
#
#
# param_n_boundaries = [(236, 0), (236, 10),
#                       ((30, 32), 0), ((30, 32), 10),
#                       ((30, 31, 32), 0), ((30, 31, 32), 5)]
#
#
# def test_1domain_wrap_options(check_l_plus1_inv):
#     """ Check that (L+1)^(-1) (L+1) x = x for 1-domain scenario for all wrapping correction options """
#     rel_err, mae = check_l_plus1_inv
#     threshold = 1.e-3
#     assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
#     assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'
#
#
# @pytest.mark.parametrize("n_size, boundary_widths", param_n_boundaries)
# @pytest.mark.parametrize("n_domains", [2])
# @pytest.mark.parametrize("wrap_correction", ['wrap_corr'])
# def test_ndomains(check_l_plus1_inv):
#     """ Check that (L+1)^(-1) (L+1) x = x when number of domains > 1
#     (for n_domains > 1, wrap_correction = 'wrap_corr' by default)"""
#     rel_err, mae = check_l_plus1_inv
#     threshold = 1.e-3
#     assert rel_err <= threshold, f'Relative error ({rel_err:.2e}) > {threshold:.2e}'
#     assert mae <= threshold, f'Max absolute error (Normalized) ({mae:.2e}) > {threshold:.2e}'
