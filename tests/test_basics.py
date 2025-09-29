import numpy as np
import pytest
from matplotlib import pyplot as plt

from wavesim.engine import (
    Array,
    NumpyArray,
    copy,
    empty_like,
    fft,
    as_complex,
    SparseArray,
    BlockArray,
    add,
    subtract,
    clone,
)
from wavesim.helmholtzdomain import Helmholtz
from wavesim.utilities import laplace_kernel_1d, diff_kernel_1d
from . import all_close, random_vector, random_permittivity

""" Performs a set of basic consistency checks for the Domain class and the HelmholtzBase multi-domain class. """


def construct_domain(permittivity, n_domains, periodic, boundary_width=8):
    """Construct a domain or multi-domain"""
    if n_domains is not None:
        permittivity = BlockArray(permittivity, n_domains=n_domains)
    return Helmholtz(
        permittivity=permittivity, periodic=periodic, 
        boundary_width=boundary_width, pixel_size=0.25, wavelength=1.0
    )


@pytest.mark.parametrize("shape", [(128, 100, 93), (50, 49, 1)])
@pytest.mark.parametrize("n_domains", [None, (1, 1, 1), (3, 2, 1)])
def test_basics(shape: tuple[int, int, int], n_domains: tuple[int, int, int] | None):
    """Tests the basic functionality of the Domain and MultiDomain classes

    Tests constructing a domain, subdividing data over subdomains,
    concatenating data from the subdomains, adding sparse sources,
    and computing the inner product.
    """

    # construct the (multi-) domain operator
    permittivity = random_permittivity(shape)
    domain = construct_domain(permittivity, n_domains, periodic=(False, True, True))
    assert domain.shape == shape
    assert domain.pixel_size == 0.25
    assert domain.wavelength == 1.0
    assert domain.periodic == (False, True, True)

    # test coordinates
    for dim in range(3):
        coordinates = domain.coordinates(dim)
        assert coordinates.shape[dim] == shape[dim]
        assert coordinates.size == shape[dim]

        coordinates_f = domain.coordinates_f(dim)
        assert coordinates_f.shape == coordinates.shape
        assert coordinates_f[0, 0, 0] == 0

        if shape[dim] > 1:
            assert all_close(coordinates.flatten()[1] - coordinates.flatten()[0], domain.pixel_size)
            assert all_close(
                coordinates_f.flatten()[1] - coordinates_f.flatten()[0],
                2.0 * np.pi / (shape[dim] * domain.pixel_size),
            )

    # perform some very basic checks
    # mainly, this tests if the partitioning and composition works correctly
    x = np.random.uniform(size=shape)
    ax = domain.allocate(x)
    assert all_close(x, ax)


@pytest.mark.parametrize("pixel_size", [1.0, 0.25])
@pytest.mark.parametrize("size", [1024, 1025])
def test_laplace_kernel(pixel_size, size):
    kernel = as_complex(NumpyArray(laplace_kernel_1d(pixel_size, size)))
    fft(kernel, axes=(0,), out=kernel)
    correct_kernel = -((2 * np.pi * np.fft.fftfreq(size, pixel_size)) ** 2)
    # a small difference is possible because the correct kernel assumes periodic boundaries, so it is not 100% correct for the non-periodic case.
    assert all_close(kernel[1:], correct_kernel[1:], rtol=0.01, atol=1e-4)


@pytest.mark.parametrize("pixel_size", [1.0, 0.25])
@pytest.mark.parametrize("size", [1024, 1025])
def test_diff_kernel(pixel_size, size):
    size = 1025
    kernel = as_complex(NumpyArray(diff_kernel_1d(pixel_size, size)))
    fft(kernel, axes=(0,), out=kernel)
    correct_kernel = 1j * 2 * np.pi * np.fft.fftfreq(size, pixel_size)
    # disregard the highest frequencies (center few elements)
    selection = slice(1, size // 2 - 10)
    assert all_close(kernel[selection], correct_kernel[selection], rtol=0.01)


@pytest.mark.parametrize("n_size", [(128, 100, 93), (50, 49, 1)])
@pytest.mark.parametrize("n_domains", [None, (1, 1, 1), (3, 2, 1)])
def test_propagator(n_size: tuple[int, int, int], n_domains: tuple[int, int, int] | None):
    """Tests the forward and inverse propagator

    The wavesim algorithm only needs the propagator (L+1)^(-1) to be implemented.
    For testing, and for evaluating the final residue, the Domain and MultiDomain classes
    also implement the 'inverse propagator L+1', which is basically the homogeneous part of the forward operator A.

    This test checks that the forward and inverse propagator are consistent, namely (L+1)^(-1) (L+1) x = x.

    Note that the propagator is domain-local, so the wrapping correction and domain
    transfer functions are not tested here.
    """

    # construct the (multi-) domain operator
    permittivity = random_permittivity(n_size)
    domain = construct_domain(permittivity, n_domains, periodic=(False, True, True))

    # assert that (L+1) (L+1)^-1 x = x
    x_orig = domain.allocate(random_vector(n_size))
    x = empty_like(x_orig)
    ulptol = 3 * np.sqrt(np.log2(x.size))  # tolerance for fft
    copy(x_orig, out=x)
    domain.propagator(x, out=x)
    domain.inverse_propagator(x, out=x)
    assert all_close(x, x_orig, ulptol=ulptol)

    # also assert that (L+1)^-1 (L+1) x = x, use different slots for input and output
    y = domain.allocate()
    x_orig = domain.allocate(x_orig.gather())
    domain.inverse_propagator(x_orig, out=y)
    domain.propagator(y, out=y)
    assert all_close(y, x_orig, ulptol=ulptol)

    # for the non-decomposed case, test if the propagator gives the correct value
    if n_domains is None:
        n_size = np.asarray(n_size)
        # choose |k| <  Nyquist, make sure k is at exact grid point in Fourier space
        for k_relative in ((0, 0, 0), (0.2, -0.15, 0.4), (0, 0, 0.5)):
            k = 2 * np.pi * np.round(n_size * k_relative) / n_size  # in 1/pixels
            k[n_size == 1] = 0.0
            plane_wave = np.exp(
                1j
                * (
                    k[0] * np.arange(n_size[0]).reshape(-1, 1, 1)
                    + k[1] * np.arange(n_size[1]).reshape(1, -1, 1)
                    + k[2] * np.arange(n_size[2]).reshape(1, 1, -1)
                )
            )
            x = domain.allocate(plane_wave)
            domain.inverse_propagator(x, out=x)
            result = x
            laplace_kernel = (
                -(k[0] ** 2 + k[1] ** 2 + k[2] ** 2) / domain.pixel_size**2
            )  # -∇² [negative of laplace kernel]
            correct_result = (
                domain.scale * laplace_kernel + domain.shift.blocks[0, 0, 0].value
            ) * plane_wave  # L+1 =  scale·(-∇²) + 1.
            # note: the result is not exactly the same because wavesim is using the real-space kernel, and we compare to
            # the Fourier-space kernel
            assert all_close(result, correct_result, rtol=0.01)


def test_basic_wrapping():
    """Simple test if the wrapping correction is applied at the correct position.

    Constructs a 1-D domain and splits it in two. A source is placed at the right edge of the left domain.
    This test does not check if the values of the correction are computed correctly
    """

    # Construct the source and the domain
    shape = (10, 1, 1)
    n_domains = (2, 1, 1)
    n_boundary = 2
    source = SparseArray.point(at=(shape[0] // 2 - 1, 0, 0), shape=shape)
    permittivity = BlockArray(1.0, shape=shape, n_domains=n_domains, factories=NumpyArray)
    domain = Helmholtz(
        permittivity=permittivity,
        pixel_size=0.25,
        wavelength=1.0,
        boundary_width=n_boundary,
        periodic=(False, True, True), 
    )

    # Add the source to a zero-filled array, check if the array is split  correctly and the source was added correctly
    x = domain.allocate(0)
    assert isinstance(x, BlockArray)
    assert x.shape == shape

    add(source, x, out=x)
    left = x.blocks[0, 0, 0].gather()
    right = x.blocks[1, 0, 0].gather()
    total = x.gather()

    assert all_close(np.concat([left, right]), total)
    assert np.all(right == 0.0)
    assert np.all(left[:-2] == 0.0)
    assert left[-1] == 1.0

    # Apply the wrapping correction and check if the source is wrapped around
    y = domain.allocate()
    domain.medium(x, out=y)

    # Check the incoming and outgoing edge transfers
    # For the left domain, we have one incoming transfer from the right and wrapping correction along axis 0
    # For the right domain, we have one outgoing transfer to the left and wrapping correction along axis 0
    left_transfer = domain.domains[0, 0, 0].transfer_in
    right_transfer = domain.domains[1, 0, 0].transfer_in
    left_wrap = domain.domains[0, 0, 0].wrapping_in
    right_wrap = domain.domains[1, 0, 0].wrapping_in

    # along dimensions 1 and 2 there should be no transfer or wrapping
    for d in range(1, 3):
        for side in range(2):
            assert left_transfer[d, side] is None
            assert right_transfer[d, side] is None
            assert left_wrap[d, side] is None
            assert right_wrap[d, side] is None

    # Check if the edges are excluded correctly.
    # We expect the left domain to have a transfer correction only at the right edge
    # and the right domain to have a transfer correction only at the left edge
    assert left_transfer[0, 0] is None  # transfer
    assert right_transfer[0, 1] is None  # transfer

    # The transfer term from right to left should be 0.0 since the right subdomain is empty
    # also, the wrapping corrections within the right domain should be 0.0
    assert all_close(left_transfer[0, 1], 0.0)  # transfer to the left from the right domain
    assert all_close(right_wrap[0, 0], 0.0)
    assert all_close(right_wrap[0, 1], 0.0)
    # the source is on the rhs of the domain, so we expect to see wrapping to the lhs
    # and not to the rhs
    assert not all_close(left_wrap[0, 0], 0.0, plot=False)
    assert all_close(left_wrap[0, 1], 0.0)

    # after applying the correction, the first n_boundary elements
    # of the left domain should be non-zero (wrapping correction)
    # and the first n_boundary elements of the right domain should be non-zero
    total2 = y.gather()
    assert all_close(total2.real[0:n_boundary], -total2.real[shape[0] // 2 : shape[0] // 2 + n_boundary])


def test_wrapped_propagator():
    """Tests the inverse propagator L+1 with wrapping corrections

    This test compares the situation of a single large domain to that of a multi-domain.
    If the wrapping and transfer corrections are implemented correctly, the results should be the same
    up to the difference in scaling factor.
    """
    # shape = (128, 100, 93, 1)
    shape = (3 * 32 * 1024, 1, 1)
    boundary_width = 16
    permittivity = random_permittivity(shape)
    domain_single = construct_domain(
        clone(permittivity), n_domains=None, periodic=(True, True, True), boundary_width=boundary_width
    )
    domain_multi = construct_domain(
        clone(permittivity), n_domains=(3, 1, 1), periodic=(True, True, True), boundary_width=boundary_width
    )
    y = SparseArray.point(at=(0, 0, 0), shape=shape, dtype=np.float32)

    x = [None, None]
    for i, domain in enumerate([domain_single, domain_multi]):
        # evaluate L+1-B = L + Vscat + wrapping correction for the multi-domain,
        # and L+1-B = L + Vscat for the full domain
        # Note that we need to compensate for scaling squared,
        # because scaling affects both the source and operators L and B
        u = domain.allocate(0.0)
        add(u, y, out=u)
        L1 = domain.allocate()
        B = domain.allocate()
        domain.inverse_propagator(u, out=L1)  # (L+1) y
        domain.medium(u, out=B)  # (1-V) y
        subtract(L1, B, out=L1)  # (L+V) y
        x[i] = L1.gather() / domain.scale

    boundaries = np.asarray(domain_multi.allocator.arguments["boundaries"][0])
    # plt.plot(x[0].real.squeeze(), label="single")
    # plt.plot(x[1].real.squeeze(), label="multi")
    # plt.plot(x[0].imag.squeeze(), label="single (imag)")
    # plt.plot(x[1].imag.squeeze(), label="multi (imag)")
    # plt.plot(boundaries, np.zeros_like(boundaries), "o", label="boundary")
    # plt.legend()

    # error at the first non-compensated point should be the maximum error
    pos = boundaries[0] - boundary_width - 1
    atol = np.abs((x[1] - x[0])[pos, 0, 0]) * 1.01
    print(pos)
    print(atol)
    assert all_close(x[0], x[1], atol=atol)
