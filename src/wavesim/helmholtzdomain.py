import numpy as np
import numba
import cupy
from numpy._typing import NDArray

from wavesim.domain import Domain
from wavesim.engine import (
    dispatch,
    NumpyArray,
    Array,
    CupyArray,
    scale,
    block_shape,
    block_enumerate,
    ConstantArray,
    BlockArray,
    as_complex,
    edges,
    matmul,
    copy,
    multiply,
    subtract,
    add,
    fft,
    ifft,
    block_iter,
)
from wavesim.utilities import laplace_kernel_1d
from wavesim.wrapcorrecteddomain import DomainController


# array has methods:
# get: return data as np array
# set: set data from np array or scalar
# mix: replace data by α·self + β·y. Has option for specifying an offset index for array y (which can be smaller than x)
# convolve: fast convolution with specified kernel
# edges: return 6-tuple with edge slices of specified width
# matmul: matrix multiplication with specified matrix along specified axis


class Helmholtz(Domain):
    """
    Helmholtz equation with an inhomogeneous refractive index.
    """

    def __init__(
        self,
        *,
        permittivity: Array,
        pixel_size: float,
        wavelength: float,
        periodic: tuple[bool, bool, bool] = (False, False, False),
        boundary_width: int = 8,
    ):
        super().__init__(
            shape=permittivity.shape,
            pixel_size=pixel_size,
            wavelength=wavelength,
            allocator=permittivity.factory().to_complex(),
            periodic=periodic,
        )
        # todo: more efficient to scale permittivity once, now it is scaled with k02 first and then with self.scale
        scale(self.k02, permittivity, out=permittivity)
        self.shift, v_scat_norm = _center(permittivity)
        self.domains = np.zeros(block_shape(permittivity), dtype=object)
        periodic = np.asarray(self.periodic)
        single_block = np.asarray(self.domains.shape) == 1  # dimensions along which there is only one subdomain
        no_correction = [s == 1 and p for p, s in zip(periodic, self.domains.shape)]
        boundary_widths = np.full((self.ndim, 2), boundary_width)
        boundary_widths[no_correction, :] = 0

        for idx, block in block_enumerate(permittivity):
            self.domains[idx] = _HelmholtzDomainV(
                block,
                pixel_size,
                self.shift.blocks[idx],
                v_scat_norm,
                periodic,
                single_block,
                boundary_widths,
            )

        if isinstance(permittivity, BlockArray):
            block_boundaries = permittivity.boundaries
        else:
            block_boundaries = [] * self.ndim
        self.domain_controller = DomainController(self.domains, periodic, boundary_widths, block_boundaries)
        self.scale = next(self.domains.flat).scale  # take scale factor of first domain (should be same for all domains)

    def medium(self, x: Array, /, *, out: Array):
        self.domain_controller.medium(x, out=out)

    def propagator(self, x: Array, /, *, out: Array):
        self.domain_controller.propagator(x, out=out)

    def inverse_propagator(self, x: Array, /, *, out: Array):
        self.domain_controller.inverse_propagator(x, out=out)


class _HelmholtzDomainV:
    """Subdomain of the Helmholtz equation with a wrapping correction incorporated in the medium operator."""

    def __init__(
        self,
        permittivity: Array,
        pixel_size: float,
        shift: Array,
        v_scat_norm: float,
        periodic: NDArray[bool],
        single_block: np.ndarray,
        boundary_widths,
    ):
        #
        # per domain data:
        #   - kernels
        #   - V_wrap
        #   - B_scat = 1 - V_scat
        permittivity = as_complex(permittivity)
        factory = permittivity.factory()
        ndim = permittivity.ndim
        shape = permittivity.shape

        # parameter validation
        if any(boundary_widths.flat) < 0:
            raise ValueError("boundary_width must be non-negative")
        for d in range(ndim):
            if not single_block[d] or not periodic[d]:  # need correction along this dimension
                if shape[d] < 2 * np.max(boundary_widths[d]):
                    raise ValueError(
                        f"Domain shape {shape} is too small for the given boundary width {boundary_widths[d]}"
                    )
                if any(boundary_widths[d]) == 0:
                    raise ValueError("Cannot have subdomains or non-periodic boundaries if boundary_width is 0")

        # construct the Laplace kernel  (-kx², or finite domain representation) for each axis
        self.kernels = tuple(
            [
                factory(_laplace_kernel(shape[d], periodic[d], pixel_size)).transpose(0, to=d, ndim=ndim)
                for d in range(ndim)
            ]
        )

        # compute the wrapping correction matrix V_wrap and adjust scale
        # for the operator norm of the wrapping correction matrices
        # also compute the number of corrections applied:
        # one transfer correction each axis that has more than one block
        # one wrapping correction for each axis that has more than one block or that is not periodic
        # todo: check!
        # note: we assume all boundary widths are the same or zero for all axes
        correction_count = sum(~single_block) + sum(np.logical_or(~single_block, ~periodic))
        v_wrap = _make_wrap_matrix(pixel_size, np.max(boundary_widths))
        v_wrap_norm = 0.0 if v_wrap.size == 0 else np.linalg.norm(v_wrap, ord=2) * correction_count
        self.scale = -0.95j / (v_scat_norm + v_wrap_norm)  # scale factor to reduce ||V|| to 0.95

        # scale and shift the kernels and V, and compute B_scat
        self.shift = shift  # this is the optimal shift needed to reduce ||V||  (also known as k0²)
        scale(self.scale, self.shift, offset=1.0, out=self.shift)  # shift -> scale * shift + 1
        scale(-self.scale, permittivity, offset=self.shift, out=permittivity)
        self.B_scat = permittivity

        # Scale the kernel components. The sum of these components equals -scale·(kx²+ky²+kz²)+shift
        # where shift = scale·k0² + 1.0
        for d, k in enumerate(self.kernels):
            scale(self.scale, k, offset=self.shift.value if d == 0 else 0.0, out=k)

        # scale V_wrap and pre-compute the transpose
        self.v_wrap = np.empty((ndim, 2), dtype=object)
        self.v_wrap[:, 0] = factory(self.scale * v_wrap)
        self.v_wrap[:, 1] = self.v_wrap[0, 0].transpose((1, 0))

        # allocate storage for wrap buffers (will be set by the DomainController)
        self.edges_out = np.empty((ndim, 2), dtype=object)
        self.transfer_in = np.empty((ndim, 2), dtype=object)
        self.transfer_out = np.empty((ndim, 2), dtype=object)
        self.wrapping_in = np.empty((ndim, 2), dtype=object)
        self.boundary_widths = boundary_widths

    def medium(self, x: Array, /, *, out: Array):
        """Applies the operator 1-V_scat.

        Note:
            Does not apply the wrapping correction. When part of a multi-domain,
            the wrapping correction is applied by the medium() function of the multi-domain object
            and this function should not be called directly.
        """
        # compute edge corrections and copy to neighbors input buffer
        x_edges = edges(x, widths=self.boundary_widths, empty_as_none=True)
        for idx, x_, v_wrap_, out_, t_out_ in block_enumerate(x_edges, self.v_wrap, self.edges_out, self.transfer_out):
            if x_ is not None:
                matmul(v_wrap_, x_, axis=idx[-2], out=out_)
                if t_out_ is not None:
                    copy(out_, out=t_out_)

        # apply the scattering operator
        multiply(x, self.B_scat, out=out)

        # wait until all other domains have computed their edges
        yield True  # not complete yet

        # apply the edge corrections
        o_edges = edges(out, widths=self.boundary_widths, empty_as_none=True)
        for t_in_, w_in_, out_ in block_iter(self.transfer_in, self.wrapping_in, o_edges):
            if t_in_ is not None:
                subtract(out_, t_in_, out=out_)
            if w_in_ is not None:
                add(out_, w_in_, out=out_)

        yield False  # done

    def propagator(self, x: Array, /, *, out: Array):
        """Applies the operator (L+1)^-1 x."""
        if not x.is_full:
            copy(x, out=out)
            x = out

        axes = tuple(range(x.ndim))
        fft(x, axes=axes, out=out)
        _helmholtz_kernel(out, *self.kernels, out=out)
        ifft(out, axes=axes, out=out) 
        yield False  # done

    def inverse_propagator(self, x: Array, out: Array):
        """Applies the operator (L+1) x .

        This operation is not needed for the Wavesim algorithm, but is provided for testing purposes,
        and can be used to evaluate the residue of the solution.
        """
        if not x.is_full:
            copy(x, out=out)
            x = out

        axes = tuple(range(x.ndim))
        fft(x, axes=axes, out=out)
        _inverse_helmholtz_kernel(out, *self.kernels, out=out)
        ifft(out, axes=axes, out=out)
        yield False  # done


def _laplace_kernel(length: int, periodic: bool, pixel_size: float) -> np.ndarray:
    """This function is called by the DomainController to compute the Fourier-space Laplace kernel for each block.

    It is called twice, first with scale = 1.0 to compute the kernel for computing the norm of V_wrap,
    and then with the actual scale that is used for the Helmholtz equation.
    """
    if periodic:
        kernel_1d = -np.fft.fftfreq(length, d=pixel_size / (2.0 * np.pi)) ** 2
    else:
        kernel_1d = laplace_kernel_1d(pixel_size, length)
        kernel_1d = np.fft.fft(kernel_1d).real
        kernel_1d[0] = 0.0  # remove numerical error, if any
    return kernel_1d


@dispatch
def _helmholtz_kernel(
    x: NumpyArray,
    kernel_x: NumpyArray,
    kernel_y: NumpyArray,
    kernel_z: NumpyArray,
    /,
    *,
    out: NumpyArray,
):
    """Applies the operator (L+1)^-1 x.

    Args:
        x: The input array. Must have shape (N_x, N_y, N_z).
        kernel_x: The kernel along the x-axis. Must have shape (N_x, 1, 1).
        kernel_y: The kernel along the y-axis. Must have shape (1, N_y, 1).
        kernel_z: The kernel along the z-axis. Must have shape (1, 1, N_z).
        out: The output array. Must have the same shape as x.
    """
    if (kernel_x.shape, kernel_y.shape, kernel_z.shape) != ((x.shape[0], 1, 1), (1, x.shape[1], 1), (1, 1, x.shape[2])):
        raise ValueError("Kernel sizes do not match the input array size")
    _numba_helmholtz_kernel(x.d, kernel_x.d, kernel_y.d, kernel_z.d, out.d)


@dispatch
def _helmholtz_kernel(
    x: CupyArray, kernel_x: CupyArray, kernel_y: CupyArray, kernel_z: CupyArray, /, *, out: CupyArray
):
    """Applies the operator (L+1)^-1 x.

    Args:
        x: The input array. Must have shape (N_x, N_y, N_z).
        kernel_x: The kernel along the x-axis. Must have shape (N_x, 1, 1).
        kernel_y: The kernel along the y-axis. Must have shape (1, N_y, 1).
        kernel_z: The kernel along the z-axis. Must have shape (1, 1, N_z).
        out: The output array. Must have the same shape as x.
    """
    if (kernel_x.shape, kernel_y.shape, kernel_z.shape) != ((x.shape[0], 1, 1), (1, x.shape[1], 1), (1, 1, x.shape[2])):
        raise ValueError("Kernel sizes do not match the input array size")
    _cupy_helmholtz_kernel(x.d, kernel_x.d, kernel_y.d, kernel_z.d, out.d)


@dispatch
def _inverse_helmholtz_kernel(
    x: NumpyArray,
    kernel_x: NumpyArray,
    kernel_y: NumpyArray,
    kernel_z: NumpyArray,
    /,
    *,
    out: NumpyArray,
):
    """Applies the operator (L+1) x.

    Args:
        x: The input array. Must have shape (N_x, N_y, N_z).
        kernel_x: The kernel along the x-axis. Must have shape (N_x, 1, 1).
        kernel_y: The kernel along the y-axis. Must have shape (1, N_y, 1).
        kernel_z: The kernel along the z-axis. Must have shape (1, 1, N_z).
        out: The output array. Must have the same shape as x.
    """
    if (kernel_x.shape, kernel_y.shape, kernel_z.shape) != ((x.shape[0], 1, 1), (1, x.shape[1], 1), (1, 1, x.shape[2])):
        raise ValueError("Kernel sizes do not match the input array size")
    _numba_inverse_helmholtz_kernel(x.d, kernel_x.d, kernel_y.d, kernel_z.d, out.d)


@dispatch
def _inverse_helmholtz_kernel(
    x: CupyArray, kernel_x: CupyArray, kernel_y: CupyArray, kernel_z: CupyArray, /, *, out: CupyArray
):
    """Applies the operator (L+1) x.

    Args:
        x: The input array. Must have shape (N_x, N_y, N_z).
        kernel_x: The kernel along the x-axis. Must have shape (N_x, 1, 1).
        kernel_y: The kernel along the y-axis. Must have shape (1, N_y, 1).
        kernel_z: The kernel along the z-axis. Must have shape (1, 1, N_z).
        out: The output array. Must have the same shape as x.
    """
    if (kernel_x.shape, kernel_y.shape, kernel_z.shape) != ((x.shape[0], 1, 1), (1, x.shape[1], 1), (1, 1, x.shape[2])):
        raise ValueError("Kernel sizes do not match the input array size")
    _cupy_inverse_helmholtz_kernel(x.d, kernel_x.d, kernel_y.d, kernel_z.d, out.d)


@numba.njit
def _numba_helmholtz_kernel(
    x: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray, kernel_z: np.ndarray, out: np.ndarray
):
    """Applies the operator (L+1)^-1 x."""
    out[:] = x / (kernel_x + kernel_y + kernel_z)


@numba.njit
def _numba_inverse_helmholtz_kernel(
    x: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray, kernel_z: np.ndarray, out: np.ndarray
):
    """Applies the operator (L+1) x."""
    out[:] = x * (kernel_x + kernel_y + kernel_z)


_cupy_helmholtz_kernel = cupy.ElementwiseKernel(
    "T x, T kernel_x, T kernel_y, T kernel_z",
    "T out",
    "out = x / (kernel_x + kernel_y + kernel_z)",
    "helmholtz_kernel",
)

_cupy_inverse_helmholtz_kernel = cupy.ElementwiseKernel(
    "T x, T kernel_x, T kernel_y, T kernel_z",
    "T out",
    "out = x * (kernel_x + kernel_y + kernel_z)",
    "inverse_helmholtz_kernel",
)


def _center(x: Array) -> tuple[BlockArray, float]:
    """Computes the shift to minimize the norm of x

    Returns an Array that represents the shift, and the norm of the shifted
    potential. When ``x`` is an ordinary Array, the returned shift array is just a ConstantArray.
    When ``x`` is a block Array, the returned shift array is a block Array with the same structure as ``x``,
    holding a ConstantArray that represents the shift for each block.
    For block arrays, computes a shift for each block independently.
    """
    if x.gather().imag.min() < 0:  # todo: change to something more efficient
        raise ValueError("Permittivity cannot contain gain")

    r_max = 0.0
    shifted_blocks = np.empty(block_shape(x), dtype=object)
    for idx, block in block_enumerate(x):
        c, r = block.enclosing_circle()
        shifted_blocks[idx] = ConstantArray(c, dtype=block.dtype, shape=block.shape)
        r_max = max(r_max, r)
    return BlockArray(shifted_blocks), r_max


def _make_wrap_matrix(pixel_size: float, n_boundary: int) -> np.ndarray:
    """Make the matrices for the wrapping correction

    Args:
        pixel_size
        n_boundary: number of pixels in the boundary

    Returns:
        wrapping correction matrix of size n_boundary × n_boundary
    """
    # construct the kernel
    kernel_section = laplace_kernel_1d(pixel_size, n_boundary * 6 + 1)

    # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
    wrap_matrix = np.zeros((n_boundary, n_boundary), dtype=np.float64)
    for r in range(n_boundary):
        wrap_matrix[r, :] = kernel_section[n_boundary - r : 2 * n_boundary - r]

    return wrap_matrix
