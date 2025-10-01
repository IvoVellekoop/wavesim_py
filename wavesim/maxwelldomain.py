import numpy as np
import numba
import cupy

from wavesim.engine import (
    dispatch,
    NumpyArray,
    Array,
    CupyArray,
    as_complex,
    scale,
    new_like,
    multiply,
    fft,
    copy,
    ifft,
)
from .domain import Domain


class Maxwell(Domain):
    """
    Time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials.

    This solves systems of the form :math:`\nabla \times \nabla \times E + k^2(r) E = -S`

    This basic form has no special treatment for boundaries; these are treated as periodic in all directions.
    """

    def __init__(
        self,
        *,
        permittivity: Array,
        pixel_size: float,
        periodic: tuple[bool, ...] = (True, True, True),
        wavelength: float,
        center_radius_override: (complex, float) = None,
    ):
        """ """

        # construct allocator for simulation vectors with 3 vector elements.
        # We use the first index for indexing the vector components (0=x, 1=y, 2=z)
        # This approach (as opposed to using the last index) optimizes memory access coalescing
        #
        vector_shape = (3, *permittivity.shape)
        allocator = permittivity.factory().to_complex()
        allocator.arguments["shape"] = vector_shape
        super().__init__(
            shape=permittivity.shape,
            pixel_size=pixel_size,
            wavelength=wavelength,
            allocator=allocator,
            periodic=periodic,
        )

        # Determine optimum shift to reduce ||V|| and scale so that ||V|| = 0.95
        permittivity = as_complex(permittivity)
        if center_radius_override is None:
            center, radius = permittivity.enclosing_circle(enforce_gain_free=True)
        else:
            center, radius = center_radius_override
        self.scale = -0.95j / radius
        self.shift = self.scale * center + 1.0

        # Apply shift and scale. Note that the permittivity needs to be scaled with k0² to get the potential
        scale(-self.scale, permittivity, offset=self.shift, out=permittivity)  # 1-V_scat
        self.B_scat = permittivity.transpose(axes=(np.newaxis, 0, 1, 2))  # add singleton dimension for broadcasting
        self.scale = self.scale / self.k02  # we skipped multiplying the permittivity by k0²

        # Construct the propagator kernels, which are just the k-space coordinates (corrected for the finite domain?)
        kernels = []
        for d in range(self.ndim):
            kernel = np.sqrt(1.0j * self.scale) * self.coordinates_f(d)
            kernels.append(new_like(permittivity, kernel))
        self.kernels = tuple(kernels)

    def propagator(self, x: Array, /, *, out: Array):
        """Applies the operator (L+1)^-1 x.

        Args:
            x: The input array. Must have shape (N_x, N_y, N_z).
            out: The output array. Must have the same shape as x.
        """
        """Applies the operator (L+1)^-1 x."""
        if not x.is_full:
            copy(x, out=out)
            x = out

        axes = tuple(range(1, self.ndim + 1))
        fft(x, axes=axes, out=out)
        maxwell_kernel(x, *self.kernels, self.shift, out=out)
        ifft(out, axes=axes, out=out)

    def inverse_propagator(self, x: Array, /, *, out: Array):
        if not x.is_full:
            copy(x, out=out)
            x = out

        axes = tuple(range(1, self.ndim + 1))
        fft(x, axes=axes, out=out)
        inverse_maxwell_kernel(x, *self.kernels, self.shift, out=out)
        ifft(out, axes=axes, out=out)

    def medium(self, x: Array, /, *, out: Array):
        """Applies the operator B=1-V.

        Args:
            x: The input array. Must have shape (N_x, N_y, N_z).
            out: The output array. Must have the same shape as x.
        """
        multiply(x, self.B_scat, out=out)


@dispatch
def maxwell_kernel(
    x: NumpyArray, k_x: NumpyArray, k_y: NumpyArray, k_z: NumpyArray, shift: complex, /, *, out: NumpyArray
):
    """Applies the operator (L+1)^-1 x.

    Args:
        x: The input array. Must have shape (N_x, N_y, N_z).
        k_x: The k-space coordinate along the x-axis. Must have shape (N_x, 1, 1). Includes scaling
        k_y: The k-space coordinate along the y-axis. Must have shape (1, N_y, 1). Includes scaling
        k_z: The k-space coordinate along the z-axis. Must have shape (1, 1, N_z). Includes scaling
        shift: The shift term, includes both scaled k0^2 and +1.
        out: The output array. Must have the same shape as x.
    """
    _numba_maxwell_kernel(x.d, k_x.d, k_y.d, k_z.d, shift, 1.0j / shift, out.d)


@dispatch
def maxwell_kernel(x: CupyArray, k_x: CupyArray, k_y: CupyArray, k_z: CupyArray, shift: complex, /, *, out: CupyArray):
    _cupy_maxwell_kernel(x.d[0], x.d[1], x.d[2], k_x.d, k_y.d, k_z.d, shift, 1.0j / shift, out.d[0], out.d[1], out.d[2])


@dispatch
def inverse_maxwell_kernel(
    x: NumpyArray, k_x: NumpyArray, k_y: NumpyArray, k_z: NumpyArray, shift: complex, /, *, out: NumpyArray
):
    """Applies the operator (L+1) x.

    Args:
        x: The input array. Must have shape (N_x, N_y, N_z).
        k_x: The scaled Fourier-space coordinates along the x-axis. Must have shape (N_x, 1, 1).
        k_y: The scaled Fourier-space coordinates along the y-axis. Must have shape (1, N_y, 1).
        k_z: The scaled Fourier-space coordinates along the z-axis. Must have shape (1, 1, N_z).
        shift: The shift term, includes both scaled k0^2 and +1.
        out: The output array. Must have the same shape as x.
    """
    _numba_inverse_maxwell_kernel(x.d, k_x.d, k_y.d, k_z.d, shift, out.d)


@dispatch
def inverse_maxwell_kernel(
    x: CupyArray, k_x: CupyArray, k_y: CupyArray, k_z: CupyArray, shift: complex, /, *, out: CupyArray
):
    _cupy_inverse_maxwell_kernel(x.d[0], x.d[1], x.d[2], k_x.d, k_y.d, k_z.d, shift, out.d[0], out.d[1], out.d[2])


@numba.njit
def _numba_maxwell_kernel(
    x: np.ndarray,
    k_x: np.ndarray,
    k_y: np.ndarray,
    k_z: np.ndarray,
    shift: complex,
    ii_shift: complex,
    out: np.ndarray,
):
    """Applies the operator (L+1)⁻¹ E, which is the dyadic Green's function for vector field propagation
    with the damping term included.

    See `_cupy_maxwell_kernel` for the explanation of the formula.
    """
    k2s = 1.0j * (k_x * k_x + k_y * k_y + k_z * k_z) + shift
    g = 1.0 / k2s
    divergence = ii_shift * (k_x * x[0] + k_y * x[1] + k_z * x[2])  # ikᵀ/shift
    out[0] = g * (x[0] + k_x * divergence)
    out[1] = g * (x[1] + k_y * divergence)
    out[2] = g * (x[2] + k_z * divergence)


@cupy.fuse
def _cupy_maxwell_kernel(x_x, x_y, x_z, k_x, k_y, k_z, shift, ii_shift, out_x, out_y, out_z):
    """Applies the operator (L+1)⁻¹ E, which is the dyadic Green's function for vector field propagation
    with the damping term included.

    In this case, (L+1)E = -scale·∇×∇×E + shift·E = scale·[∇²E - ∇(∇⋅E)] + shift·E
    In Fourier space, this is L·E = -scale·[(kᵀk) E - k (kᵀE)] + shift·E
    For performance, k is pre-multiplied by sqrt(i self.scale)
    so that L·E = i(kᵀk)E - i(k kᵀ)·E + shift·E =

    We invert this 3×3 matrix locally for each k.
    We start by inverting the diagonal element: g=[i(kᵀk)+shift]⁻¹ and
    use the Sherman-Morrison formula to compute the correction for the -i(k kᵀ) term:

    (L+1)⁻¹ = g - g (-ik kᵀ) g / (1 + (-ikᵀ g k))
            = g + ig kkᵀ g / (1 - ikᵀk g)
            = g + ig kkᵀ / (g⁻¹ - ikᵀk)
            = g + ig kkᵀ / (i(kᵀk)+shift-i(kᵀk))
      = g (1 + [1.0j/ shift] kkᵀ)

    Note: the complex reciprocal may be expensive to compute.
    If this really is a bottleneck, and the real part of shift is always 1.0,
    the computation may be simplified slightly.

    Args:
        x_x: The x-component of the input vector field. Must have shape (N_x, N_y, N_z).
        x_y: The y-component of the input vector field. Must have shape (N_x, N_y, N_z).
        x_z: The z-component of the input vector field. Must have shape (N_x, N_y, N_z).
        k_x: The scaled Fourier-space coordinates along the x-axis. Must have shape (N_x, 1, 1).
        k_y: The scaled Fourier-space coordinates along the y-axis. Must have shape (1, N_y, 1).
        k_z: The scaled Fourier-space coordinates along the z-axis. Must have shape (1, 1, N_z).
        shift: The shift term, includes both scaled k0^2 and +1.
        ii_shift: Precomputed 1.0j / shift.
    """
    k2s = 1.0j * (k_x * k_x + k_y * k_y + k_z * k_z) + shift  # i(kᵀk)+shift
    g = 1.0 / k2s  # vector wave equation Green's function g=[i(kᵀk)+shift]⁻¹
    divergence = ii_shift * (k_x * x_x + k_y * x_y + k_z * x_z)  # [1.0j/ shift]kᵀ·E
    out_x[:] = g * (x_x + k_x * divergence)
    out_y[:] = g * (x_y + k_y * divergence)
    out_z[:] = g * (x_z + k_z * divergence)


@numba.njit
def _numba_inverse_maxwell_kernel(
    x: np.ndarray,
    k_x: np.ndarray,
    k_y: np.ndarray,
    k_z: np.ndarray,
    shift: complex,
    out: np.ndarray,
):
    """Applies the operator (L+1) x, which is the dyadic Green's function for vector field propagation
    with the damping term included.

    In this case, (L+1)E = -scale·∇×∇×E + shift·E = scale·[∇²E - ∇(∇⋅E)] + shift·E
    In Fourier space, this is L·E = -scale·[|k|²E - k k⋅E] + shift·E
    since k is pre-multiplied by sqrt(i self.scale), we can simply use
    a scale factor of -i (to compensate for the i in the sqrt term).
    Therefore, we have L·E = i|k|²E - ik k·E + shift·E = [i|k|² - ikkᵀ + shift]·E
    """
    k2s = 1.0j * (k_x * k_x + k_y * k_y + k_z * k_z) + shift
    divergence = -1.0j * (k_x * x[0] + k_y * x[1] + k_z * x[2])  # -ikᵀ
    out[0] = k2s * x[0] + k_x * divergence
    out[1] = k2s * x[1] + k_y * divergence
    out[2] = k2s * x[2] + k_z * divergence


@cupy.fuse
def _cupy_inverse_maxwell_kernel(x_x, x_y, x_z, k_x, k_y, k_z, shift, inv_shift, out_x, out_y, out_z):
    k2s = 1.0j * (k_x * k_x + k_y * k_y + k_z * k_z) + shift
    divergence = -1.0j * (k_x * x_x + k_y * x_y + k_z * x_z)  # -ikᵀ
    out_x[:] = k2s * x_x + k_x * divergence
    out_y[:] = k2s * x_y + k_y * divergence
    out_z[:] = k2s * x_z + k_z * divergence
