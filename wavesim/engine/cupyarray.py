import cupy as cp
from cupy import cuda
from typing import Sequence
import numpy as np

from .array import array_like, scalar, Array, dispatch
from .index_utils import shape_like, pos_to_slices


class CupyArray(Array):
    """Numpy-implementation of the array interface.

    This implementation is not shared, it is only used internally by the numpy engine.
    """

    def __init__(
        self,
        data: array_like | cp.ndarray | Array | None,
        *,
        device: cuda.Device | int | None = None,
        shape: shape_like | None = None,
        copy: bool = False,
        dtype=None,
    ):
        """Constructs a new CupyArray.
        Args:
            data: numpy array to copy into the new array
            device: cuda device (from cupy) or integer enumeration of the device to place the array on.
                If None, the currently active device is used.
            shape: shape of the array, only used when data is a scalar, ignored otherwise.
            copy:
                If False (default) and the data is a ndarray of the correct type, the data array is stored directly.
                if True, the data is copied into the new array. This value is ignored for scalars.
        """
        if isinstance(data, CupyArray):
            data = data.d
        with cp.cuda.Device(device) as device:
            if data is None:
                if shape is None:
                    raise ValueError("Shape must be specified when data is a scalar")
                self.d = cp.empty(shape, dtype=dtype)
            elif np.isscalar(data):
                if shape is None:
                    raise ValueError("Shape must be specified when data is a scalar")
                self.d = cp.full(shape, data, dtype=dtype)
            elif isinstance(data, cp.ndarray):
                if data.device == device:
                    self.d = data.copy() if copy else data
                else:
                    self.d = cp.empty_like(data)
                    cp.copyto(self.d, data)
            elif isinstance(data, Array):
                data = data.gather()
                self.d = cp.array(data, dtype=dtype) if copy else cp.asarray(data, dtype=dtype)
            else:
                self.d = cp.array(data, dtype=dtype) if copy else cp.asarray(data, dtype=dtype)

        super().__init__(shape=self.d.shape, dtype=self.d.dtype)

    def _slice(self, start: np.array, stop: np.array) -> "CupyArray":
        return CupyArray(self.d[*pos_to_slices(start, stop)])

    def gather(self):
        return self.d.get()

    @property
    def is_full(self):
        return True

    def transpose(
        self, axes: int | Sequence[int | None], *, to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> Array:

        from_axes, new_axes, shape = self._parse_transpose(axes, to, ndim)
        d = cp.expand_dims(self.d.transpose(from_axes), new_axes)
        return CupyArray(d, shape=shape)

    def enclosing_circle(self, enforce_gain_free=False) -> tuple[complex, float]:
        # we could convert this to a kernel to avoid
        # memory allocation in the last step.
        # however, since this is only done once, before
        # other data is allocated,
        # it is probably not worth the effort.
        r_min = self.d.real.min()
        r_max = self.d.real.max()
        i_min = self.d.imag.min()
        i_max = self.d.imag.max()
        if enforce_gain_free and i_min < 0:
            raise ValueError("Imaginary part of the array musts be non-negative.")

        c = (r_min + r_max) / 2 + 1j * (i_min + i_max) / 2
        r = abs(self.d - c).max()
        return complex(c), float(r)


# Kernels for the cupy engine.
_cupy_mix = cp.ElementwiseKernel(
    "T alpha, T a, T beta, T b",
    "T out",
    "out = alpha * a + beta * b",
    "mix",
    no_return=True,
)

_cupy_lerp = cp.ElementwiseKernel(
    "T a, T b, T weight",
    "T out",
    "out = a + weight * (b - a)",
    "lerp",
    no_return=True,
)

@dispatch
def divide(a: CupyArray | scalar, b: CupyArray | scalar, /, *, out: CupyArray):
    cp.divide(_get_value(a, out.dtype), _get_value(b, out.dtype), out=out.d)


@dispatch
def add(a: CupyArray | scalar, b: CupyArray | scalar, /, *, out: CupyArray):
    cp.add(_get_value(a, out.dtype), _get_value(b, out.dtype), out=out.d)


@dispatch
def subtract(a: CupyArray | scalar, b: CupyArray | scalar, /, *, out: CupyArray):
    cp.subtract(_get_value(a, out.dtype), _get_value(b, out.dtype), out=out.d)


@dispatch
def mix(alpha: scalar, a: CupyArray | scalar, beta: scalar, b: CupyArray | scalar, /, *, out: CupyArray):
    """Computes out->α·a + β·b."""
    dtype = out.dtype
    _cupy_mix(_get_value(alpha, dtype), _get_value(a, dtype), _get_value(beta, dtype), _get_value(b, dtype), out.d)


def add(a: Array | scalar, b: Array | scalar, /, *, out: Array):
    """Computes a + b

    This is a convenience function for mix(1.0, a, 1.0, b, out=out)
    """
    # mix(1.0, a, 1.0, b, out=out)
    add(a, b, out=out)


def subtract(a: Array | scalar, b: Array | scalar, /, *, out: Array):
    """Computes a - b

    This is a convenience function for mix(1.0, a, -1.0, b, out=out)
    """
    # mix(1.0, a, -1.0, b, out=out)
    subtract(a, b, out=out)


@dispatch
def lerp(a: CupyArray | scalar, b: CupyArray | scalar, weight: CupyArray | scalar, /, *, out: CupyArray):
    """Computes out->a + weight·(b-a)."""
    _cupy_lerp(_get_value(a, out.dtype), _get_value(b, out.dtype), _get_value(weight, out.dtype), out.d)


@dispatch
def multiply(a: CupyArray | scalar, b: CupyArray | scalar, /, *, out: CupyArray):
    cp.multiply(_get_value(a, out.dtype), _get_value(b, out.dtype), out=out.d)


@dispatch
def matmul(matrix: CupyArray, x: CupyArray, /, *, axis: int, out: CupyArray):
    """Matrix multiplication along one axis.

    The multiplication is performed along the specified axis, all other axes of the input are untouched.
    All 1-D slices along the specified axis, are left-multiplied by ``matrix``, and stored in the corresponding
    slice of the output array. Requires ``input.shape(axis) == matrix.shape(1)`` and ``output.shape(axis) == matrix.shape(0)``
    """
    # create a view of both input and output, where the axis over which we multiply are
    # moved to second last axis
    d_in = cp.moveaxis(x.d, axis, x.d.ndim - 2)
    d_out = cp.moveaxis(out.d, axis, x.d.ndim - 2)
    cp.matmul(matrix.d, d_in, out=d_out)


@dispatch
def inner_product(a: CupyArray, b: CupyArray, /) -> complex:
    retval = cp.vdot(a.d, b.d)
    return retval.real if a is b else retval  # remove small imaginary part if present


@dispatch
def norm_squared(a: CupyArray, /) -> complex:
    retval = cp.vdot(a.d, a.d)
    return retval.real  # remove small imaginary part if present


@dispatch
def copy(value: np.ndarray | cp.ndarray, /, *, out: CupyArray):
    """Copies data from a numpy array into this Array."""
    with out.d.device:
        out.d = None  # should release memory
        out.d = cp.array(value, dtype=out.dtype)


@dispatch
def copy(value: CupyArray, /, *, out: CupyArray):
    """Copies data from a numpy array into this Array."""
    cp.copyto(out.d, value.d)


@dispatch
def copy(value: scalar, /, *, out: CupyArray):
    """Sets all elements in the array to the specified scalar value."""
    out.d.fill(value)


@dispatch
def fft(x: CupyArray, /, *, axes: tuple[int, ...], out: CupyArray):
    """Perform an FFT along the specified axis"""
    if out.shape != x.shape or out.dtype != x.dtype or (out.dtype != np.complex64 and out.dtype != np.complex128):
        raise ValueError(
            f"Output array {x.shape, x.dtype} must have the same shape"
            f" and dtype as the input array {out.shape, out.dtype}, and both dtypes must be complex."
        )
    out.d = cp.fft.fftn(x.d, axes=axes)


@dispatch
def ifft(x: CupyArray, /, *, axes: tuple[int, ...], out: CupyArray):
    """Perform an inverse FFT along the specified axis"""
    if out.shape != x.shape or out.dtype != x.dtype or (out.dtype != np.complex64 and out.dtype != np.complex128):
        raise ValueError("Output array must have the same shape and dtype as the input array")
    out.d = cp.fft.ifftn(x.d, axes=axes)


def _get_value(x: CupyArray | scalar, dtype) -> np.ndarray | scalar:
    if isinstance(x, CupyArray):
        if x.dtype != dtype:
            raise ValueError(f"Expected dtype {dtype}, got {x.dtype}")
        return x.d

    if dtype.kind == "c":
        return complex(x)
    else:
        if cp.dtype(x).kind == "c":
            raise ValueError(f"Expected real value, got complex value {x}")
        return x
