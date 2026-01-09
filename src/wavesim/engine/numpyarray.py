from typing import Sequence
from numba import njit

import numpy as np
import scipy

from .array import array_like, scalar, Array, dispatch
from .index_utils import shape_like, pos_to_slices


class NumpyArray(Array):
    """Numpy-implementation of the array interface.

    This implementation is not shared, it is only used internally by the numpy engine.
    """

    def __init__(
        self,
        data: array_like | Array | None,
        *,
        shape: shape_like | None = None,
        copy: bool = False,
        dtype=None,
    ):
        """Constructs a new NumpyArray.
        Args:
            data: numpy array to copy into the new array
            shape: shape of the array, only used when data is a scalar, ignored otherwise.
            copy:
                If False (default) and the data is a ndarray of the correct type, the data array is stored directly.
                if True, the data is copied into the new array. This value is ignored for scalars.
            dtype:
        """
        if data is None:
            if shape is None:
                raise ValueError("Shape must be specified when data is a scalar")
            self.d = np.empty(shape, dtype=dtype)
        elif np.isscalar(data):
            if shape is None:
                raise ValueError("Shape must be specified when data is a scalar")
            self.d = np.full(shape, data, dtype=dtype)
        elif isinstance(data, Array):
            data = data.gather()
            self.d = np.array(data, dtype=dtype) if copy else np.asarray(data, dtype=dtype)
        else:
            self.d = np.array(data, dtype=dtype) if copy else np.asarray(data, dtype=dtype)
        super().__init__(shape=self.d.shape, dtype=self.d.dtype)

    def _slice(self, start: np.array, stop: np.array) -> "NumpyArray":
        return NumpyArray(self.d[*pos_to_slices(start, stop)])

    def gather(self):
        return self.d

    @property
    def is_full(self):
        return True

    def transpose(
        self, axes: int | Sequence[int | None], *, to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> Array:

        from_axes, new_axes, shape = self._parse_transpose(axes, to, ndim)
        d = np.expand_dims(self.d.transpose(from_axes), new_axes)
        return NumpyArray(d, shape=shape)

    def enclosing_circle(self, enforce_gain_free=False) -> tuple[complex, float]:
        r_min = np.min(self.d.real)
        r_max = np.max(self.d.real)
        i_min = np.min(self.d.imag)
        i_max = np.max(self.d.imag)
        if enforce_gain_free and i_min < 0:
            raise ValueError("Imaginary part of the array musts be non-negative.")

        c = (r_min + r_max) / 2 + 1j * (i_min + i_max) / 2
        r = np.max(np.abs(self.d - c))
        return complex(c), float(r)


# Kernels for the numpy engine. These are optimized with numba
# the option fastmath=True enables the use of fused multiply-add functions, which are faster and *more* accurate


@njit(fastmath=True)
def _numba_mix(alpha, a, beta, b, out):
    out[:] = alpha * a + beta * b


@njit(fastmath=True)
def _numba_lerp(a, b, weight, out):
    out[:] = a + weight * (b - a)


@dispatch
def divide(a: NumpyArray | scalar, b: NumpyArray | scalar, /, *, out: NumpyArray):
    np.divide(_get_value(a), _get_value(b), out=out.d)


@dispatch
def add(a: NumpyArray | scalar, b: NumpyArray | scalar, /, *, out: NumpyArray):
    np.add(_get_value(a), _get_value(b), out=out.d)


@dispatch
def subtract(a: NumpyArray | scalar, b: NumpyArray | scalar, /, *, out: NumpyArray):
    np.subtract(_get_value(a), _get_value(b), out=out.d)


@dispatch
def mix(alpha: scalar, a: NumpyArray | scalar, beta: scalar, b: NumpyArray | scalar, /, *, out: NumpyArray):
    """Computes out->α·a + β·b."""
    _numba_mix(alpha, _get_value(a), beta, _get_value(b), out.d)


@dispatch
def lerp(a: NumpyArray | scalar, b: NumpyArray | scalar, weight: NumpyArray | scalar, /, *, out: NumpyArray):
    """Computes out->a + weight·(b-a)."""
    _numba_lerp(_get_value(a), _get_value(b), _get_value(weight), out.d)


@dispatch
def multiply(a: NumpyArray | scalar, b: NumpyArray | scalar, /, *, out: NumpyArray):
    np.multiply(_get_value(a), _get_value(b), out=out.d)


@dispatch
def matmul(matrix: NumpyArray, x: NumpyArray, /, *, axis: int, out: NumpyArray):
    """Matrix multiplication along one axis.

    The multiplication is performed along the specified axis, all other axes of the input are untouched.
    All 1-D slices along the specified axis, are left-multiplied by ``matrix``, and stored in the corresponding
    slice of the output array. Requires ``input.shape(axis) == matrix.shape(1)`` and ``output.shape(axis) == matrix.shape(0)``
    """
    # create a view of both input and output, where the axis over which we multiply are
    # moved to second last axis
    d_in = np.moveaxis(x.d, axis, x.d.ndim - 2)
    d_out = np.moveaxis(out.d, axis, x.d.ndim - 2)
    np.matmul(matrix.d, d_in, out=d_out)


@dispatch
def inner_product(a: NumpyArray, b: NumpyArray, /) -> complex:
    retval = np.vdot(a.d, b.d)
    return retval.real if a is b else retval  # remove small imaginary part if present


@dispatch
def norm_squared(a: NumpyArray, /) -> complex:
    retval = np.vdot(a.d, a.d)
    return retval.real  # remove small imaginary part if present


@dispatch
def copy(value: np.ndarray, /, *, out: NumpyArray):
    """Copies data from a numpy array into this Array."""
    out.d[:] = value


@dispatch
def copy(value: scalar, /, *, out: NumpyArray):
    """Sets all elements in the array to the specified scalar value."""
    out.d.fill(value)


@dispatch
def fft(x: NumpyArray, /, *, axes: tuple[int, ...], out: NumpyArray):
    """Perform an FFT along the specified axis"""
    # todo: prevent new memory allocation
    if out.shape != x.shape or out.dtype != x.dtype or (out.dtype != np.complex64 and out.dtype != np.complex128):
        raise ValueError(
            f"Output array {x.shape, x.dtype} must have the same shape"
            f" and dtype as the input array {out.shape, out.dtype}, and both dtypes must be complex."
        )
    out.d = scipy.fft.fftn(x.d, axes=axes)


@dispatch
def ifft(x: NumpyArray, /, *, axes: tuple[int, ...], out: NumpyArray):
    """Perform an inverse FFT along the specified axis"""
    # todo: prevent new memory allocation
    if out.shape != x.shape or out.dtype != x.dtype or (out.dtype != np.complex64 and out.dtype != np.complex128):
        raise ValueError("Output array must have the same shape and dtype as the input array")
    out.d = scipy.fft.ifftn(x.d, axes=axes)


if __name__ == "__main__":
    # Test the numba engine
    c_ = np.ones(2**12, dtype=np.float64)
    a_ = 2.0 * c_
    _numba_mix(2.0, a_, 3.0, c_, c_)
    code = list(_numba_mix.inspect_llvm().values())[0]
    asm = list(_numba_mix.inspect_asm().values())[0]
    print(f"Using vectorized fused multiply-add:{'vfmadd' in asm}")
    print(f"Using inaccurate reciprocal:{'rcp' in asm}")
    print(f"Using inaccurate reciprocal square root:{'rsqrt' in asm}")
    pass


def _get_value(x: NumpyArray | scalar) -> np.ndarray | scalar:
    return x.d if isinstance(x, NumpyArray) else x
