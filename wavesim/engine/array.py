import math
from abc import ABC, abstractmethod
from typing import Sequence, Iterable

import numpy as np

from .dispatcher import dispatch
from .index_utils import slices_to_pos, shape_like, indexing_type
from copy import copy as object_copy

# Define type aliases for better readability
scalar = complex | float
array_like = np.ndarray | scalar


class Array(ABC):
    """Array interface, common for all backends

    A backend should register the following functions using the @dispatch decorator:
    - mix(alpha: scalar, a: Array, beta: scalar, b: Array, out: Array)
    - multiply(a: Array, b: Array, out: Array)
    - divide(a: Array, b: Array, out: Array)
    - matmul(matrix: Array, a: Array, axis: int, out: Array)
    - inner_product(a: Array, b: Array) -> complex
    - norm_squared(a: Array) -> complex
    - lerp(a: Array, b: Array, weight: scalar, out: Array)
    - copy(value: array_like, a: Array)
    - copy(value: scalar, a: Array)

    Where one or more of the Array classes is replaced by the specific backend class.
    """

    def __init__(self, /, *, shape: shape_like, dtype: np.dtype, **kwargs):
        if dtype not in (np.float16, np.float32, np.float64, np.complex64, np.complex128):
            raise ValueError(f"Unsupported data type {dtype}")
        self.dtype = dtype
        self.shape = tuple(shape)
        self._kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, dtype=<{self.dtype}>)"

    def __getitem__(self, slices: indexing_type) -> "Array":
        """Return a view of the array with the specified slices.

        Args:
            slices: tuple of slices or indices for each dimension.
                Negative numbers are counted from the end of the array.
                If an integer is used for indexing, the indexed dimension is removed from the shape (as in numpy).
                Alternatively, an ax_ object may be used for indexing.

        Raises:
            IndexError: if the slice is out of bounds
        """
        return self._slice(*slices_to_pos(slices, self.shape))

    def slice(self, start: Iterable[int], stop: Iterable[int]) -> "Array":
        """Return a view of the array with the specified slices.

        The specified start and stop point must lie within the array boundaries,
        except for broadcast axes, for which the start and stop points are ignored
        and an axis of size 1 is returned.

        Args:
            start: start position for each axis
            stop: stop position for each axis
                The slices are all guaranteed lie inside the array boundaries:
                0 ≤ start ≤ stop ≤ shape
        """
        start = np.fromiter((start_ if sz != 1 else 0 for start_, sz in zip(start, self.shape)), dtype=int)
        stop = np.fromiter((stop_ if sz != 1 else 1 for stop_, sz in zip(stop, self.shape)), dtype=int)
        return self._slice(start, stop)

    def __setitem__(self, slices: indexing_type, value):
        """Set the data of the array with the specified slices.

        The value can be a scalar, a numpy array, or another Array object, or anything else that the backend supports.
        """
        s = self[slices]
        copy(value, out=s)

    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self.shape)

    @property
    def size(self):
        """Total number of elements"""
        return math.prod(self.shape)

    @abstractmethod
    def _slice(self, start: np.array, stop: np.array) -> "Array":
        """Returns a slice of the array.

        Args:
            start: start position for each axis
            stop: stop position for each axis
                The slices are all guaranteed lie inside the array boundaries:
                0 ≤ start ≤ stop ≤ shape
        """
        ...

    @abstractmethod
    def gather(self):
        """Return the data as a numpy array"""
        ...

    @property
    @abstractmethod
    def is_full(self):
        """Returns True if the array is fully populated with data

        True means that the array can be overwritten with full data. For sparse or constant arrays return False.
        """
        ...

    def enclosing_circle(self, enforce_gain_free=False) -> tuple[complex, float]:
        """Computes a circle enclosing all complex values in the array

        The circle does not need to be the smallest enclosing circle.
        It just should be large enough to contain all values:

        max ``|x-c|`` ≤ r

        Arguments:
            enforce_gain_free: When True, this check if the real part of the array elements is non-negative.
                If not, raises an error.

        Returns:
            (c:complex, r:float) the center and radius of the circle
        """
        raise NotImplementedError(f"enclosing_circle not implemented for {self}")

    @abstractmethod
    def transpose(
        self, axes: int | Sequence[int | None], *, to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> "Array":
        """Returns a view of the data with the axes transposed

        Args:
            axes: selection of the axes from the source (self). When an integer, this is equivalent to (axes,).
                Each of the old axes must be present exactly once, except for axes of size 1, which can be removed
                The axes sequence can include 'np.newaxis' entries to indicate newly added broadcast axes (of size 1).
            to: the new axes of the target array. When absent, this is equivalent to 0, 1, 2, 3, ndim-1
            ndim: the total number of axes in the target array.
                The 'to' array is extended with broadcast axes if necessary.
                When absent, equals the number of elements in ``axes`` or the maximum value in ``to`` + 1,
                whichever is higher

        Example:
                # transpose a 1-D array to an array with shape (∞, ∞, N, ∞) in three equivalent ways
                xb1 = x.transpose((np.newaxis, np.newaxis, 0, np.newaxis))
                xb2 = x.transpose(0, to=2, ndim=4)
                xb3 = x.transpose((0, np.newaxis), to=(2,3))
        """
        ...

    def _parse_transpose(
        self, axes: int | Sequence[int | None], to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Helper function to parse the arguments of the transpose function.

        Returns:
            tuple of:
                from_axes: list of axes to select from the source array in order
                new_axes: positions where to insert new broadcast axes (compatible with np.expand_dims).
                    These are the positions in the array _after_ inserting the new axes, so it is not
                    compatible with ``np.insert``
                all_axes: same list as ``from_axis``, interleaved with ``np.newaxis`` at the new axes
        """
        # 'axes' contains a list of all axes in the source array
        # 'ndim' of the output array can be specified, or is inferred from the axes and 'to' arguments
        if not isinstance(axes, Sequence):
            axes = (axes,)
        ndim = ndim if ndim is not None else (np.max(to) + 1 if to is not None else len(axes))
        if ndim < self.ndim:
            raise ValueError("ndim cannot be lower than the dimensionality of the source array (length of 'axes')")

        # now place each of the axes in the correct order, and add broadcast axes if needed
        full_from = np.full(ndim, np.newaxis)
        if to is not None:
            to = np.atleast_1d(np.asarray(to))
            full_from[to] = axes
        else:
            full_from[: len(axes)] = axes

        from_axes = tuple(a for a in full_from if a is not None)
        new_axes = tuple(i for i, a in enumerate(full_from) if a is None)
        shape = tuple(self.shape[a] if a is not None else 1 for a in full_from)
        return tuple(from_axes), tuple(new_axes), shape

    def factory(self) -> "Factory":
        """Returns a factory object that can be used to create new arrays with properties equal to the current array."""
        return Factory(self)


class Factory:
    def __init__(self, template):
        self.constructor = template.__class__
        self.arguments = template._kwargs.copy()
        self.arguments["shape"] = template.shape
        self.arguments["dtype"] = template.dtype

    def __call__(self, data: array_like | None = None, **kwargs):
        """Return an empty array with the same shape, position, and data type as this array, unless overridden."""
        # use defaults for missing arguments
        arguments = {**self.arguments, **kwargs}
        return self.constructor(data, **arguments)

    def to_complex(self):
        """Return a factory object that creates complex arrays"""
        f = object_copy(self)
        f.arguments = object_copy(f.arguments)
        dtype = self.arguments["dtype"]
        if dtype == np.complex64 or dtype == np.complex128:
            return self
        if dtype == np.float32:
            dtype = np.complex64
        elif dtype == np.float64:
            dtype = np.complex128
        else:
            raise ValueError(f"Cannot convert array of dtype {dtype} to complex")
        f.arguments["dtype"] = dtype
        return f

    def to_real(self):
        """Return a factory object that creates real arrays"""
        f = object_copy(self)
        f.arguments = object_copy(f.arguments)
        dtype = self.arguments["dtype"]
        if dtype == np.float32 or dtype == np.float64:
            return self
        if dtype == np.complex64:
            dtype = np.float32
        elif dtype == np.complex128:
            dtype = np.float64
        else:
            raise ValueError(f"Cannot convert array of dtype {dtype} to real")
        f.arguments["dtype"] = dtype
        return f


# region scalar implementations (should be the first in the list)
@dispatch
def mix(alpha: scalar, a: scalar, beta: scalar, b: scalar, /, *, out: Array):
    """Computes out = α·a + β·b"""
    copy(alpha * a + beta * b, out=out)


@dispatch
def multiply(a: scalar, b: scalar, /, *, out: Array):
    """Computes out = a*b"""
    copy(a * b, out=out)


@dispatch
def divide(a: scalar, b: scalar, /, *, out: Array):
    """Computes out = a/b"""
    copy(a / b, out=out)


@dispatch
def add(a: scalar, b: scalar, /, *, out: Array):
    """Computes out = a+b"""
    copy(a + b, out=out)


@dispatch
def subtract(a: scalar, b: scalar, /, *, out: Array):
    """Computes out = a-b"""
    copy(a - b, out=out)


@dispatch
def inner_product(a: scalar, b: scalar, /) -> complex:
    return np.conj(a) * b


@dispatch
def norm_squared(a: scalar, /) -> complex:
    return np.conj(a) * a


@dispatch
def lerp(
    a: scalar,
    b: scalar,
    weight: scalar,
    /,
    *,
    out: Array,
):
    """Computes out = (1-weight)·a + weight·b = a + weight · (b-a)"""
    copy(a + weight * (b - a), out=out)


# endregion


# region Functions that have a default implementation in terms of other functions
@dispatch
def lerp(a: Array | scalar, b: Array | scalar, weight: scalar, /, *, out: Array):
    """Computes out = (1-weight)·a + weight·b = a + weight · (b-a)

    Default implementation in case weight is a scalar.
    """
    mix(1.0 - weight, a, weight, b, out=out)


@dispatch
def add(a: Array | scalar, b: Array | scalar, /, *, out: Array):
    """Computes a + b

    This is a convenience function for mix(1.0, a, 1.0, b, out=out)
    """
    # mix(1.0, a, 1.0, b, out=out)
    raise NotImplementedError(f"add not implemented for arrays of type ({type(a)}, {type(b)})->{type(out)}")


@dispatch
def subtract(a: Array | scalar, b: Array | scalar, /, *, out: Array):
    """Computes a - b

    This is a convenience function for mix(1.0, a, -1.0, b, out=out)
    """
    # mix(1.0, a, -1.0, b, out=out)
    raise NotImplementedError(f"add not implemented for arrays of type ({type(a)}, {type(b)})->{type(out)}")


def scale(alpha: scalar, a: Array | scalar, /, *, offset: Array | scalar = 0.0, out: Array):
    """Computes out = α·a + offset

    This is a convenience function for mix(alpha, a, 0.0, offset, out=out)
    """
    mix(alpha, a, 1.0, offset, out=out)


@dispatch
def copy(value: Array, /, *, out: Array):
    """Copies data from one array to the other

    This default implementation uses numpy arrays an intermediate,
    and relies on the copy(a: np.ndarray, out: SomeArrayType) to be defined for each array type."""
    np.broadcast_shapes(value.shape, out.shape)
    copy(value.gather(), out=out)


@dispatch
def copy(value: int, /, *, out: Array):
    """Implementation for setting a value to an integer.

    This typically happens when setting an array to 0 or 1 (instead of 0.0 or 1.0)
    """
    copy(float(value), out=out)


# endregion


# region Functions that should be implemented by the backend


@dispatch
def copy(value: np.ndarray | scalar, /, *, out: Array):
    """Stores a numpy array or scalar in the array"""
    raise NotImplementedError(f"copy not implemented for {type(value)}->{type(out)}")


@dispatch
def mix(alpha: scalar, a: Array | scalar, beta: scalar, b: Array | scalar, /, *, out: Array):  # noqa unused arguments
    """Computes out = α·a + β·b"""
    raise NotImplementedError(f"mix not implemented for arrays of type {type(a)}, {type(b)}, {type(out)}")


@dispatch
def lerp(
    a: Array | scalar,
    b: Array | scalar,
    weight: Array | scalar,
    /,
    *,
    out: Array,
):
    """Computes out = (1-weight)·a + weight·b = a + weight · (b-a)"""
    raise NotImplementedError(
        f"lerp not implemented for arrays of type ({type(a)}, {type(b)}, {type(weight)})->{type(out)}"
    )


@dispatch
def multiply(a: Array | scalar, b: Array | scalar, /, *, out: Array):
    """Computes out = a*b"""
    raise NotImplementedError(f"multiply not implemented for arrays of type ({type(a)}, {type(b)})->{type(out)}")


@dispatch
def divide(a: Array | scalar, b: Array | scalar, /, *, out: Array):
    """Computes out = a/b"""
    raise NotImplementedError(f"divide not implemented for arrays of type ({type(a)}, {type(b)}), {type(out)}")


@dispatch
def inner_product(a: Array | scalar, b: Array | scalar, /) -> complex:
    raise NotImplementedError(f"inner_product not implemented for arrays of type ({type(a)}, {type(b)})")


@dispatch
def norm_squared(a: Array | scalar, /) -> complex:
    raise NotImplementedError(f"norm_squared not implemented for arrays of type ({type(a)})")


@dispatch
def matmul(matrix: Array, x: Array, /, *, axis: int, out: Array):  # noqa unused arguments
    raise NotImplementedError(f"matmul not implemented for arrays of type ({type(matrix)}, {type(x)})->{type(out)}")


@dispatch
def fft(x: Array, /, *, axes: tuple[int, ...], out: Array):  # noqa unused arguments
    """Perform an FFT along the specified axis"""
    raise NotImplementedError(f"fft not implemented for {type(x)}->{type(out)}")


@dispatch
def ifft(x: Array, /, *, axes: tuple[int, ...], out: Array):  # noqa unused arguments
    """Perform an inverse FFT along the specified axis"""
    raise NotImplementedError(f"ifft not implemented for {type(x)}->{type(out)}")


# endregion


# region Factory functions
def new_like(x: Array, data, *, copy: bool = True, **kwargs) -> Array:  # noqa shadows name copy
    """Create a new array with the same parameters as the input, and initialize it with 'data'

    Args:
        x: Array to copy the parameters from
        data: data to initialize the array with
        copy: when True, always copies the input data to a newly allocated array. When False, just wraps the data
        kwargs: optional parameters to change in the new array, such as shape or pos
    """
    return x.factory().__call__(data, copy=copy, **kwargs)


def empty_like(x: Array, **kwargs) -> Array:
    """Create an empty (non-initialized) array with the same parameters as the input.

    Args:
        x: Array to copy the parameters from
        kwargs: optional parameters to change in the new array, such as shape or pos
    """
    return new_like(x, None, **kwargs)


def zeros_like(x: Array, **kwargs) -> Array:
    """Create a zero-filled array with the same shape and dtype as the input.

    Args:
        x: Array to copy the parameters from
        kwargs: optional parameters to change in the new array, such as shape or pos
    """
    return new_like(x, 0, **kwargs)


def as_complex(x: Array) -> Array:
    """Converts the array to a complex array, by adding a zero imaginary part"""
    if x.dtype.kind == "c":
        return x
    if x.dtype == np.float32:
        dtype = np.complex64
    elif x.dtype == np.float64:
        dtype = np.complex128
    else:
        raise ValueError(f"Cannot convert array of dtype {x.dtype} to complex")
    retval = empty_like(x, dtype=dtype)
    copy(x, out=retval)
    return retval


def as_type(x: Array, dtype: np.dtype) -> Array:
    """Converts the array to the specified datatype"""
    if x.dtype == dtype:
        return x
    retval = empty_like(x, dtype=dtype)
    copy(x, out=retval)
    return retval


def clone(x: Array) -> Array:
    """Clone the array"""
    out = empty_like(x)
    copy(x, out=out)
    return out


# endregion
