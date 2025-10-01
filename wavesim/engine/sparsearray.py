from typing import Sequence
import numpy as np

from .array import Array, copy, scalar, scale, Factory
from .constantarray import ConstantArray
from .dispatcher import dispatch
from .index_utils import shape_like
from .numpyarray import NumpyArray


class SparseArray(Array):
    """A sparse array that can be used to represent a sum of arrays at different positions.

    Note: for the included arrays, a size of 1 is not broadcast to the full shape.
    """

    def __init__(
        self, data: Array | Sequence[Array], /, *, 
        at: shape_like | Sequence[shape_like], 
        shape: shape_like, 
        dtype = None,
    ):
        self.data = (data,) if isinstance(data, Array) else tuple(data)
        self.positions = np.asarray(at).reshape(-1, len(shape))
        if len(self.data) != len(self.positions):
            raise ValueError("The number of arrays and positions must match")
        if len(self.data) > 0:
            dtype = self.data[0].dtype
        elif dtype is None:
            raise ValueError("dtype must be specified when no arrays are provided")

        # all arrays must have the same number of dimensions and data type
        # they should also fit within the specified shape
        for a, pos in zip(self.data, self.positions):
            if any(pos < 0) or any(pos + a.shape > shape):  # noqa type inference bug
                raise ValueError(
                    f"Array of shape {a.shape} at position {pos} does not fit within the specified shape {shape}"
                )
            if a.ndim != len(shape):
                raise ValueError(f"Array of shape {a.shape} at position {pos} has the wrong number of dimensions")
            if a.dtype != dtype:
                raise ValueError("All arrays must have the same data type")

        super().__init__(shape=shape, dtype=dtype, at=at)

    def gather(self) -> np.ndarray:
        result = NumpyArray(None, shape=self.shape, dtype=self.dtype)
        copy(self, out=result)
        return result.gather()

    def _slice(self, start: np.array, stop: np.array) -> Array:
        data = []
        at = []
        for a, pos in zip(self.data, self.positions):
            # convert to array-relative position and crop slices to fit within the array
            relative_start = np.maximum(start - pos, 0)
            relative_stop = np.minimum(stop - pos, a.shape)
            if all(relative_start < relative_stop):  # noqa
                data.append(a._slice(relative_start, relative_stop))
                at.append(tuple(np.maximum(pos - start, 0)))

        shape = stop - start
        if len(data) == 0:
            return ConstantArray(0.0, shape=shape)
        return SparseArray(data, at=at, shape=shape)

    def transpose(
        self, axes: int | Sequence[int | None], *, to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> Array:
        from_axes, new_axes, shape = self._parse_transpose(axes, to, ndim)

        def transpose_position(p):
            p = np.asarray(p)[from_axes]  # shuffle the axes
            for i in new_axes:  # insert new axes (pos = 0)
                p = np.insert(p, i, 0)
            return tuple(p)

        data = [a.transpose(axes, to=to, ndim=ndim) for a in self.data]
        positions = [transpose_position(p) for p in self.positions]
        return SparseArray(data, at=positions, shape=shape)

    def __add__(self, other: "SparseArray | None"):
        """Combines two sparse arrays into a new one containing the blocks of both arrays."""
        if other is None:
            return self

        data = self.data + other.data
        positions = np.concat((self.positions, other.positions))
        return SparseArray(data, at=positions, shape=self.shape)

    def __radd__(self, other: "SparseArray | None"):
        """Combines two sparse arrays into a new one containing the blocks of both arrays."""
        if other is None:
            return self
        return other.__add__(self)

    @property
    def is_full(self):
        return False

    def factory(self) -> Factory:
        raise NotImplementedError("SparseArray does not support cloning")

    @staticmethod
    def point(*, value: scalar = 1.0, at: shape_like, shape: shape_like, dtype=None):
        """Creates a sparse array containing a single point."""
        return SparseArray(ConstantArray(value, shape=(1,) * len(shape), dtype=dtype), at=at, shape=shape)


@dispatch
def mix(
    alpha: scalar, a: SparseArray | Array | scalar, beta: scalar, b: SparseArray | Array | scalar, /, *, out: Array
):
    """Computes out->α·a + β·b."""

    # first scale the non-sparse part and store in out
    if not isinstance(b, SparseArray):
        scale(beta, b, out=out)
    elif not isinstance(a, SparseArray):
        scale(alpha, a, out=out)
    else:
        copy(0.0, out=out)

    # then add the sparse parts
    if isinstance(a, SparseArray):
        for a_, out_ in blocked(a, out):
            scale(alpha, a_, offset=out_, out=out_)
    if isinstance(b, SparseArray):
        for b_, out_ in blocked(b, out):
            scale(beta, b_, offset=out_, out=out_)


@dispatch
def multiply(a: SparseArray | Array | scalar, b: SparseArray | Array | scalar, /, *, out: Array):
    """Computes out = a*b"""
    if out is a or out is b:
        raise NotImplementedError("The output array must be different from the input arrays")

    copy(0.0, out=out)
    if isinstance(a, SparseArray):
        for a_, b_, out_ in blocked(a, b, out):
            multiply(a_, b_, out=out_)
    else:  # isinstance(b, SparseArray):
        for b_, a_, out_ in blocked(b, a, out):
            multiply(a_, b_, out=out_)


@dispatch
def add(a: SparseArray | Array | scalar, b: SparseArray | Array | scalar, /, *, out: Array):
    """Computes out = a+b"""
    if not isinstance(b, SparseArray):
        copy(b, out=out)
    elif not isinstance(a, SparseArray):
        copy(a, out=out)
    else:
        copy(0.0, out=out)

    # then add the sparse parts
    if isinstance(a, SparseArray):
        for a_, out_ in blocked(a, out):
            add(a_, out_, out=out_)
    if isinstance(b, SparseArray):
        for b_, out_ in blocked(b, out):
            add(b_, out_, out=out_)


@dispatch
def subtract(a: SparseArray | Array | scalar, b: SparseArray | Array | scalar, /, *, out: Array):
    """Computes out = a-b"""
    mix(1.0, a, -1.0, b, out=out)


@dispatch
def inner_product(a: SparseArray | Array, b: SparseArray | Array, /) -> complex:
    inner = 0.0
    if isinstance(a, SparseArray):
        for a_, b_, _ in blocked(a, b):
            inner += inner_product(a_, b_)
    else:  # isinstance(b, SparseArray):
        for b_, a_, _ in blocked(b, a):
            inner += inner_product(a_, b_)

    return inner.real if a is b else inner  # remove small imaginary part if present


@dispatch
def norm_squared(a: SparseArray | Array, /) -> complex:
    inner = 0.0
    for a_, _ in blocked(a):
        inner += inner_product(a_, a_)

    return inner.real if a is b else inner  # remove small imaginary part if present


@dispatch
def copy(value: SparseArray, /, *, out: Array):
    """Copies data from a SparseArray into another Array."""
    copy(0.0, out=out)
    for value_, out_ in blocked(value, out):
        copy(value_, out=out_)


def blocked(*args):
    """Generator that iterates over all blocks, returning slice, block for each block
    Args:
        args: Arrays to iterate over. The first array should be a sparsearray
    """
    for a, pos in zip(args[0].data, args[0].positions):
        stop = np.add(pos, a.shape)
        yield a, *tuple(b._slice(pos, stop) if isinstance(b, Array) else b for b in args[1:])
