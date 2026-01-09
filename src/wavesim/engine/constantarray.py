from typing import Sequence

import numpy as np

from .array import Array, scalar
from .dispatcher import dispatch
from .index_utils import shape_like


class ConstantArray(Array):
    """Array that contains a constant value, not writable"""

    def __init__(self, data: scalar, /, *, shape: shape_like, copy=True, dtype=None):
        """

        Args:
            data:
            shape:
            copy: provided for compatibility of the signature only, data is always copied
            dtype:
        """
        if isinstance(data, float):
            dtype = dtype or np.float32
        elif isinstance(data, complex):
            dtype = dtype or np.complex64
        else:
            raise ValueError(f"Unsupported value type {type(data)}")

        self.value = data
        super().__init__(shape=shape, dtype=dtype)

    def gather(self) -> np.ndarray:
        return np.full(self.shape, self.value, dtype=self.dtype)

    @property
    def is_full(self):
        return False

    def transpose(
        self, axes: int | Sequence[int | None], *, to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> Array:
        from_axes, new_axes, shape = self._parse_transpose(axes, to, ndim)
        return ConstantArray(self.value, shape=shape)

    def _slice(self, start: np.array, stop: np.array) -> "ConstantArray":
        return ConstantArray(self.value, shape=stop - start)

    def __repr__(self):
        return super().__repr__() + f", value={self.value})"


@dispatch
def mix(
    alpha: scalar, a: ConstantArray | Array | scalar, beta: scalar, b: ConstantArray | Array | scalar, /, *, out: Array
):
    """Computes out = α·a + β·b"""
    mix(alpha, _get_value(a), beta, _get_value(b), out=out)


@dispatch
def lerp(
    a: ConstantArray | Array | scalar,
    b: ConstantArray | Array | scalar,
    weight: ConstantArray | Array | scalar,
    /,
    *,
    out: Array,
):
    """Computes out = a*weight + b * (1-weight)"""
    lerp(_get_value(a), _get_value(b), _get_value(weight), out=out)


@dispatch
def multiply(a: ConstantArray | Array | scalar, b: ConstantArray | Array | scalar, /, *, out: Array):
    """Computes out = a*b"""
    multiply(_get_value(a), _get_value(b), out=out)


@dispatch
def add(a: ConstantArray | Array | scalar, b: ConstantArray | Array | scalar, /, *, out: Array):
    """Computes out = a+b"""
    add(_get_value(a), _get_value(b), out=out)


@dispatch
def subtract(a: ConstantArray | Array | scalar, b: ConstantArray | Array | scalar, /, *, out: Array):
    """Computes out = a-b"""
    subtract(_get_value(a), _get_value(b), out=out)


@dispatch
def divide(a: ConstantArray | Array | scalar, b: ConstantArray | Array | scalar, /, *, out: Array):
    """Computes out = a/b"""
    divide(_get_value(a), _get_value(b), out=out)


@dispatch
def inner_product(a: ConstantArray | Array | scalar, b: ConstantArray | Array | scalar, /) -> complex:
    return inner_product(_get_value(a), _get_value(b))


@dispatch
def norm_squared(a: ConstantArray | Array | scalar,/) -> complex:
    return norm_squared(_get_value(a))


@dispatch
def copy(value: ConstantArray, /, *, out: Array):
    """Copies constant data into the out array."""
    copy(value.value, out=out)


@dispatch
def copy(value: scalar, /, *, out: ConstantArray):
    """Copies constant data into the out array."""
    out.value = value


def _get_value(x: Array) -> scalar | Array:
    """Converts a ConstantArray to a scalar"""
    if isinstance(x, ConstantArray):
        return x.value
    return x
