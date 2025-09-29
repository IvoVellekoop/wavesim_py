from typing import Sequence, SupportsIndex
import numpy as np
from numpy.typing import NDArray

from .array import Array, multiply, divide, copy, empty_like, zeros_like, fft, ifft, new_like
from .index_utils import ax_
from .blockarray import BlockArray, block_shape, block_iter

boundary_spec = SupportsIndex, Sequence[tuple[SupportsIndex | SupportsIndex]]


def reshape_axis(array: np.ndarray, *, ndim: int, axis: int):
    """Reshapes a 1-D array to an n-dimensional array
    Args:
        array: 1-D array to reshape
        ndim: number of dimensions of the output array
        axis: axis along which to store the data in the output array

    Returns:
        an n-dimensional array with the data stored along the specified axis
    """
    assert array.ndim == 1
    shape = (1,) * axis + (array.size,) + (1,) * (ndim - axis - 1)
    return array.reshape(shape)


def convolve(x: Array, kernel: Array, *, out: Array):
    """Computes the convolution x * y through a fast convolution.

    out = F⁻¹(F(x) · F(y))
    """
    if kernel is out:
        raise ValueError("The output array must be different from the kernel")
    if not x.is_full:
        copy(x, out=out)
        x = out
    axes = tuple(range(x.ndim))
    fft(x, axes=axes, out=out)
    multiply(out, kernel, out=out)
    ifft(out, axes=axes, out=out)


def deconvolve(x: Array, kernel: Array, *, out: Array):
    if kernel is out:
        raise ValueError("The output array must be different from the kernel")
    if not x.is_full:
        copy(x, out=out)
        x = out
    axes = tuple(range(x.ndim))
    fft(x, axes=axes, out=out)
    divide(out, kernel, out=out)
    ifft(out, axes=axes, out=out)


def combine(x: BlockArray) -> Array:
    """Concatenates a BlockArray into a single array

    The returned array has the same dtype and Array class as the first block in the original BlockArray.
    """
    template = x
    while isinstance(template, BlockArray):
        template = x.blocks.flat[0]
    return new_like(template, x.gather())


def pad(x: Array, widths: boundary_spec, mode: str = "zero", block_array: bool = False) -> Array:
    """Pad the array with the specified widths along each dimension

    Args:
        x: Array to pad
        widths: sequence of (before, after) padding width along each dimension,
            or an integer to use the same padding for all edges
        mode: padding mode ('zero' or 'edge')
        block_array:
            if True, return a BlockArray with the padded data. The original data is not copied.
            The dtype and Array class for the edges are copied from 'x'
            if False, return a new Array with the padded data. The original data is copied.
            The dtype and Array class are copied from 'x'
    Returns:
        A new Array with the padded data
    """
    if len(widths) != x.ndim:
        raise ValueError("Number of widths must match the number of dimensions of the data array")

    # create a numpy array holding all the blocks, and add padding to the boundary coordinates
    blocks = np.array(x, dtype=object).reshape((1,) * x.ndim)
    for d, w in enumerate(widths):
        original_blocks = blocks
        if w[0] > 0:
            padding = _pad_1d(original_blocks, width=w[0], axis=d, left=True, mode=mode)
            blocks = np.append(padding, blocks, axis=d)
        if w[1] > 0:
            padding = _pad_1d(original_blocks, width=w[1], axis=d, left=False, mode=mode)
            blocks = np.append(blocks, padding, axis=d)

    if blocks.size == 1:
        return x

    padded = BlockArray(blocks)
    if block_array:
        return padded
    else:
        # merge blocks into one array of the same type as x
        return combine(padded)


def _pad_1d(blocks: NDArray[Array], width: int, axis: int, left: bool, mode: str) -> NDArray[Array]:
    padding = np.empty_like(blocks)
    for block_index, block in np.ndenumerate(blocks):
        shape = list(block.shape)
        shape[axis] = width
        if mode == "zero":
            p = zeros_like(block, shape=shape)
        else:
            p = empty_like(block, shape=shape)
            if left:
                copy(block[ax_(axis)[0]], out=p)
            else:
                copy(block[ax_(axis)[-1]], out=p)
        padding[block_index] = p
    return padding


def edges(x: Array, widths: boundary_spec, empty_as_none: bool = False) -> NDArray[Array]:
    """Returns views to all edges of the array with the specified widths.

    Args:
        x: Array to extract the edges from
        widths: widths of the edges along each dimension.
        empty_as_none: If True, return None for empty edges. Otherwise, return a zero-sized array.
    Returns:
        x.ndim × 2 object array views of the edges of the array
    """
    widths = np.full(shape=(x.ndim, 2), fill_value=widths) if np.isscalar(widths) else np.asarray(widths)
    retval = np.empty((x.ndim, 2), dtype=object)
    for d, w in enumerate(widths):
        # Note: need special case for w[1]==0, because [-0:] does not work
        retval[d, 0] = x[ax_(d)[: w[0]]] if w[0] != 0 or not empty_as_none else None
        retval[d, 1] = x[ax_(d)[-w[1] :]] if w[1] != 0 else (x[ax_(d)[-1:-1]] if not empty_as_none else None)

    return retval


def block_edges(x: Array, widths: boundary_spec, *, empty_as_none: bool = False) -> NDArray[Array]:
    """Similar to edges, but also extracts all internal edges of a BlockArray.

    Args:
        x: The BlockArray. A regular array is treated as a 1×1×1×... block array.
        widths: The width of the edges to extract for each dimension
        empty_as_none: If True, return None for empty edges. Otherwise, return an empty array.
    Returns:
        An array containing views for the external and internal edges of the x.
        It has dimension (*x.blocks.shape, x.ndim, 2), where the last axis contains the left and right edges.
    """
    all_edges = np.empty(shape=(*block_shape(x), x.ndim, 2), dtype=object)
    for block, edges_ in block_iter(x, all_edges):
        edges_[:, :] = edges(block, widths, empty_as_none=empty_as_none)

    return all_edges
