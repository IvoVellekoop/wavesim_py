from itertools import cycle
from typing import Sequence, Iterable, Generator

import numpy as np
from numpy.typing import NDArray

from .array import scalar, Array, dispatch, add, Factory
from .index_utils import shape_like, ax_
from .numpyarray import NumpyArray


class BlockArray(Array):
    """An array consisting of a regular grid of subarrays (blocks).

    The blocks must be non-overlapping and cover the entire array.
    """

    @property
    def is_full(self):
        return all(b.is_full for b in self)

    def __init__(
        self,
        x: Array | NDArray[Array] | np.ndarray | scalar | None,
        /,
        *,
        shape: shape_like | None = None,
        dtype=None,
        n_domains: shape_like | None = None,
        boundaries: Sequence[Sequence[int]] | None = None,
        factories: NDArray[object] | Sequence[Factory] | Factory | None = None,
        copy: bool = False,  # noqa shadows name copy
    ):
        """Construct a new BlockArray (a partitioned array).

        There are five ways to construct a BlockArray:
            1. An un-initialized array. x = None. shape, dtype, and factories should be specified,
                as well as either boundaries or n_domains.
            2. A new array initialized with a scalar. x: scalar. Same requirements as 1.
            3. From a regular ndarray or sequence (of sequences of) values. Factories should be specified.
            4. From an Array object. x: Array. The dtype, and factories are taken from the Array,
                unless specified explicitly. shape is always taken from the array, the input argument is ignored.
                either boundaries or n_domains should be specified.
            5. From an ndarray of Arrays. x: NDArray[Array]. The dtype, and factories are taken from the Arrays.
                boundaries and n_domains are ignored (boundaries are taken from the Arrays).
                factories must be specified.


        Returns the subarrays as a BlockArray. If copy=False, the BlockArray contains views of the original data, unless
        for blocks that have different factory or dtype than the original data.

        Args:
            x: The Array to split, None, or a scalar to fill the new array with, or an ndarray of Arrays for
                pre-split data.
            n_domains: The number of domains to split the array into along each dimension.
                The domains will have approximately equal size.
                Takes precedence over ``boundaries``, ignored for pre-split data.
            boundaries: The internal boundaries along which to split the array, specified as a sequence of integers for
                each dimension. Ignored for pre-split data or when ``n_domains`` is specified.
            shape: The shape of the new array. Only used when x is a scalar or None, ignored otherwise.
            dtype: The dtype of the new array. If None, the dtype of the input data is used.
            factories: The factories to use for the new array. If None, the factories of the input array are used.
            copy: If True, the data is copied into the new array. If False, the data is stored directly.
        """
        blocks = None
        if x is None:
            # non-initialized data (case 1)
            pass
        elif np.isscalar(x):
            # scalar data (case 2)
            dtype = np.dtype(type(x)) if dtype is None else dtype
        elif isinstance(x, Array):
            # existing array (case 4)
            shape = x.shape
            factories = x.factory() if factories is None else factories
            dtype = x.dtype if dtype is None else dtype
        elif isinstance(x, np.ndarray) and x.dtype == object:
            # pre-split data (case 5)
            factories = block_map(x, lambda b: b.factory()) if factories is None else factories
            boundaries, shape, dtype = BlockArray._verify_pre_split_data(x)
            n_domains = None
            blocks = x
        else:
            # regular ndarray data, or anything that can converted to it (case 3)
            x = NumpyArray(x, dtype=dtype, shape=shape, copy=False)
            shape = x.shape
            dtype = x.dtype

        # at this point, shape, dtype, and factories should be defined, as well as boundaries and/or n_domains
        # x is an Array, a scalar, or None
        if factories is None:
            raise ValueError("Factories must be specified")
        if shape is None:
            raise ValueError("Shape must be specified")
        if dtype is None:
            raise ValueError("dtype must be specified")

        if n_domains is not None:
            # divide over approximately equally sized domains
            # this option overrides any boundaries that may have been specified
            boundaries = []
            boundaries_ex = []
            for n, length in zip(n_domains, shape):
                if not (1 <= n <= length):
                    raise ValueError(f"Number of domains {n} must be between 1 and the length of the array {length}")

                smaller_size = length // n
                larger_count = length % n
                smaller_count = n - larger_count
                positions = np.cumsum((smaller_size + 1,) * larger_count + (smaller_size,) * smaller_count)
                boundaries.append(positions[:-1])
                boundaries_ex.append((0, *positions))
        else:
            # check if all boundaries are increasing and within the shape of the data
            if len(boundaries) != len(shape):
                raise ValueError("The number of boundaries must match the number of dimensions of the data")
            boundaries_ex = [(0, *b, s) for b, s in zip(boundaries, shape)]
            for b, s in zip(boundaries_ex, shape):
                if (s > 0 and any(b[i + 1] <= b[i] for i in range(len(b) - 1))) or (s == 0 and len(b) > 2):
                    raise ValueError(f"Invalid block boundaries {boundaries} for shape {shape}")

        # process the factory input (if present). Turn into one long repeating sequence of factories
        factories = np.ravel(np.asarray(factories, dtype=object))

        # Partition the data into blocks according to the boundaries.
        if blocks is None:
            blocks = np.empty([len(b) - 1 for b in boundaries_ex], dtype=object)
            for idx, factory in zip(np.ndindex(blocks.shape), cycle(factories)):
                start = np.array([b[i] for b, i in zip(boundaries_ex, idx)])
                stop = np.array([b[i + 1] for b, i in zip(boundaries_ex, idx)])
                value = x.slice(start, stop) if isinstance(x, Array) else x
                blocks[idx] = factory(value, copy=copy, dtype=dtype, shape=stop - start)

        self.boundaries = boundaries
        self.blocks = blocks
        self.factories = factories
        super().__init__(shape=shape, dtype=dtype, factories=factories, boundaries=self.boundaries)

    def gather(self) -> np.ndarray:
        """Combine the subarrays into a single array

        This is done by adding this array to a numpy zero array.
        """
        total = NumpyArray(None, shape=self.shape, dtype=self.dtype)
        for out, block in blocked((total, self)):
            copy(block, out=out)
        return total.gather()

    def __iter__(self) -> Iterable[Array]:
        return self.blocks.flat

    def transpose(
        self, axes: int | Sequence[int | None], *, to: int | Sequence[int] | None = None, ndim: int | None = None
    ) -> Array:
        # return a new array with each contained block transposed.
        # The block and position array themselves are not transposed
        from_axes, new_axes, shape = self._parse_transpose(axes, to, ndim)

        new_blocks = block_map(self.blocks, lambda x: x.transpose(axes, to=to, ndim=ndim))
        new_blocks = np.expand_dims(new_blocks.transpose(from_axes), new_axes)

        return BlockArray(new_blocks)

    def _slice(self, start: np.array, stop: np.array) -> Array:
        # remove all blocks that are completely outside the bounds,
        # keep all blocks that are completely inside the bounds and adjust position
        # slice all blocks that are partially inside the bounds and adjust position
        blocks = self.blocks.copy()
        for d, (s_start, s_stop) in enumerate(zip(start, stop)):
            b = [0, *self.boundaries[d], self.shape[d]]

            # remove blocks that are completely outside the bounds
            keep = np.nonzero(np.logical_and(s_start < b[1:], s_stop > b[:-1]))[0]
            if len(keep) == 0:
                # this happens when we select a zero-sized slice directly at the edge of a block
                # in this case, select a zero slice from the block directly before the start==stop index
                # Unless start==stop==0, in which case we select the first block.
                keep = 0 if s_start == 0 else (np.searchsorted(b, s_start) - 1)
                keep = (keep, keep)

            blocks = blocks[keep[0] : keep[-1] + 1, ...]
            b = b[keep[0] : keep[-1] + 2]

            # clip blocks that are partially inside the bounds
            if s_start > b[0] or s_stop < b[1]:  # clip first block
                part_start = s_start - b[0]
                part_end = np.minimum(s_stop, b[1]) - b[0]
                for i in np.ndindex(blocks.shape[1:]):
                    blocks[0, *i] = blocks[0, *i][ax_(d)[part_start:part_end]]
            if len(b) > 2 and s_stop < b[-1]:  # clip last block (should not also be the first (and only) block)
                part_start = 0
                part_end = s_stop - b[-2]
                for i in np.ndindex(blocks.shape[1:]):
                    blocks[-1, *i] = blocks[-1, *i][ax_(d)[part_start:part_end]]

            # proceed to the next dimension
            blocks = np.moveaxis(blocks, 0, -1)

        return BlockArray.new(blocks)

    @staticmethod
    def _verify_pre_split_data(blocks: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], np.dtype]:
        """Computes boundaries and shape for the pre-split array.

        Also verifies that all blocks have the same dimensionality, data type, and that all blocks form a regular grid
        Returns internal boundaries, shape and dtype
        """
        ndim = blocks.ndim
        dtype = blocks.flat[0].dtype

        # get the shapes for all blocks
        shapes = np.empty((*blocks.shape, ndim), dtype=int)
        for block_index, block in np.ndenumerate(blocks):
            shapes[block_index] = block.shape
            if block.dtype != dtype:
                raise ValueError(
                    f"Blocks must all have the same dtype {dtype},"
                    f"found block at {block_index} with dtype {block.dtype}"
                )
            if block.ndim != ndim:
                raise ValueError(
                    f"Blocks must all have the same number of dimensions {ndim},"
                    f"found block at {block_index} with {block.ndim} dimensions"
                )

        # verify that along dimension 'd', the blocks are aligned with the boundaries
        boundaries = []
        for d in range(ndim):
            std = np.std(shapes, axis=d)
            std[..., d] = 0.0
            if np.any(std):
                raise ValueError("Block shapes are not aligned with the boundaries")

            shape_along_dim = shapes[*((0,) * d + (slice(None),) + (0,) * (ndim - d - 1)), d]
            boundaries.append(tuple(int(b) for b in np.cumsum(shape_along_dim)))

        return tuple(b[:-1] for b in boundaries), tuple(b[-1] for b in boundaries), dtype

    @staticmethod
    def new(blocks: NDArray[Array]) -> Array:
        """Create a new BlockArray from an array of blocks.

        The data is not copied.
        If there is only one block, the block is returned directly.

        Args:
            blocks: The blocks to use for the new BlockArray.
        """
        if blocks.size == 1:
            return blocks.item()
        return BlockArray(blocks)


@dispatch
def mix(
    alpha: scalar,
    a: BlockArray | Array | scalar,
    beta: scalar,
    b: BlockArray | Array | scalar,
    /,
    *,
    out: BlockArray | Array,
):
    """Computes out->α·a + β·b."""
    for a_, b_, out_ in blocked((a, b, out)):
        mix(alpha, a_, beta, b_, out=out_)


@dispatch
def add(
    a: BlockArray | Array | scalar,
    b: BlockArray | Array | scalar,
    /,
    *,
    out: BlockArray | Array,
):
    """Computes out->a + b."""
    for a_, b_, out_ in blocked((a, b, out)):
        add(a_, b_, out=out_)


@dispatch
def subtract(
    a: BlockArray | Array | scalar,
    b: BlockArray | Array | scalar,
    /,
    *,
    out: BlockArray | Array,
):
    """Computes out->a - b."""
    for a_, b_, out_ in blocked((a, b, out)):
        subtract(a_, b_, out=out_)


@dispatch
def lerp(
    a: BlockArray | Array | scalar,
    b: BlockArray | Array,
    weight: BlockArray | Array | scalar,
    /,
    *,
    out: BlockArray | Array,
):
    """Computes out = (1-weight)·a + weight·b = a + weight · (b-a)"""
    for a_, b_, w_, out_ in blocked((a, b, weight, out)):
        lerp(a_, b_, w_, out=out_)


@dispatch
def multiply(a: BlockArray | Array | scalar, b: BlockArray | Array | scalar, /, *, out: BlockArray | Array):
    for a_, b_, out_ in blocked((a, b, out)):
        multiply(a_, b_, out=out_)


@dispatch
def divide(a: BlockArray | Array | scalar, b: BlockArray | Array | scalar, /, *, out: BlockArray | Array):
    for a_, b_, out_ in blocked((a, b, out)):
        divide(a_, b_, out=out_)


@dispatch
def inner_product(a: BlockArray | Array | scalar, b: BlockArray | Array | scalar, /) -> complex:
    inner = 0.0
    for a_, b_ in blocked((a, b)):
        inner += inner_product(a_, b_)

    return inner.real if a is b else inner  # remove small imaginary part if present


@dispatch
def norm_squared(a: BlockArray, /) -> complex:
    norm2 = 0.0
    for a_ in a.blocks.flat:
        norm2 += norm_squared(a_)

    return norm2


@dispatch
def copy(value: Array, /, *, out: BlockArray):
    """Copies data from another array into this BlockArray."""
    for value_, out_ in blocked((value, out)):
        copy(value_, out=out_)


@dispatch
def copy(value: np.ndarray, /, *, out: BlockArray):
    """Copies data from a numpy array into this Array."""
    value = NumpyArray(value)
    for value_, out_ in blocked((value, out)):
        copy(value_, out=out_)


@dispatch
def copy(value: scalar, /, *, out: BlockArray):
    """Copies data from a numpy array into this Array."""
    for (out_,) in blocked((out,)):
        copy(value, out=out_)


@dispatch
def matmul(matrix: Array, x: BlockArray, /, *, axis: int, out: BlockArray):
    if isinstance(matrix, BlockArray) or not matrix.is_full:
        raise NotImplementedError("Matrix must be a full array")
    for x_, out_ in blocked((x, out)):
        matmul(matrix, x_, axis=axis, out=out_)


def blocked(args: tuple[Array, ...]) -> Generator[tuple[Array | scalar, ...], None, None]:
    """An iterator for block-wise iteration over a set of BlockArrays.

    The blocks are defined such that they are all within one subblock for all arrays,
    i.e. the slices do not cross a subblock boundary for any of the input arrays
    """
    block_arrays = [a for a in args if isinstance(a, BlockArray)]
    ndim = len(block_arrays[0].boundaries)

    # determine the split points along all axes
    positions = []
    for d in range(ndim):
        p = {0}  # 0 is always a split point
        for a in block_arrays:
            p.update(a.boundaries[d])  # all internal boundaries are split points
            if a.shape[d] != 1:  # the end of the array is a split point if the axis is not of length 1 (broadcast)
                p.add(a.shape[d])
        if len(p) == 1:  # no split points or end points found, this means all axes are broadcast (all length 1)
            p.add(np.max([a.shape[d] for a in args if isinstance(a, Array)]))
        positions.append(tuple(sorted(p)))

    # iterate over all subblocks, return respective slices for each input
    sub_block_count = tuple(len(p) - 1 for p in positions)
    for sub_block_index in np.ndindex(sub_block_count):
        start = np.array([positions[d][i] for d, i in enumerate(sub_block_index)])
        stop = np.array([positions[d][i + 1] for d, i in enumerate(sub_block_index)])
        yield tuple(a.slice(start, stop) if isinstance(a, Array) else a for a in args)


def _as_ndarray(x):
    """Convert a BlockArray to a numpy ndarray of objects.

    Also accepts Arrays, which are treated as BlockArrays with one block.
    """
    if isinstance(x, BlockArray):
        return x.blocks
    elif isinstance(x, Array):
        return np.array(x, dtype=object).reshape((1,) * x.ndim)
    else:
        return x


def block_iter(*args):
    # arguments are BlockArray, Array or NDArray[Array]
    args = [_as_ndarray(x) for x in args]
    for idx in np.ndindex(args[0].shape):
        if len(args) == 1:
            yield args[0][idx]
        else:
            yield tuple(a[idx] for a in args)


def block_enumerate(*args):
    # arguments are BlockArray, Array or NDArray[Array]
    args = [_as_ndarray(x) for x in args]
    for idx in np.ndindex(args[0].shape):
        yield idx, *tuple(a[idx] for a in args)


def block_map(x: Array | NDArray[Array], func: callable) -> np.ndarray:
    """Map a function over all blocks of a BlockArray
    Args:
        x: The BlockArray. An Array is treated as a BlockArray with 1 block
        func: The function to apply to each block
    Returns:
        np.ndarray of objects with the results.
    """
    x = _as_ndarray(x)
    blocks = np.empty_like(x, dtype=object)
    for idx, block in np.ndenumerate(x):
        blocks[idx] = func(block)

    return blocks


def block_shape(x: Array) -> tuple[int, ...]:
    """Returns the number of blocks in each dimension of the BlockArray

    Args:
        x: The BlockArray. An Array is treated as a BlockArray with one block
    Returns:
        tuple with the number of blocks in each dimension
    """
    return _as_ndarray(x).shape
