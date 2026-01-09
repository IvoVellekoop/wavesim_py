from typing import Iterable, SupportsIndex, Sequence

import numpy as np

shape_like = Sequence[int]
slice_entry = slice | SupportsIndex
indexing_type = "ax_", slice_entry | Iterable[slice_entry]


class ax_:  # noqa. A special class name is ok for this special class
    """A placeholder for the axis argument in functions that operate along an axis.

    This object can be used in [] indexing to specify the axis along which an operation should be performed.
    The ax_ object takes one or more axes as argument, followed by an indexing expression along those axes.
    All other axes in the indexing expression are treated as ``:``
    For example:

        data[ax_(3, 6)[5:6, :]]

    is equivalent to:

        data[:, :, :, 5:6, :, :, 0, ...]
    """

    def __init__(self, *axes):
        self.axes = tuple(axes)
        self.slices = None

    def __getitem__(self, slices: indexing_type):
        self.slices = slices
        return self

    def generate(self, ndim: int):
        all_slices = ()
        if not isinstance(self.slices, Iterable):
            self.slices = (self.slices,)
        for a, s in zip(self.axes, self.slices):
            # append ':' slices until reaching the axis
            all_slices += (slice(None),) * (a - len(all_slices))
            # append the stored slice
            all_slices += (s,)

        # append ':' slices for the remaining axes
        all_slices += (slice(None),) * (ndim - len(all_slices))
        return all_slices


def roi_start(slices: tuple[slice, ...]) -> tuple[int, ...]:
    """Get the start positions of the slices in a tuple of slices.

    Args:
        slices: a single integer or slice, a tuple of integer/slice objects, or an ax_ object.
    Returns:
        tuple of start positions for each axis
    """
    return tuple(s.start for s in slices)


def slices_to_pos(slices: indexing_type, shape: shape_like) -> tuple[np.array, np.array]:
    """Convert a tuple of slices to start and end position

    This function processes integer indices, slices with positive, negative and None entries, and ax_ objects.
    Note that integer indices are converted to size-1 slices,
    which differs from the numpy behavior (which collapses the axis for integer indices)

    Args:
        slices: a single integer or slice, a tuple of integer/slice objects, or an ax_ object.
        shape: shape of the array, used to convert negative indices to positions counted from the end of the array
    Returns:
        tuple of start position (inclusive) and end position (exclusive)
    Raises:
        ValueError: if the number of slices does not match the number of dimensions of the array
        IndexError: if the slice is out of bounds for the array
    """
    ndim = len(shape)
    if isinstance(slices, ax_):
        slices = slices.generate(ndim)
    elif not isinstance(slices, Iterable):
        slices = (slices,)
    if len(slices) != ndim:
        raise ValueError(f"The number of slices should match the dimensionality of the array {ndim}")

    slice_start = np.zeros(ndim, dtype=int)
    slice_stop = np.ones(ndim, dtype=int)
    for d, s in enumerate(slices):
        if shape[d] == 1:
            # Special case for broadcasting that allows us to select a region in a broadcast array.
            # Regardless of the slice, always return a 1-element slice (which can be broadcast to any size)
            # unless the slice size is 0, which means we should return an empty slice.
            zero_size_slice = isinstance(s, slice) and (s.start or 0) == s.stop  # handle s.start=None as s.start=0
            slice_start[d] = 0
            slice_stop[d] = 0 if zero_size_slice else 1
            continue
        if isinstance(s, slice):
            if s.step is not None and s.step != 1:
                raise NotImplementedError("Slices with step != 1 are not supported")
            slice_start[d] = _process_index(s.start, shape=shape[d], default=0)
            slice_stop[d] = _process_index(s.stop, shape=shape[d], default=shape[d])
            if slice_stop[d] < slice_start[d]:
                raise IndexError(f"Invalid slice {s} for shape {shape}")
        else:  # int
            slice_start[d] = _process_index(s, shape=shape[d], default=shape[d])
            slice_stop[d] = slice_start[d] + 1
        if slice_start[d] < 0 or slice_stop[d] > shape[d] + 1:
            raise IndexError(f"Slice {s} out of bounds for shape {shape}")

    return slice_start, slice_stop


def _process_index(index: SupportsIndex | None, shape: int, default: int) -> int:
    if index is None:
        return default
    ii = int(index)
    if ii < 0:
        ii += shape
    return ii


def pos_to_slices(start: shape_like | int, stop: shape_like) -> tuple[slice, ...]:
    """Create a tuple of slices that selects a block of an array.

    Args:
        start: The start position of the block, use '0' as shorthand for ``(0,) * len(stop)``
        stop: The end position of the block.
    """
    if isinstance(start, int) and start == 0:
        start = (0,) * len(stop)
    if len(start) != len(stop):
        raise ValueError(f"Start and stop must have the same length, got {len(start)} and {len(stop)}")
    return tuple(slice(f, l) for f, l in zip(start, stop))
