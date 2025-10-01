from typing import Sequence

import numpy as np
from wavesim.engine.array import Array, lerp, as_complex, new_like
from wavesim.engine.functions import pad, edges, boundary_spec


def add_absorbing_boundaries(
    permittivity: Array, boundary_widths: boundary_spec, strength: float, *, periodic: Sequence[bool] | None = None
) -> tuple[Array, tuple[slice, ...]]:
    """Pads the permittivity array with absorbing boundaries.

    Args:
        permittivity: The permittivity array
        strength: Strength of the absorbing boundaries.
            This is the largest imaginary part of the curve that is added to the permittivity
        boundary_widths: Width of the absorbing boundaries as an array of shape (ndim, 2) or a
            tuple of tuples (boundary_left, boundary_right), or a scalar to indicate all boundaries have the same width
        periodic: Boolean for each axis, if True the boundary is periodic and no absorbing boundary is added.
            This is especially useful when passing an integer value as boundary widths.

    Returns:
        tuple of
        - permittivity: The permittivity array with absorbing boundaries added.
        - roi: A tuple of slices that can be used to extract the original permittivity array from the padded array.
    """
    ndim = permittivity.ndim
    permittivity = as_complex(permittivity)  # ensure permittivity has a complex data type
    if ndim != 3:
        raise ValueError("The permittivity must be a 3-dimensional array")
    if np.isscalar(boundary_widths):
        boundary_widths = np.full((ndim, 2), boundary_widths)
    else:
        boundary_widths = np.asarray(boundary_widths)
        if boundary_widths.shape != (ndim, 2) and boundary_widths.shape != (ndim,):
            raise ValueError(
                "The number of boundary widths must match the number of dimensions of the permittivity array"
            )
        if boundary_widths.shape == (ndim,):
            boundary_widths = np.repeat(boundary_widths[...,None], 2, axis=1)
    if periodic is not None:
        boundary_widths[np.asarray(periodic), :] = 0  # set periodic boundaries to zero width

    # pad the permittivity array (repeating the values at the edges)
    # then add a linear-ramp absorption profile to each of the boundaries
    permittivity = pad(permittivity, boundary_widths, mode="edge", block_array=False)
    boundaries = edges(permittivity, boundary_widths)

    for (d, side), edge in np.ndenumerate(boundaries):
        # linear ramp absorption profile
        # this interpolates between the original permittivity and 1.0j * strength
        width = boundary_widths[d, side]
        if width == 0:
            continue
        profile = (np.arange(1, width + 1) - 0.21) / (width + 0.66)  # this gives 1 - weight
        profile = new_like(permittivity, profile if side == 1 else np.flip(profile))
        profile = profile.transpose(0, to=d, ndim=ndim)  # align array along d axis, broadcast along other axes
        lerp(edge, 1.0j * strength, profile, out=edge)

    return permittivity, tuple(slice(bw[0], -bw[1] if bw[1] > 0 else None) for bw in boundary_widths)


def normalize(x: np.ndarray, min_val: float = None, max_val: float = None, a: float = 0, b: float = 1) -> float:
    """Normalize x to the range [a, b]

    :param x: Input array
    :param min_val: Minimum value (of x)
    :param max_val: Maximum value (of x)
    :param a: Lower bound for normalization
    :param b: Upper bound for normalization
    :return: Normalized x
    """
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    normalized_x = (x - min_val) / (max_val - min_val) * (b - a) + a
    return normalized_x


def laplace_kernel_1d(pixel_size: float, length: int) -> np.ndarray:
    """Compute the kernel for the Laplace operator

    This function computes the exact Laplace kernel that corresponds to the second derivative of a sinc basis function,
    and then truncates it to a finite length. The truncation is done in such a way that the kernel still has
    an average value of zero, meaning that operating on a constant function will yield zero.

    Returns:
        The kernel for the Laplace operator in real space.
        The elements at negative positions are wrapped to the end of the array.
    """

    # original way (introduces wrapping artifacts in the kernel)
    # return -self.coordinates_f(dim) ** 2

    if length == 1:
        return np.zeros((1,))

    # x = [0, π, 2π .... -2π, -π]
    x = np.fft.fftfreq(length, 1 / (np.pi * length))

    # x_kernel = 2.0 * c / x**2 - 2.0 * s / x**3 + s / x
    # with all s=sin(x)=0
    x[0] = 1  # prevent division by 0
    x_kernel = 2.0 * np.cos(x) / x**2
    x_kernel[0] = 1.0 / 3.0  # remove singularity at x=0
    x_kernel *= -np.pi**2 / pixel_size**2

    # adjust end point(s) to ensure the kernel has an average value of zero
    # todo: find a more elegant way to do this
    if length % 2 == 0:
        x_kernel[length // 2] -= np.sum(x_kernel)  # even length, adjust only the furthest end point
    else:
        x_kernel[length // 2 : length // 2 + 1] -= 0.5 * np.sum(x_kernel)  # odd length, adjust the two end points

    return x_kernel


def diff_kernel_1d(pixel_size: float, length: int) -> np.ndarray:
    """Compute the kernel for the first derivative operator

    Returns:
        The kernel for the first derivative operator in real space.
        The elements at negative positions are wrapped to the end of the array.
    """
    x = np.fft.ifftshift(np.arange(-np.floor(length / 2), np.ceil(length / 2), dtype=np.float64)) * np.pi
    if x.size == 1:
        return x  # = [0.0]
    x[0] = 1  # prevent division by 0 warning
    x_kernel = np.pi / pixel_size * np.cos(x) / x
    x_kernel[0] = 0  # remove singularity at x=0
    return x_kernel
