import numpy as np
from typing import Sequence
from scipy.fft import fftn, ifftn, fftshift

from wavesim.engine.index_utils import shape_like
from wavesim.engine import Array
from .utilities import normalize


def random_permittivity(shape: shape_like, pixel_size: float, seed = 0) -> Array:
    """Construct a random permittivity (n^2) between 1 and 2 with a small positive imaginary part 
    (between 0 and 0.03), and a refractive index of 1 in the first and last 5 pixels of the x axis"""
    # convert shape from micrometers to pixels
    shape = tuple((np.asarray(shape) / pixel_size).astype(int))

    np.random.seed(seed)  # Set the random seed for reproducibility
    n = (
        1.0 + np.random.rand(*shape).astype(np.float32) + 0.03j * np.random.rand(*shape).astype(np.float32)
    )  # Random refractive index
    n = smooth_n(n, shape)  # Low pass filter to remove sharp edges
    n.real = normalize(n.real, a=1.0, b=2.0)  # Normalize to [1, 2]
    n.imag = normalize(n.imag, a=0.0, b=0.03)  # Normalize to [0, 0.03]
    # make sure that the imaginary part of n² is positive
    mask = (n**2).imag < 0
    n.imag[mask] *= -1.0

    n[0:5, :, :] = 1
    n[-5:, :, :] = 1
    return n**2


def smooth_n(n, shape):
    """Low pass filter to remove sharp edges"""
    n_fft = fftn(n)
    w = (window(shape[1]).T @ window(shape[0])).T[:, :, None] * window(shape[2]).reshape(1, 1, shape[2])
    n = ifftn(n_fft * fftshift(w))
    n = np.clip(n.real, a_min=1.0, a_max=None) + 1.0j * np.clip(n.imag, a_min=0.0, a_max=None)

    assert (n**2).imag.min() >= 0, "Imaginary part of n² is negative"
    assert n.shape == shape, "n and shape do not match"
    assert n.dtype == np.complex64, "n is not complex64"
    return n


def window(x):
    """Create a window function for low pass filtering"""
    c0 = round(x / 4)
    cl = (x - c0) // 2
    cr = cl
    if c0 + cl + cr != x:
        c0 = x - cl - cr
    return np.concatenate(
        (
            np.zeros((1, cl), dtype=np.complex64),
            np.ones((1, c0), dtype=np.complex64),
            np.zeros((1, cr), dtype=np.complex64),
        ),
        axis=1,
    )


def cuboids_permittivity(
    shape: shape_like,  # Shape of the simulation domain in micrometer (μm)
    pixel_size: float,
    origin: str,
    origin_size_material: Sequence[
        tuple[Sequence[float], Sequence[float], np.complex64]
        ] = None,  # sequence of tuples (originN, sizeN, material) for each shape
    background: np.complex64 = np.complex64(1 + 0j),  # background refractive index value
) -> np.ndarray:
    """
    Create a NumPy array of complex64 with shape dimensions to define the permittivity distribution of the model.
    The whole area is filled with the background refractive index value, and the specified regions are filled with
    the refractive index values of the materials. Ensures that the origin_size_material do not overlap. Handles 1D, 2D, and 3D shapes.

    Args:
        shape (Sequence[int]): Shape of the array (1D, 2D, or 3D).
        pixel_size: pixel size in micrometer (μm)
        origin (str): Define the permittivity blocks with origin as 'origin' or 'topleft.'
        origin_size_material (Sequence[tuple[Sequence[float], Sequence[float], np.complex64]]):
            Sequence of tuples where each tuple defines (originN, sizeN, material).
        background (np.complex64): The refractive index value of the background.

    Returns:
        np.ndarray: Array with permittivity distribution.
    """
    if origin not in ['center', 'topleft']:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")

    # convert shape from micrometers to pixels
    shape = (np.asarray(shape) / pixel_size).astype(int)

    # Create a NumPy array of complex64 with shape dimensions, filled with the background value
    model = np.full(shape, fill_value=background, dtype=np.complex64)

    # Create a mask to track filled regions
    filled_mask = np.zeros(shape, dtype=bool)

    for originN, sizeN, material in origin_size_material:
        # Calculate the origin_ and size_ in pixels
        origin_ = (np.asarray(originN) / pixel_size).astype(int)
        size_ = (np.asarray(sizeN) / pixel_size).astype(int)

        # Calculate the start and end indices for the filled region
        if origin == 'center':
            start = [max(origin_[i] - size_[i] // 2, 0) for i in range(len(shape))]
            end = [min(start[i] + size_[i], shape[i]) for i in range(len(shape))]
        elif origin == 'topleft':
            start = [max(origin_[i], 0) for i in range(len(shape))]
            end = [min(start[i] + size_[i], shape[i]) for i in range(len(shape))]

        # Ensure indices are within bounds
        start = [max(0, start[i]) for i in range(len(shape))]
        end = [min(shape[i], end[i]) for i in range(len(shape))]

        # Check for overlap with already filled regions
        slices = tuple(slice(start[i], end[i]) for i in range(len(shape)))
        if np.any(filled_mask[slices]):
            raise ValueError(f"origin_size_material in the sequence overlap. Please ensure non-overlapping origin_size_material.")

        # Fill the specified region with the material value
        model[slices] = material

        # Update the filled mask
        filled_mask[slices] = True

    # Square the refractive index to get the permittivity
    model = model**2
    return model


def sphere_permittivity(n_size, pixel_size, sphere_radius, sphere_index, bg_index, center=None):
    """
    Returns a 3-D matrix of refractive indices for single sphere and the coordinate ranges.
    The refractive index will be sphere_index inside a sphere and bg_index outside the sphere,
    with radius sphere_radius (in micrometer) and pixel_size (in micrometer).

    Args:
        n_size: Number of points in each dimension (Nx, Ny, Nz)
        pixel_size: Pixel size in micrometer (μm)
        sphere_radius: Sphere radius in micrometer (μm)
        sphere_index: Refractive index of the sphere
        bg_index: Background refractive index
        center: Center of the sphere (default: [0, 0, 0])
    """

    center = [0, 0, 0] if center is None else center

    # Calculate coordinate ranges
    center = (np.array(n_size) / 2) * pixel_size + center
    x_range = np.reshape((np.arange(1, n_size[1] + 1) * pixel_size - center[0]), (1, n_size[1], 1)).astype(np.float32)
    y_range = np.reshape((np.arange(1, n_size[0] + 1) * pixel_size - center[1]), (n_size[0], 1, 1)).astype(np.float32)
    z_range = np.reshape((np.arange(1, n_size[2] + 1) * pixel_size - center[2]), (1, 1, n_size[2])).astype(np.float32)

    # Calculate refractive index
    inside = (x_range**2 + y_range**2 + z_range**2) < sphere_radius**2
    n = np.full(n_size, bg_index) + inside * (sphere_index - bg_index)

    return n.astype(np.complex64), x_range, y_range, z_range
