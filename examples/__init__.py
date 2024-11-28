import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.fft import fftn, ifftn, fftshift
from wavesim.utilities import normalize, relative_error


def random_permittivity(shape):
    np.random.seed(0)  # Set the random seed for reproducibility
    n = (1.0 + np.random.rand(*shape).astype(np.float32) +
         0.03j * np.random.rand(*shape).astype(np.float32))  # Random refractive index
    n = smooth_n(n, shape)  # Low pass filter to remove sharp edges
    n.real = normalize(n.real, a=1.0, b=2.0)  # Normalize to [1, 2]
    n.imag = normalize(n.imag, a=0.0, b=0.03)  # Normalize to [0, 0.03]
    # make sure that the imaginary part of n² is positive
    mask = (n ** 2).imag < 0
    n.imag[mask] *= -1.0

    n[0:5, :, :] = 1
    n[-5:, :, :] = 1
    return n ** 2


def smooth_n(n, shape):
    """Low pass filter to remove sharp edges"""
    n_fft = fftn(n)
    w = (window(shape[1]).T @ window(shape[0])).T[:, :, None] * window(shape[2]).reshape(1, 1, shape[2])
    n = ifftn(n_fft * fftshift(w))
    n = np.clip(n.real, a_min=1.0, a_max=None) + 1.0j * np.clip(n.imag, a_min=0.0, a_max=None)

    assert (n ** 2).imag.min() >= 0, 'Imaginary part of n² is negative'
    assert n.shape == shape, 'n and shape do not match'
    assert n.dtype == np.complex64, 'n is not complex64'
    return n


def window(x):
    """Create a window function for low pass filtering"""
    c0 = round(x / 4)
    cl = (x - c0) // 2
    cr = cl
    if c0 + cl + cr != x:
        c0 = x - cl - cr
    return np.concatenate((np.zeros((1, cl), dtype=np.complex64), 
                           np.ones((1, c0), dtype=np.complex64), 
                           np.zeros((1, cr), dtype=np.complex64)), axis=1)


def construct_source(source_type, at, shape):
    if source_type == 'point':
        return torch.sparse_coo_tensor(at[:, None], torch.tensor([1.0]), shape, dtype=torch.complex64)
    elif source_type == 'plane_wave':
        return source_plane_wave(at, shape)
    elif source_type == 'gaussian_beam':
        return source_gaussian_beam(at, shape)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def source_plane_wave(at, shape):
    """ Set up source, with size same as permittivity, 
        and a plane wave source on one edge of the domain """
    # TODO: use CSR format instead?
    data = np.ones((1, shape[1], shape[2]), dtype=np.float32)  # the source itself
    return torch.sparse_coo_tensor(at, data, shape, dtype=torch.complex64)


def source_gaussian_beam(at, shape):
    """ Set up source, with size same as permittivity, 
        and a Gaussian beam source on one edge of the domain """
    # TODO: use CSR format instead?
    std = (shape[1] - 1) / (2 * 3)
    source_amplitude = gaussian(shape[1], std).astype(np.float32)
    source_amplitude = np.outer(gaussian(shape[2], std).astype(np.float32), 
                                source_amplitude.astype(np.float32))
    source_amplitude = torch.tensor(source_amplitude[None, ...])
    data = torch.zeros((1, shape[1], shape[2]), dtype=torch.complex64)
    data[0, 
           0:shape[1], 
           0:shape[2]] = source_amplitude
    return torch.sparse_coo_tensor(at, data, shape, dtype=torch.complex64)


def plot(x, x_ref, re=None, normalize_x=True):
    """Plot the computed field x and the reference field x_ref.
    If x and x_ref are 1D arrays, the real and imaginary parts are plotted separately.
    If x and x_ref are 2D arrays, the absolute values are plotted.
    If x and x_ref are 3D arrays, the central slice is plotted.
    If normalize_x is True, the values are normalized to the same range.
    The relative error is (computed, if needed, and) displayed.
    """

    re = relative_error(x, x_ref) if re is None else re

    if x.ndim == 1 and x_ref.ndim == 1:
        plt.subplot(211)
        plt.plot(x_ref.real, label='Analytic')
        plt.plot(x.real, label='Computed')
        plt.legend()
        plt.title(f'Real part (RE = {relative_error(x.real, x_ref.real):.2e})')
        plt.grid()

        plt.subplot(212)
        plt.plot(x_ref.imag, label='Analytic')
        plt.plot(x.imag, label='Computed')
        plt.legend()
        plt.title(f'Imaginary part (RE = {relative_error(x.imag, x_ref.imag):.2e})')
        plt.grid()

        plt.suptitle(f'Relative error (RE) = {re:.2e}')
        plt.tight_layout()
        plt.show()
    else:
        if x.ndim == 3 and x_ref.ndim == 3:
            x = x[x.shape[0]//2, ...]
            x_ref = x_ref[x_ref.shape[0]//2, ...]

        x = np.abs(x)
        x_ref = np.abs(x_ref)
        if normalize_x:
            min_val = min(np.min(x), np.min(x_ref))
            max_val = max(np.max(x), np.max(x_ref))
            a = 0
            b = 1
            x = normalize(x, min_val, max_val, a, b)
            x_ref = normalize(x_ref, min_val, max_val, a, b)
        else:
            a = None
            b = None

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(x_ref, cmap='hot_r', vmin=a, vmax=b)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Reference')

        plt.subplot(122)
        plt.imshow(x, cmap='hot_r', vmin=a, vmax=b)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Computed')

        plt.suptitle(f'Relative error (RE) = {re:.2e}')
        plt.tight_layout()
        plt.show()
