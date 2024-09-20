import numpy as np
from scipy.special import exp1
import matplotlib.pyplot as plt
from wavesim.utilities import normalize, relative_error


def analytical_solution(n_size0, pixel_size, wavelength=None):
    """ Compute analytic solution for 1D case """
    x = np.arange(0, n_size0 * pixel_size, pixel_size, dtype=np.float32)
    x = np.pad(x, (n_size0, n_size0), mode='constant', constant_values=np.nan)
    h = pixel_size
    # wavenumber (k)
    if wavelength is None:
        k = 1. * 2. * np.pi * pixel_size
    else:
        k = 1. * 2. * np.pi / wavelength
    phi = k * x
    u_theory = (1.0j * h / (2 * k) * np.exp(1.0j * phi)  # propagating plane wave
                - h / (4 * np.pi * k) * (
        np.exp(1.0j * phi) * (exp1(1.0j * (k - np.pi / h) * x) - exp1(1.0j * (k + np.pi / h) * x)) -
        np.exp(-1.0j * phi) * (-exp1(-1.0j * (k - np.pi / h) * x) + exp1(-1.0j * (k + np.pi / h) * x)))
    )
    small = np.abs(k * x) < 1.e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory[n_size0:-n_size0]


def plot(x, x_ref, re=None):
    if re is None:
        re = relative_error(x, x_ref)

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
            x = x[:, :, x.shape[2]//2]
            x_ref = x_ref[:, :, x_ref.shape[2]//2]

        x = np.abs(x)
        x_ref = np.abs(x_ref)
        min_val = min(np.min(x), np.min(x_ref))
        max_val = max(np.max(x), np.max(x_ref))

        a = 0
        b = 1
        x = normalize(x, min_val, max_val, a, b)
        x_ref = normalize(x_ref, min_val, max_val, a, b)

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
