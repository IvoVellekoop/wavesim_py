import numpy as np
import matplotlib.pyplot as plt
from wavesim.utilities import normalize, relative_error


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
