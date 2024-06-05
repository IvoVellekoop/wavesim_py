import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from utilities import relative_error

def plot(a, b, re=None):
    if re is None:
        re = relative_error(a, b)
    
    if a.ndim == 1 and b.ndim == 1:
        plt.subplot(211)
        plt.plot(a.real, label='Computed')
        plt.plot(b.real, label='Analytic')
        plt.legend()
        plt.title(f'Real part (RE = {relative_error(a.real, b.real):.2e})')
        plt.grid()

        plt.subplot(212)
        plt.plot(a.imag, label='Computed')
        plt.plot(b.imag, label='Analytic')
        plt.legend()
        plt.title(f'Imaginary part (RE = {relative_error(a.imag, b.imag):.2e})')
        plt.grid()

        plt.suptitle(f'Relative error (RE) = {re:.2e}')
        plt.tight_layout()
        plt.show()

    if a.ndim == 2 and b.ndim == 2:
        a = np.abs(a)
        b = np.abs(b)
        vmin = min(np.min(a), np.min(b))
        vmax = max(np.max(a), np.max(b))

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(a, cmap='hot_r', vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Reference')

        plt.subplot(122)
        plt.imshow(b, cmap='hot_r', vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Computed')

        plt.suptitle(f'Relative error (RE) = {re:.2e}')
        plt.tight_layout()
        plt.show()

