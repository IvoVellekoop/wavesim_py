import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from utilities import relative_error

def plot(a, b, re=None):
    if re is None:
        re = relative_error(a, b)
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


