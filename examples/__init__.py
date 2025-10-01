import os
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from time import localtime, strftime

import sys
sys.path.append(".")
from tests import relative_error
from wavesim.utilities import normalize

# Copy these lines to the top of your script in case there is an error due to
# the current working directory being "examples" (happens in some IDEs)
import os
if os.path.basename(os.getcwd()) == "examples":
    os.chdir("..")
    os.makedirs('logs', exist_ok=True)
else:
    try:
        os.makedirs('logs', exist_ok=True)
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'examples' directory.")


def plot_computed(u, pixel_size, normalize_u=True, cmap="inferno", log=False, show=True, save=False, filename=None):
    date_time = strftime("%Y%m%d_%H%M%S", localtime())
    if u.ndim == 1:
        u_range = np.arange(u.shape[0]) * pixel_size
        plt.subplot(211)
        plt.plot(u_range, u.real)
        plt.xlabel(r"$x~(\mu m)$")
        plt.title("Real part")
        plt.grid()

        plt.subplot(212)
        plt.plot(u_range, u.imag)
        plt.xlabel(r"$x~(\mu m)$")
        plt.title("Imaginary part")
        plt.grid()

        plt.tight_layout()
        if filename is None:
            filename = f'logs/fig1d_{date_time}'
    
    else:
        if u.ndim == 4:
            u = u[0, ...]  # x-polarization

        if u.ndim == 3:
            u = u[:, :, u.shape[2] // 2]

        u = np.abs(u)

        if normalize_u:
            min_val = np.min(u)
            u = normalize(u, a=1e-10 if min_val == 0 else 0.01 * min_val)

        extent = np.array([0, u.shape[0], u.shape[1], 0]) * pixel_size
        plt.imshow(u.T, cmap=cmap, extent=extent, norm=colors.LogNorm() if log else None)
        plt.xlabel(r"$x~(\mu m)$")
        plt.ylabel(r"$y~(\mu m)$")
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_title(r"$|E|$")
        plt.tight_layout()

        if filename is None:
            filename = f'logs/fig2d_{date_time}'

    plt.savefig(f'{filename}.png', bbox_inches='tight', pad_inches=0.03, dpi=300) if save else None
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300) if save else None
    plt.show() if show else None
    plt.close('all')


def plot_computed_and_reference(u, u_ref, pixel_size, re=None, normalize_u=True, cmap="inferno", log=False, show=True, save=False, filename=None):
    """Plot the computed field u and the reference field u_ref.
    If u and u_ref are 1D arrays, the real and imaginary parts are plotted separately.
    If u and u_ref are 2D arrays, the absolute values are plotted.
    If u and u_ref are 3D arrays, the central slice is plotted.
    If normalize_u is True, the values are normalized to the same range.
    The relative error is (computed, if needed, and) displayed.
    """

    re = relative_error(u, u_ref) if re is None else re
    u_range = np.arange(u.shape[0]) * pixel_size
    date_time = strftime("%Y%m%d_%H%M%S", localtime())

    if u.ndim == 1 and u_ref.ndim == 1:
        plt.subplot(211)
        plt.plot(u_range, u_ref.real, label="Reference")
        plt.plot(u_range, u.real, label="Computed")
        plt.plot(u_range, u_ref.real - u.real, label="Difference")
        plt.xlabel(r"$x~(\mu m)$")
        plt.legend()
        plt.title(f"Real part (RE = {relative_error(u.real, u_ref.real):.2e})")
        plt.grid()

        plt.subplot(212)
        plt.plot(u_range, u_ref.imag, label="Reference")
        plt.plot(u_range, u.imag, label="Computed")
        plt.plot(u_range, u_ref.imag - u.imag, label="Difference")
        plt.xlabel(r"$x~(\mu m)$")
        plt.legend()
        plt.title(f"Imaginary part (RE = {relative_error(u.imag, u_ref.imag):.2e})")
        plt.grid()

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()

        if filename is None:
            filename = f'logs/fig1d_{date_time}'

    else:
        if u.ndim == 4 and u_ref.ndim == 4:
            u = u[0, ...]
            u_ref = u_ref[0, ...]

        if u.ndim == 3 and u_ref.ndim == 3:
            u = u[u.shape[0] // 2, ...]
            u_ref = u_ref[u_ref.shape[0] // 2, ...]

        u = np.abs(u)
        u_ref = np.abs(u_ref)
        if normalize_u:
            min_val = min(np.min(u), np.min(u_ref))
            max_val = max(np.max(u), np.max(u_ref))
            a = 1e-10 if min_val == 0 else 0.01 * min_val
            b = 1
            u = normalize(u, min_val, max_val, a, b)
            u_ref = normalize(u_ref, min_val, max_val, a, b)
        else:
            a = None
            b = None

        extent = np.array([0, u.shape[0], u.shape[1], 0]) * pixel_size

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        if log:
            plt.imshow(u_ref, cmap=cmap, norm=colors.LogNorm(vmin=a, vmax=b), extent=extent)
        else:
            plt.imshow(u_ref, cmap=cmap, vmin=a, vmax=b, extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel(r"$x~(\mu m)$")
        plt.ylabel(r"$y~(\mu m)$")
        plt.title("Reference")

        plt.subplot(122)
        if log:
            plt.imshow(u, cmap=cmap, norm=colors.LogNorm(vmin=a, vmax=b), extent=extent)
        else:
            plt.imshow(u, cmap=cmap, vmin=a, vmax=b, extent=extent)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel(r"$x~(\mu m)$")
        plt.ylabel(r"$y~(\mu m)$")
        plt.title("Computed")

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()
        if filename is None:
            filename = f'logs/fig2d_{date_time}'

    plt.savefig(f'{filename}.png', bbox_inches='tight', pad_inches=0.03, dpi=300) if save else None
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300) if save else None
    plt.show() if show else None
    plt.close('all')
