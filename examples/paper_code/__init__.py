import os
import sys
import torch
import numpy as np
from time import time
from torch.fft import fftn, ifftn, fftshift
from porespy.generators import random_spheres
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
from matplotlib import rc, rcParams, colors

sys.path.append(".")
sys.path.append("..")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the wavesim iteration
from wavesim.utilities import preprocess, normalize, relative_error


font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 13}
rc('font', **font)
rcParams['mathtext.fontset'] = 'cm'


def random_spheres_refractive_index(n_size, n_medium=1.33, k_medium=0.01, r=24, clearance=0):
    n = random_spheres(im_or_shape=n_size, r=r, clearance=clearance, seed=12345)
    n = np.array(n, dtype=np.complex64)
    n[n == 1] = n_medium + k_medium*1j
    n[n == 0] = 1.0
    return n
    

def random_refractive_index(n_size):
    torch.manual_seed(0)  # Set the random seed for reproducibility
    n = (1.0 + torch.rand(n_size, dtype=torch.float32) +
         0.03j * torch.rand(n_size, dtype=torch.float32))  # Random refractive index
    n = smooth_n(n, n_size)  # Low pass filter to remove sharp edges
    n.real = normalize(n.real, a=1.0, b=2.0)  # Normalize to [1, 2]
    n.imag = normalize(n.imag, a=0.0, b=0.03)  # Normalize to [0, 0.05]
    # make sure that the imaginary part of n² is positive
    mask = (n ** 2).imag < 0
    n.imag[mask] *= -1.0

    n[0:5, :, :] = 1
    n[-5:, :, :] = 1
    return n.numpy()


def smooth_n(n, n_size):
    """Low pass filter to remove sharp edges"""
    n_fft = fftn(n)
    w = (window(n_size[1]).T @ window(n_size[0])).T[:, :, None] * (window(n_size[2])).view(1, 1, n_size[2])
    n = ifftn(torch.multiply(n_fft, fftshift(w)))
    n = torch.clamp(n.real, min=1.0) + 1.0j * torch.clamp(n.imag, min=0.0)

    assert (n ** 2).imag.min() >= 0, 'Imaginary part of n² is negative'
    assert n.shape == n_size, 'n and n_size do not match'
    assert n.dtype == torch.complex64, 'n is not complex64'
    return n


def window(x):
    """Create a window function for low pass filtering"""
    c0 = round(x / 4)
    cl = (x - c0) // 2
    cr = cl
    if c0 + cl + cr != x:
        c0 = x - cl - cr
    return torch.cat((torch.zeros(1, cl),
                      torch.ones(1, c0),
                      torch.zeros(1, cr)), dim=1)


def construct_source(n_size, boundary_array):
    """ Set up source, with size same as n + 2*boundary_widths, 
        and a plane wave source on one edge of the domain """
    # TODO: use CSR format instead?
    # TODO: source size should not include boundaries (framework should shift source to correct position)
    n_ext = tuple(np.array(n_size) + 2 * boundary_array)
    indices = [[boundary_array[0]]]  # the start of the source (right at the boundary). Transpose list
    values = torch.ones(1, n_ext[1], n_ext[2])  # the source itself
    return torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)


def sim_3d_random(
    filename, 
    sim_size, 
    n_domains, 
    n_boundary=8, 
    n_medium=1.33, 
    k_medium=0.01, 
    r=24, 
    clearance=0, 
    full_residuals=False, 
    device=None
):
    """Run a simulation with the given parameters and save the results to a file"""

    wavelength = 1.  # Wavelength in micrometers
    pixel_size = wavelength / 4  # Pixel size in wavelength units
    boundary_wavelengths = 5  # Boundary width in wavelengths
    # boundary_widths = [round(boundary_wavelengths * wavelength / pixel_size)] * 3  #, 0, 0]  # Boundary width in pixels
    boundary_widths = [round(boundary_wavelengths * wavelength / pixel_size), 0, 0]  # Boundary width in pixels
    # Periodic boundaries True (no wrapping correction) if boundary width is 0, else False (wrapping correction)
    periodic = tuple(np.where(np.array(boundary_widths) == 0, True, False))
    n_dims = np.count_nonzero(sim_size != 1)  # Number of dimensions

    # Size of the simulation domain
    n_size = np.ones_like(sim_size, dtype=int)
    n_size[:n_dims] = sim_size[:n_dims] * wavelength / pixel_size  # Size of the simulation domain in pixels
    n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

    for i in range(n_dims):
        filename += f'{sim_size[i]}_'
    # filename += f'bw{boundary_wavelengths}_domains'
    filename += f'bw'
    for i in range(n_dims):
        filename += f'{boundary_wavelengths * (not periodic[i])}_'
    filename += f'domains'
    if n_domains is None:
        filename += '111'
    else:
        for i in range(n_dims):
            filename += f'{n_domains[i]}'
        filename += f'_nb{int(n_boundary * pixel_size)}'
    filename += f'_r{int(r * pixel_size)}'
    filename += f'_clearance{int(clearance * pixel_size)}'

    if os.path.exists(filename + '.npz'):
        print(f"File {filename}.npz already exists. Loading data from file.")
        data = np.load(filename + '.npz')
        n = data['n']
        u = data['u']
        sim_time = data['sim_time']
        iterations = data['iterations']
        residual_norm = data['residual_norm']
    else:
        print(f"File {filename}.npz does not exist. Running simulation.")

        n = random_spheres_refractive_index(n_size, n_medium=n_medium, k_medium=k_medium, r=r, clearance=clearance)  # Random spheres refractive index

        # return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
        n, boundary_array = preprocess((n ** 2),
                                       boundary_widths)  # permittivity is n², but uses the same variable n

        print(f"Size of n: {n_size}")
        print(f"Size of n in GB: {n.nbytes / (1024 ** 3):.2f}")
        assert n.imag.min() >= 0, 'Imaginary part of n² is negative'
        assert (n.shape == np.asarray(n_size) + 2 * boundary_array).all(), 'n and n_size do not match'
        assert n.dtype == np.complex64, f'n is not complex64, but {n.dtype}'

        source = construct_source(n_size, boundary_array)

        if n_domains is None:  # single domain
            domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength,
                                     pixel_size=pixel_size, n_boundary=n_boundary, device=device)
            n_domains = (1, 1, 1)
        else:
            domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength,
                                 pixel_size=pixel_size, n_domains=n_domains, n_boundary=n_boundary)

        start = time()
        # Field u and state object with information about the run
        u, iterations, residual_norm = run_algorithm(domain, source, max_iterations=10000,
                                                     full_residuals=full_residuals)
        sim_time = time() - start

        # crop the field to the region of interest
        u = u[*(slice(boundary_widths[i], 
                      u.shape[i] - boundary_widths[i]) for i in range(3))].cpu().numpy()
        n = np.sqrt(n[*(slice(boundary_widths[i], 
                      n.shape[i] - boundary_widths[i]) for i in range(3))].real)

        np.savez_compressed(f'{filename}.npz',
                            n=n,
                            u=u,
                            sim_time=sim_time,
                            iterations=iterations,
                            residual_norm=residual_norm)

    print(f'\nTime {sim_time:2.2f} s; Iterations {iterations}; '
          f'Residual norm {residual_norm[-1] if full_residuals else residual_norm:.3e}')
    return dict(n=n, u=u, sim_time=sim_time, iterations=iterations, residual_norm=residual_norm,
                n_size=n_size, boundary_widths=boundary_widths, n_domains=n_domains,
                pixel_size=pixel_size)


def plot_validation(figname, sim_ref, sim, plt_norm='log', inset=False):
    assert sim_ref['n_size'] == sim['n_size'], 'n_size do not match'
    assert sim_ref['boundary_widths'] == sim['boundary_widths'], 'boundary_widths do not match'
    assert sim_ref['pixel_size'] == sim['pixel_size'], 'pixel_size do not match'
    assert np.allclose(sim_ref['n'], sim['n']), 'n does not match'

    print(f"Relative error: {relative_error(sim['u'], sim_ref['u']):.2e}")

    size = sim_ref['n'].shape
    n = sim_ref['n'][:, :, size[2] // 2].T
    u_ref = np.abs(sim_ref['u'])[:, :, size[2] // 2].T
    u = np.abs(sim['u'])[:, :, size[2] // 2].T

    residuals1 = sim_ref['residual_norm']
    residuals2 = sim['residual_norm']
    iterations1 = sim_ref['iterations']
    iterations2 = sim['iterations']
    n_size = sim_ref['n_size']
    boundary_widths = sim_ref['boundary_widths']
    n_domains = sim['n_domains']
    pixel_size = sim['pixel_size']
    boundary_widths = np.array([boundary_widths] * 3) if np.isscalar(boundary_widths) else np.array(boundary_widths)

    extent = np.array([0, n_size[0], n_size[1], 0]) * pixel_size

    ncols = 4
    max_val = max(np.max(u_ref), np.max(u))
    min_val = min(np.min(u_ref), np.min(u))

    u_ref = normalize(u_ref, min_val=min_val, max_val=max_val)
    u = normalize(u, min_val=min_val, max_val=max_val)

    vmin = 1.e-3
    vmax = 1.
    cmap = 'inferno'

    if plt_norm == 'linear':
        plt_norm_ = colors.Normalize(vmin=vmin, vmax=vmax)
    elif plt_norm == 'log':
        plt_norm_ = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif plt_norm == 'power':
        plt_norm_ = colors.PowerNorm(gamma=0.1, vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(12, 3))
    gs_n = GridSpec(1, 1, right=0.296)  # , top=0.832, bottom=0.1575)
    gs = GridSpec(1, 2, left=0.335, right=0.68, wspace=0.1, width_ratios=[0.92, 1])
    gs1 = GridSpec(1, 1, left=0.78, right=0.95, top=0.81, bottom=0.18)  # , top=0.832, bottom=0.1575)

    pad = 0.04
    fraction = 0.0453
    xticks = np.arange(0, extent[1] + 1, np.round(extent[1] // 5, -1 if extent[1] // 5 >= 10 else 0))
    yticks = np.arange(0, extent[2] + 1, np.round(extent[2] // 5, -1 if extent[1] // 5 >= 10 else 0))

    ax0 = fig.add_subplot(gs_n[0])
    im0 = ax0.imshow(n, cmap=cmap, extent=extent)
    cbar0 = plt.colorbar(mappable=im0, ax=ax0, fraction=fraction, pad=pad)
    cbar0.ax.set_title(r'$n$')
    ax0.set_xlabel(r'$x~(\lambda)$')
    ax0.set_ylabel(r'$y~(\lambda)$')
    ax0.set_title('Refractive index')
    ax0.set_xticks(xticks)
    ax0.set_yticks(yticks)
    ax0.text(0.5, -0.36, '(a)', transform=ax0.transAxes, horizontalalignment='center')

    col = 0
    ax1 = fig.add_subplot(gs[col])
    ax1.imshow(u_ref, cmap=cmap, extent=extent, norm=plt_norm_)
    ax1.set_xlabel(r'$x~(\lambda)$')
    ax1.set_title('Domain = 1')
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([])
    ax1.text(0.5, -0.36, '(b)', transform=ax1.transAxes, horizontalalignment='center')

    col += 1
    ax2 = fig.add_subplot(gs[col])
    im2 = ax2.imshow(u, cmap=cmap, extent=extent, norm=plt_norm_)
    cbar2 = plt.colorbar(mappable=im2, ax=ax2, fraction=fraction, pad=pad)
    cbar2.ax.set_title(r'$|\mathbf{x}|$')
    domain_size = (np.asarray(n_size) + 2 * boundary_widths) / np.asarray(n_domains)
    domain_size = np.round(domain_size).astype(int)

    if inset:
        x1, x2, y1, y2 = 10, 60, 100, 50  # subregion of the original image
        for ax, data in zip([ax0, ax1, ax2], [n, u_ref, u]):
            axins = ax.inset_axes(
                [0.55, 0.55, 0.47, 0.47],
                xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
            axins.imshow(data, extent=extent, cmap=cmap, norm=plt_norm_ if not ax == ax0 else None)
            axins.set_xticks([])
            axins.set_yticks([])
            c = 'red'
            lw = 1.
            # set axes edge color as c
            for spine in axins.spines.values():
                spine.set_color(c)
                spine.set_linewidth(lw)
            # draw a bbox of the region of the inset axes in the parent axes without connecting lines
            rect, lines = ax.indicate_inset_zoom(axins, edgecolor=c, alpha=1., lw=lw)
            for line in lines:
                line.set_alpha(0.)
            if ax == ax2:
                for i in range(n_domains[0] - 1):
                    axins.axvline(x=((i + 1) * domain_size[0] - boundary_widths[0]) * pixel_size,
                                c='k', ls='dashdot', lw=1.)
                for i in range(n_domains[1] - 1):
                    axins.axhline(y=((i + 1) * domain_size[1] - boundary_widths[1]) * pixel_size,
                                c='k', ls='dashdot', lw=1.)

    if n_domains[0] > 1:
        ax2.axvline(x=(domain_size[0] - boundary_widths[0]) * pixel_size,
                    c='k', ls='dashdot', lw=1., label=f'Subdomain\n boundary')
    if n_domains[1] > 1:
        ax2.axhline(y=(domain_size[1] - boundary_widths[1]) * pixel_size,
                    c='k', ls='dashdot', lw=1., label=f'Subdomain\n boundary')
    for i in range(1, n_domains[0] - 1):
        ax2.axvline(x=((i + 1) * domain_size[0] - boundary_widths[0]) * pixel_size,
                    c='k', ls='dashdot', lw=1.)
    for i in range(1, n_domains[1] - 1):
        ax2.axhline(y=((i + 1) * domain_size[1] - boundary_widths[1]) * pixel_size,
                    c='k', ls='dashdot', lw=1.)
    total_domains = np.prod(n_domains)
    if total_domains > 1:
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles[-1:], labels=labels[-1:], 
                   loc='lower left', bbox_to_anchor=(0.28, -0.02),  # inside the plot
                #    bbox_to_anchor=(0.6, -0.1),  # outside the plot
                   borderpad=0.2, handlelength=1.3, handletextpad=0.4, framealpha=0.7)
    ax2.set_xlabel(r'$x~(\lambda)$')
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([])
    ax2.set_title(f'Domains = {n_domains}')
    txt = '(c)'
    ax2.text(0.5, -0.36, txt, transform=ax2.transAxes, horizontalalignment='center')
    col += 1

    ax3 = fig.add_subplot(gs1[0])
    ax3.loglog(np.arange(1, iterations1 + 1),
               residuals1, lw=1.5, label='Domain = 1')
    ax3.loglog(np.arange(1, iterations2 + 1), residuals2, lw=2., c='k',
               ls='dashed', label=f'Domains = {n_domains}')

    ax3.xaxis.set_major_locator(LogLocator(numticks=15))
    ax3.xaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    ax3.yaxis.set_major_locator(LogLocator(numticks=15))
    ax3.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    ax3.grid(which='major', axis='both', linestyle='-', linewidth=0.5)
    ax3.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)

    ax3.set_yticks([1.e+0, 1.e-2, 1.e-4, 1.e-6])
    y_min = min(6.e-7, 0.8 * np.nanmin(residuals1), 0.8 * np.nanmin(residuals2))
    y_max = max(2.e+0, 1.2 * np.nanmax(residuals1), 1.2 * np.nanmax(residuals2))
    ax3.set_ylim([y_min, y_max])

    ax3.legend(framealpha=0.6, borderpad=0.2, handlelength=1.3, handletextpad=0.4, labelspacing=0.3)
    ax3.set_title(f'Residual')
    ax3.set_ylabel('Residual')
    ax3.set_xlabel('Iterations')
    txt = '(d)'
    ax3.text(0.5, -0.36, txt, transform=ax3.transAxes, horizontalalignment='center')

    plt.savefig(figname, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close('all')
