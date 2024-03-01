import numpy as np
# from numpy.fft import fftfreq
# from scipy.sparse import dok_matrix
import os
import torch
from torch.fft import fftfreq
from itertools import chain

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_dtype(torch.float32)


def preprocess(n=np.ones((1, 1, 1)),  # Refractive index distribution
               source=np.zeros((1, 1, 1)),  # Direct source term instead of amplitude and location
               wavelength=1.,  # Wavelength in um (micron)
               ppw=4,  # points per wavelength
               boundary_widths=(20, 20, 20),  # Width of absorbing boundaries
               n_domains=(1, 1, 1),  # Number of subdomains to decompose into, in each dimension
               omega=10):  # compute the fft over omega times the domain size
    """ Set up and preprocess parameters to pass to HelmholtzBase """

    n = check_input_dims(n)  # Ensure n is a 3-d array
    n_dims = get_dims(n)  # Number of dimensions in problem
    n_roi = np.array(n.shape)  # Num of points in ROI (Region of Interest)

    boundary_widths = check_input_len(boundary_widths, 0, n_dims)  # Ensure it's a 3-element array with 0s after n_dims
    # Separate into _pre (before) and _post (after) boundaries, as _post can be changed to satisfy domain
    # decomposition conditions (max subdomain size, all subdomains same size, domain size is int)
    boundary_pre = np.floor(boundary_widths)
    boundary_post = np.ceil(boundary_widths)

    # determine number of subdomains based on max size, ensure that all are of the same size (pad if not),
    # modify boundary_post and n_ext, and cast parameters to int
    n_ext = n_roi + boundary_pre + boundary_post  # n_roi + boundaries on either side(s)
    n_domains = check_input_len(n_domains, 1, n_dims)  # Ensure n_domains is a 3-element array with 1s after n_dims
    if (n_domains == 1).all():  # If 1 domain, implies no domain decomposition
        domain_size = n_ext.copy()
    else:  # Else, domain decomposition
        domain_size = n_ext/n_domains

        # To add: n_domains unequal if one dimension much large than the other(s). Currently n_domains same in all dims

        # Increase boundary_post in dimension(s) until all subdomains are of the same size
        while (domain_size[:n_dims] != np.max(domain_size[:n_dims])).any():
            boundary_post[:n_dims] += (n_domains[:n_dims] * (np.max(domain_size[:n_dims]) - domain_size[:n_dims]))
            n_ext = n_roi + boundary_pre + boundary_post
            domain_size = n_ext/n_domains

        # Increase number of subdomains until subdomain size is less than max_subdomain_size
        max_subdomain_size = 500  # max permissible size of one sub-domain
        while (domain_size > max_subdomain_size).any():
            n_domains[np.where(domain_size > max_subdomain_size)] += 1
            domain_size = n_ext/n_domains

        # Increase boundary_post in dimension(s) until the subdomain size is int
        while (domain_size % 1 != 0).any() or (boundary_post % 1 != 0).any():
            boundary_post = np.round(boundary_post + n_domains * (np.ceil(domain_size) - domain_size), 2)
            n_ext = n_roi + boundary_pre + boundary_post
            domain_size = np.round(n_ext/n_domains, 2)

    # Cast below 4 parameters to int because they are used in padding, indexing/slicing, creation of arrays
    boundary_pre = boundary_pre.astype(int)
    boundary_post = boundary_post.astype(int)
    n_domains = n_domains.astype(int)
    domain_size = domain_size.astype(int)

    k0 = (1. * 2. * np.pi) / wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
    v_raw = (k0 ** 2) * (n ** 2)
    v_raw = pad_boundaries(v_raw, boundary_pre, boundary_post, mode="edge")  # pad v_raw using edge values

    # compute tiny non-zero minimum value to prevent division by zero in homogeneous media
    pixel_size = wavelength / ppw  # Grid pixel size in um (micron)
    mu_min = ((10.0 / (boundary_widths[:n_dims] * pixel_size)) if (
            boundary_widths != 0).any() else check_input_len(0, 0, n_dims)).astype(np.float32)
    mu_min = max(np.max(mu_min), np.max(1.e+0 / (np.array(v_raw.shape[:n_dims]) * pixel_size)))
    v_min = np.imag((k0 + 1j * np.max(mu_min)) ** 2)

    source = check_input_dims(source)  # Ensure source term is a 3-d array
    if source.shape != n.shape:
        source = pad_boundaries(source, (0, 0, 0), tuple(np.array(n.shape) - np.array(source.shape)), 
                                mode="constant")
    source = torch.tensor(source, dtype=torch.complex64, device=device)
    source = pad_boundaries_torch(source, boundary_pre, boundary_post, mode="constant")  # pad source term (scale later)

    # compute the fft over omega times the domain size
    omega = check_input_len(omega, 1, n_dims)  # Ensure omega is a 3-element array with 1s after n_dims

    return (n_roi, source, n_dims, boundary_widths, boundary_pre, boundary_post,
            n_domains, domain_size, omega, v_min, v_raw, device)
    # return locals()


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function
    :param x: Size of the ARL
    :return boundary_: Boundary"""
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


def check_input_dims(x):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source) """
    for _ in range(3 - x.ndim):
        x = np.expand_dims(x, axis=-1)
    return x


def check_input_len(x, e, n_dims):
    """ Convert 'x' to a 3-element numpy array, appropriately, i.e., either repeat, or add 'e'. """
    if isinstance(x, int) or isinstance(x, float):
        x = n_dims*tuple((x,)) + (3-n_dims) * (e,)
    elif len(x) == 1:
        x = n_dims*tuple(x) + (3-n_dims) * (e,)
    elif isinstance(x, list) or isinstance(x, tuple):
        x += (3 - len(x)) * (e,)
    if isinstance(x, np.ndarray):
        x = np.concatenate((x, np.zeros(3 - len(x))))
    return np.array(x)


def dft_matrix(n):
    """ Create a discrete Fourier transform matrix of size n x n. Faster than scipy dft function """
    r = np.arange(n)
    omega = np.exp((-2 * np.pi * 1j) / n)  # remove the '-' for inverse fourier
    return np.vander(omega ** r, increasing=True).astype(np.complex64)  # faster than meshgrid


def full_matrix(operator, d):
    """ Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d) """
    nf = np.prod(d)
    # m = dok_matrix((nf, nf), dtype=np.complex64)
    m = torch.zeros(*(nf, nf), dtype=torch.complex64, device=device)

    b = np.zeros(d, dtype=np.complex64)
    b.flat[0] = 1
    b = torch.tensor(b, device=device)
    for i in range(nf):
        # m[:, i] = operator(np.roll(b, i)).ravel()
        m[:, i] = torch.ravel(operator(torch.roll(b, i)))
    return m.cpu().numpy()


# def full_matrix(operator, d):
#     """ Converts operator to an 2D square matrix of size d.
#     (operator should be a function taking a single column vector as input?) """
#     shape = list(d)
#     nf = np.prod(d)
#     m = dok_matrix((nf, nf), dtype=np.complex64)
#     # b = csr_matrix(([1], ([0],[0])), shape=(nf, 1), dtype=np.complex64)
#     b = np.zeros((nf, 1), dtype=np.complex64)
#     b[0] = 1
#     for i in range(nf):
#         m[:, i] = np.reshape(operator(np.reshape(b, shape)), (-1,))
#         # b.indices = (b.indices+1)%b.shape[0]
#         b = np.roll(b, (1, 0), axis=(0, 1))
#     return m


def get_dims(n):
    """ Return dimensions after applying the custom squeeze function """
    n = squeeze_(n)
    return n.ndim


def laplacian_sq_f(n_dims, n_fft, pixel_size=1.):
    """ Laplacian squared Fourier space coordinates for given size, spacing, and dimensions
    :param n_dims: number of dimensions
    :param n_fft: window length
    :param pixel_size: sample spacing
    :return Laplacian squared in Fourier coordinates"""
    l_p = coordinates_f(n_fft[0], pixel_size) ** 2
    for d in range(1, n_dims):
        l_p = torch.unsqueeze(l_p, -1) + torch.unsqueeze(coordinates_f(n_fft[d], pixel_size) ** 2, 0)

    for _ in range(3 - n_dims):  # ensure l_p has 3 dimensions
        l_p = torch.unsqueeze(l_p, -1)
    return l_p


def coordinates_f(n_, pixel_size=1.):
    return (2 * torch.pi * fftfreq(n_, pixel_size)).to(device)


def pad_boundaries(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) in all dimensions """
    pad_width = tuple(zip(boundary_pre, boundary_post))  # pairs ((a0, b0), (a1, b1), (a2, b2))
    return np.pad(x, pad_width, mode)


def pad_boundaries_torch(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) in all dimensions """
    t = zip(boundary_pre[::-1], boundary_post[::-1])  # reversed pairs (a2, b2) (a1, b1) (a0, b0)
    pad_width = tuple(chain.from_iterable(t))  # flatten to (a2, b2, a1, b1, a0, b0)
    return torch.nn.functional.pad(x, pad_width, mode)


def pad_func(m, boundary_pre, boundary_post, n_roi, n_dims):
    """ Apply Anti-reflection boundary layer (ARL) filter on the boundaries """
    for i in range(n_dims):
        left_boundary = boundary_(boundary_pre[i])
        right_boundary = np.flip(boundary_(boundary_post[i]))
        full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
        m = np.moveaxis(m, i, -1) * full_filter  # transpose m to multiply last axis with full_filter
        m = np.moveaxis(m, -1, i)  # transpose back
    return m.astype(np.complex64)


def max_abs_error(e, e_true):
    """ (Normalized) Maximum Absolute Error (MAE) ||e-e_true||_{inf} / ||e_true|| """
    return np.max(np.abs(e - e_true)) / np.linalg.norm(e_true)  # np.max(np.abs(e_true))  # 


def relative_error(e, e_true):
    """ Relative error ⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩ """
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)


def squeeze_(n):
    """ Custom squeeze function that only squeezes the last dimension(s) if they are of size 1 """
    while n.shape[-1] == 1:
        n = np.squeeze(n, axis=-1)
    return n
