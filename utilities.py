import numpy as np
import os
import torch
from torch.fft import fftfreq
from itertools import chain

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        # To set maximum domain_size based on n_dims
        # max_domain_size = np.array([10000, 1000, 200])
        # if min(n_ext) < max_domain_size[n_dims-1]:
        #     n_domains = np.array([1., 1., 1.])

        # Modify n_domains and domain_size to optimum values. Enables different number of domains in each dimension
        # round n_ext to nearest 10 and get new n_domains based on min(n_ext)
        n_domains = np.ceil(n_ext / (min(np.round(n_ext[:n_dims], -1)) / n_domains))  # Update n_domains
        domain_size = n_ext / n_domains  # Update domain_size

        # Increase boundary_post in dimension(s) to satisfy n_domains and domain_size conditions
        # Difference between original n_ext and new n_ext based on modified domain_size
        boundary_add = n_domains[:n_dims]*domain_size[np.argmin(n_ext[:n_dims])] - n_ext[:n_dims]
        boundary_add = boundary_add - np.min(boundary_add)  # Shift all elements of boundary_add to non-negative values
        boundary_post[:n_dims] += boundary_add  # Increase boundary_post
        n_ext = n_roi + boundary_pre + boundary_post  # Update n_ext
        domain_size = n_ext / n_domains  # Update domain_size

        # Increase boundary_post in dimension(s) until the subdomain size is int
        while (domain_size % 1 != 0).any() or (boundary_post % 1 != 0).any():
            boundary_post = np.round(boundary_post + n_domains * (np.ceil(domain_size) - domain_size), 2)
            n_ext = n_roi + boundary_pre + boundary_post
            domain_size = np.round(n_ext / n_domains, 2)

        assert np.all(domain_size[:n_dims] == domain_size[0])  # Assert all subdomains are of the same size
        # assert (domain_size[:n_dims] <= max_domain_size[n_dims-1]).all()  # Assert size(subdomains) <= max_domain_size
        assert (domain_size % 1 == 0).all()  # Assert domain_size is an integer
        assert (boundary_post % 1 == 0).all()  # Assert boundary_post is an integer
        assert (boundary_post >= np.ceil(boundary_widths)).all()  # Assert boundary_post is not smaller than initial

    # Cast below 4 parameters to int because they are used in padding, indexing/slicing, creation of arrays
    boundary_pre = boundary_pre.astype(int)
    boundary_post = boundary_post.astype(int)
    n_domains = n_domains.astype(int)
    domain_size = domain_size.astype(int)

    k0 = (1. * 2. * np.pi) / wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
    v_raw = (k0 ** 2) * (n ** 2)

    # compute tiny non-zero minimum value to prevent division by zero in homogeneous media
    pixel_size = wavelength / ppw  # Grid pixel size in um (micron)
    mu_min = ((10.0 / (boundary_widths[:n_dims] * pixel_size)) if (
            boundary_widths != 0).any() else check_input_len(0, 0, n_dims)).astype(np.float32)
    mu_min = max(np.max(mu_min), np.max(1.e-3 / (n_ext[:n_dims] * pixel_size)))
    v_min = 0.5 * np.imag((k0 + 1j * np.max(mu_min)) ** 2)

    source = check_input_dims(source)  # Ensure source term is a 3-d array
    if source.shape != n.shape:
        source = pad_boundaries(source, (0, 0, 0), np.array(n.shape) - np.array(source.shape),
                                mode="constant")
    source = torch.tensor(source, dtype=torch.complex64, device=device)
    source = pad_boundaries_torch(source, boundary_pre, boundary_post, mode="constant")  # pad source term (scale later)

    # compute the fft over omega times the domain size
    omega = check_input_len(omega, 1, n_dims)  # Ensure omega is a 3-element array with 1s after n_dims

    return (n_roi, source, n_dims, boundary_widths, boundary_pre, boundary_post,
            n_domains, domain_size, omega, v_min, v_raw, device)
    # return locals()


def check_input_dims(x):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source) """
    for _ in range(3 - x.ndim):
        x = np.expand_dims(x, axis=-1)
    return x


def check_input_len(x, e, n_dims):
    """ Convert 'x' to a 3-element numpy array, appropriately, i.e., either repeat, or add 'e'. """
    if isinstance(x, int) or isinstance(x, float):
        x = n_dims * tuple((x,)) + (3 - n_dims) * (e,)
    elif len(x) == 1:
        x = n_dims * tuple(x) + (3 - n_dims) * (e,)
    elif isinstance(x, list) or isinstance(x, tuple):
        x += (3 - len(x)) * (e,)
    if isinstance(x, np.ndarray):
        x = np.concatenate((x, np.zeros(3 - len(x))))
    return np.array(x)


def get_dims(n):
    """ Return dimensions after applying the custom squeeze function """
    n = squeeze_(n)
    return n.ndim


def squeeze_(n):
    """ Custom squeeze function that only squeezes the last dimension(s) if they are of size 1 """
    while n.shape[-1] == 1:
        n = np.squeeze(n, axis=-1)
    return n


# Laplacian
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


# Padding and anti-reflection boundary layer (ARL)
def pad_func(m, boundary_pre, boundary_post, n_roi, n_dims):
    """ Pad and then Apply Anti-reflection boundary layer (ARL) filter on the boundaries """
    m = pad_boundaries(m, boundary_pre, boundary_post, mode="edge")  # pad m using edge values
    m = torch.tensor(m, dtype=torch.complex64, device=device)
    for i in range(n_dims):
        left_boundary = boundary_(boundary_pre[i])
        right_boundary = boundary_(boundary_post[i]).flipud()
        full_filter = torch.cat((left_boundary, torch.ones(n_roi[i], device=device), right_boundary))
        m = torch.moveaxis(m, i, -1) * full_filter  # transpose m to multiply last axis with full_filter
        m = torch.moveaxis(m, -1, i)  # transpose back
    return m


def pad_boundaries(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) in all dimensions """
    pad_width = tuple(zip(boundary_pre, boundary_post))  # pairs ((a0, b0), (a1, b1), (a2, b2))
    return np.pad(x, pad_width, mode)


def pad_boundaries_torch(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) in all dimensions """
    t = zip(boundary_pre[::-1], boundary_post[::-1])  # reversed pairs (a2, b2) (a1, b1) (a0, b0)
    pad_width = tuple(chain.from_iterable(t))  # flatten to (a2, b2, a1, b1, a0, b0)
    return torch.nn.functional.pad(x, pad_width, mode)


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function
    :param x: Size of the ARL
    :return boundary_: Boundary"""
    return torch.tensor(np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007]),
                        dtype=torch.float32, device=device)


# Used in tests
def full_matrix(operator, d):
    """ Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d) """
    nf = np.prod(d)
    m = torch.zeros(*(nf, nf), dtype=torch.complex64, device=device)
    b = torch.zeros(tuple(d), dtype=torch.complex64, device=device)
    b.view(-1)[0] = 1
    for i in range(nf):
        m[:, i] = torch.ravel(operator(torch.roll(b, i)))
    return m.cpu().numpy()


# Metrics
def max_abs_error(e, e_true):
    """ (Normalized) Maximum Absolute Error (MAE) ||e-e_true||_{inf} / ||e_true|| """
    return np.max(np.abs(e - e_true)) / np.linalg.norm(e_true)


def max_relative_error(e, e_true):
    """Computes the maximum error, normalized by the rms of the true field."""
    return np.max(np.abs(e - e_true)) / np.sqrt(np.mean(np.abs(e_true) ** 2))


def relative_error(e, e_true):
    """ Relative error ⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩ """
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)
