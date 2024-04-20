from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from itertools import chain


def partition(array: Tensor, n_domains: tuple[int, int, int]) -> np.ndarray:
    """ Split a 3-D array into a 3-D set of subarrays of approximately equal sizes."""
    n_domains = np.array(n_domains)
    size = np.array(array.shape)
    if any(size < n_domains) or any(n_domains <= 0) or len(n_domains) != 3:
        raise ValueError(
            f"Number of domains {n_domains} must be larger than 1 and less than or equal to the size of the array {array.shape}")

    # Calculate the size of each domain
    large_domain_size = np.ceil(size / n_domains).astype(int)
    small_domain_count = large_domain_size * n_domains - size
    large_domain_count = n_domains - small_domain_count
    subdomain_sizes = [
        (large_domain_size[dim],) * large_domain_count[dim] + (large_domain_size[dim] - 1,) * small_domain_count[
            dim] for dim in range(3)]

    split = _sparse_split if array.is_sparse else torch.split

    array = split(array, subdomain_sizes[0], dim=0)
    array = [split(part, subdomain_sizes[1], dim=1) for part in array]
    array = [[split(part, subdomain_sizes[2], dim=2) for part in subpart] for subpart in array]
    return list_to_array(array, dim=3)


def list_to_array(input: list, dim: int) -> np.ndarray:
    """ Convert a nested list of depth `dim` to a numpy object array """
    # first determine the size of the final array
    size = np.zeros(dim, dtype=int)
    outer = input
    for i in range(dim):
        size[i] = len(outer)
        outer = outer[0]

    # allocate memory
    array = np.empty(size, dtype=object)

    # flatten the input array
    for i in range(dim - 1):
        input = sum(input, input[0][0:0])  # works both for tuples and lists

    # copy to the output array
    ra = array.reshape(-1)
    assert ra.base is not None  # must be a view
    for i in range(ra.size):
        ra[i] = input[i]
    return array


def _sparse_split(tensor: Tensor, sizes: Sequence[int], dim: int) -> np.ndarray:
    """ Split a COO-sparse tensor into a 3-D set of subarrays of approximately equal sizes."""
    coordinate_to_domain = np.array(sum([(idx,) * size for idx, size in enumerate(sizes)], ()))
    domain_starts = np.cumsum((0,) + sizes)
    tensor = tensor.coalesce()
    indices = tensor.indices().cpu().numpy()
    domains = coordinate_to_domain[indices[dim, :]]

    def extract_subarray(domain: int) -> Tensor:
        mask = domains == domain
        domain_indices = indices[:, mask]
        if len(domain_indices) == 0:
            return None
        domain_values = tensor.values()[mask]
        domain_indices[dim, :] -= domain_starts[domain]
        size = list(tensor.shape)
        size[dim] = sizes[domain]
        return torch.sparse_coo_tensor(domain_indices, domain_values, size)

    return [extract_subarray(d) for d in range(len(sizes))]


def combine(domains: np.ndarray, device=None) -> Tensor:
    """ Concatenates a 3-d array of 3-d tensors"""

    # Calculate total size for each dimension
    total_size = [
        sum(tensor.shape[0] for tensor in domains[:, 0, 0]),
        sum(tensor.shape[1] for tensor in domains[0, :, 0]),
        sum(tensor.shape[2] for tensor in domains[0, 0, :])
    ]

    # allocate memory
    template = domains[0, 0, 0]
    result_tensor = torch.empty(size=total_size, dtype=template.dtype, device=device or template.device)

    # Fill the pre-allocated tensor
    index0 = 0
    for i, tensor_slice0 in enumerate(domains[:, 0, 0]):
        index1 = 0
        for j, tensor_slice1 in enumerate(domains[0, :, 0]):
            index2 = 0
            for k, tensor in enumerate(domains[0, 0, :]):
                tensor = domains[i, j, k]
                if tensor.is_sparse:
                    tensor = tensor.to_dense()
                end0 = index0 + tensor.shape[0]
                end1 = index1 + tensor.shape[1]
                end2 = index2 + tensor.shape[2]
                result_tensor[index0:end0, index1:end1, index2:end2] = tensor
                index2 += tensor.shape[2]
            index1 += domains[i, j, 0].shape[1]
        index0 += tensor_slice0.shape[0]

    return result_tensor


def preprocess(n, source, wavelength, ppw, boundary_widths, n_domains, omega):
    """ Preprocess the input parameters for the simulation
    :param n: Refractive index distribution 
    :param source: Direct source term instead of amplitude and location
    :param wavelength: Wavelength in um (micron)
    :param ppw: Points per wavelength
    :param boundary_widths: Width of absorbing boundaries
    :param n_domains: Number of subdomains to decompose into, in each dimension
    :param omega: Compute the fft over omega times the domain size
    :return: Preprocessed parameters for the HelmholtzBase class """
    n = check_input_dims(n.astype(np.complex64))  # Ensure n is a 3-d array
    n_dims = get_dims(n)  # Number of dimensions in simulation
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
        # Modify such that decompose into domains only when total_size*bytes (or bits?) > t * (memory of one GPU/device)
        # If decomposing, size of one domain * bytes (or bits) < t * (memory of one GPU)
        # Here t a factor, say 0.8? Try different values
        # To set maximum domain_size based on n_dims 
        max_domain_size = np.array([100000, 100000, 500])
        if min(n_ext) <= max_domain_size[n_dims - 1]:
            n_domains = np.ones(3)

        # Modify n_domains and domain_size to optimum values. Enables different number of domains in each dimension
        # round n_ext to nearest 10 and get new n_domains based on min(n_ext)
        n_domains = np.ceil(n_ext / (min(np.round(n_ext[:n_dims], -1)) / n_domains))  # Update n_domains
        domain_size = n_ext / n_domains  # Update domain_size

        # Increase boundary_post in dimension(s) to satisfy n_domains and domain_size conditions
        # Difference between original n_ext and new n_ext based on modified domain_size
        boundary_add = n_domains[:n_dims] * domain_size[np.argmin(n_ext[:n_dims])] - n_ext[:n_dims]
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

    # Pad source to the size of n_ext = n_roi + boundary_pre + boundary_post
    source = check_input_dims(source)  # Ensure source term is a 3-d array
    if source.shape != n.shape:  # If source term is not given, pad to the size of
        source = pad_boundaries(source, (0, 0, 0), np.array(n.shape) - np.array(source.shape),
                                mode='constant')
    source = torch.tensor(source, dtype=torch.complex64)
    source = pad_boundaries_torch(source, boundary_pre, boundary_post, mode='constant')  # pad source term (scale later)

    k0 = (1. * 2. * np.pi) / wavelength  # wave-vector k = 2*pi/lambda, where lambda = 1.0 um (micron)
    n_sq = add_absorption(n ** 2, boundary_pre, boundary_post, n_roi, n_dims)  # add absorption to n^2
    v_raw = (k0 ** 2) * n_sq  # raw potential v = k^2 * n^2

    # compute tiny non-zero minimum value to prevent division by zero in homogeneous media
    pixel_size = wavelength / ppw  # Grid pixel size in um (micron)
    mu_min = ((10.0 / (boundary_widths[:n_dims] * pixel_size)) if (
            boundary_widths != 0).any() else check_input_len(0, 0, n_dims)).astype(np.float32)
    mu_min = max(np.max(mu_min), np.max(1.e-3 / (n_ext[:n_dims] * pixel_size)))
    v_min = 0.5 * np.imag((k0 + 1j * np.max(mu_min)) ** 2)

    # compute the fft over omega times the domain size
    omega = check_input_len(omega, 1, n_dims)  # Ensure omega is a 3-element array with 1s after n_dims

    return (n_roi, source, n_dims, boundary_widths, boundary_pre, boundary_post,
            n_domains, domain_size, omega, v_min, v_raw)
    # return locals()


def check_input_dims(x):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source)
    :param x: Input array
    :return: Array with 3 dimensions """
    for _ in range(3 - x.ndim):
        x = np.expand_dims(x, axis=-1)  # Expand dimensions to 3
    return x


def check_input_len(x, e, n_dims):
    """ Check the length of input arrays and expand them to 3 elements if necessary. Either repeat or add 'e'
    :param x: Input array
    :param e: Element to add
    :param n_dims: Number of dimensions
    :return: Array with 3 elements """
    if isinstance(x, int) or isinstance(x, float):  # If x is a single number
        x = n_dims * tuple((x,)) + (3 - n_dims) * (e,)  # Repeat the number n_dims times, and add (3-n_dims) e's
    elif len(x) == 1:  # If x is a single element list or tuple
        x = n_dims * tuple(x) + (3 - n_dims) * (e,)  # Repeat the element n_dims times, and add (3-n_dims) e's
    elif isinstance(x, list) or isinstance(x, tuple):  # If x is a list or tuple
        x += (3 - len(x)) * (e,)  # Add (3-len(x)) e's
    if isinstance(x, np.ndarray):  # If x is a numpy array
        x = np.concatenate((x, np.zeros(3 - len(x))))  # Concatenate with (3-len(x)) zeros
    return np.array(x)


def get_dims(n):
    """ Get the number of dimensions of 'n' 
    :param n: Input array
    :return: Number of dimensions """
    n = squeeze_(n)  # Squeeze the last dimension if it is 1
    return n.ndim  # Number of dimensions


def squeeze_(n):
    """ Squeeze the last dimension of 'n' if it is 1 
    :param n: Input array
    :return: Squeezed array """
    while n.shape[-1] == 1:
        n = np.squeeze(n, axis=-1)
    return n


# Add absorption to the refractive index squared
def add_absorption(m, boundary_pre, boundary_post, n_roi, n_dims):
    """ Add (weighted) absorption to the refractive index squared 
    :param m: array (Refractive index squared)
    :param boundary_pre: Boundary before
    :param boundary_post: Boundary after
    :param n_roi: Number of points in the region of interest
    :param n_dims: Number of dimensions
    :return: m with absorption """
    w = np.ones_like(m)  # Weighting function (1 everywhere)
    w = pad_boundaries(w, boundary_pre, boundary_post, mode='linear_ramp')  # pad w using linear_ramp
    a = 1 - w  # for absorption, inverse weighting 1 - w
    for i in range(n_dims):
        left_boundary = boundary_(boundary_pre[i])  # boundary_ is a linear window function
        right_boundary = np.flip(boundary_(boundary_post[i]))  # flip is a vertical flip
        full_filter = np.concatenate((left_boundary, np.ones(n_roi[i], dtype=np.float32), right_boundary))
        a = np.moveaxis(a, i, -1) * full_filter  # transpose to last dimension, apply filter
        a = np.moveaxis(a, -1, i)  # transpose back to original position
    a = 1j * a  # absorption is imaginary

    m = pad_boundaries(m, boundary_pre, boundary_post, mode='edge')  # pad m using edge values
    m = w * m + a  # add absorption to m
    return m


def pad_boundaries(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) in all dimensions using numpy pad
    :param x: Input array
    :param boundary_pre: Boundary before
    :param boundary_post: Boundary after
    :param mode: Padding mode
    :return: Padded array """
    pad_width = tuple(zip(boundary_pre, boundary_post))  # pairs ((a0, b0), (a1, b1), (a2, b2))
    return np.pad(x, pad_width, mode)


def pad_boundaries_torch(x, boundary_pre, boundary_post, mode):
    """ Pad 'x' with boundary_pre (before) and boundary_post (after) in all dimensions using PyTorch functional.pad
    :param x: Input tensor
    :param boundary_pre: Boundary before
    :param boundary_post: Boundary after
    :param mode: Padding mode
    :return: Padded tensor """
    t = zip(boundary_pre[::-1], boundary_post[::-1])  # reversed pairs (a2, b2) (a1, b1) (a0, b0)
    pad_width = tuple(chain.from_iterable(t))  # flatten to (a2, b2, a1, b1, a0, b0)
    return torch.nn.functional.pad(x, pad_width, mode)


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function
    :param x: Size of the ARL
    :return boundary_: Boundary"""
    return ((np.arange(1, x + 1) - 0.21).T / (x + 0.66)).astype(np.float32)


# Used in tests
def full_matrix(operator, d):
    """ Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d) 
    :param operator: Operator to convert to a matrix
    :param d: Dimensions of the operator
    :return: Matrix representation of the operator """
    nf = np.prod(d)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.zeros(*(nf, nf), dtype=torch.complex64, device=device)
    b = torch.zeros(tuple(d), dtype=torch.complex64, device=device)
    b.view(-1)[0] = 1
    for i in range(nf):
        m[:, i] = torch.ravel(operator(torch.roll(b, i)).to(device))
    return m.cpu().numpy()


# Metrics
def max_abs_error(e, e_true):
    """ (Normalized) Maximum Absolute Error (MAE) ||e-e_true||_{inf} / ||e_true|| 
    :param e: Computed field
    :param e_true: True field
    :return: (Normalized) MAE """
    return np.max(np.abs(e - e_true)) / np.linalg.norm(e_true)


def max_relative_error(e, e_true):
    """Computes the maximum error, normalized by the rms of the true field 
    :param e: Computed field
    :param e_true: True field
    :return: (Normalized) Maximum Relative Error """
    return np.max(np.abs(e - e_true)) / np.sqrt(np.mean(np.abs(e_true) ** 2))


def relative_error(e, e_true):
    """ Relative error ⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩ 
    :param e: Computed field
    :param e_true: True field
    :return: Relative Error """
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)
