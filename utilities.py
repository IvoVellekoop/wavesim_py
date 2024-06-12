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
            f"Number of domains {n_domains} must be larger than 1 and "
            f"less than or equal to the size of the array {array.shape}")

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
    return list_to_array(array, depth=3)


def list_to_array(input: list, depth: int) -> np.ndarray:
    """ Convert a nested list of depth `depth` to a numpy object array """
    # first determine the size of the final array
    size = np.zeros(depth, dtype=int)
    outer = input
    for i in range(depth):
        size[i] = len(outer)
        outer = outer[0]

    # allocate memory
    array = np.empty(size, dtype=object)

    # flatten the input array
    for i in range(depth - 1):
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


def preprocess(n, source, boundary_widths=10):
    """ Preprocess the input parameters for the simulation
    :param n: Refractive index distribution 
    :param source: Direct source term instead of amplitude and location
    :param boundary_widths: Boundary widths (in pixels)
    :return: Preprocessed permittivity (with boundaries and absorption) and source (with boundaries) """
    n = check_input_dims(n)  # Ensure n is a 3-d array
    if n.dtype != np.complex64:
        n = n.astype(np.complex64)
    n_dims = get_dims(n)  # Number of dimensions in simulation
    n_roi = np.array(n.shape)  # Num of points in ROI (Region of Interest)

    # Ensure boundary_widths is a 3-element array of ints with 0s after n_dims
    boundary_widths = check_input_len(boundary_widths, 0, n_dims).astype(int)

    # Pad source to the size of n_ext = n_roi + boundary_pre + boundary_post
    source = check_input_dims(source)  # Ensure source term is a 3-d array
    if source.shape != n.shape:  # If source term is not given, pad to the size of n
        source = pad_boundaries(source, (0, 0, 0), np.array(n.shape) - np.array(source.shape), mode='constant')
    source = torch.tensor(source, dtype=torch.complex64)
    source = pad_boundaries(source, boundary_widths, mode='constant')  # pad source term (scale later)

    n = add_absorption(n ** 2, boundary_widths, n_roi, n_dims)  # add absorption to n^2
    # n = torch.tensor(n, dtype=torch.complex64)

    return n, source


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


def add_absorption(m, boundary_widths, n_roi, n_dims):
    """ Add (weighted) absorption to the permittivity (refractive index squared)
    :param m: array (permittivity)
    :param boundary_widths: Boundary widths
    :param n_roi: Number of points in the region of interest
    :param n_dims: Number of dimensions
    :return: m with absorption """
    w = np.ones_like(m)  # Weighting function (1 everywhere)
    w = pad_boundaries(w, boundary_widths, mode='linear_ramp')  # pad w using linear_ramp
    a = 1 - w  # for absorption, inverse weighting 1 - w
    for i in range(n_dims):
        left_boundary = boundary_(boundary_widths[i])  # boundary_ is a linear window function
        right_boundary = np.flip(left_boundary)  # flip is a vertical flip
        full_filter = np.concatenate((left_boundary, np.ones(n_roi[i], dtype=np.float32), right_boundary))
        a = np.moveaxis(a, i, -1) * full_filter  # transpose to last dimension, apply filter
        a = np.moveaxis(a, -1, i)  # transpose back to original position
    a = 1j * a  # absorption is imaginary

    m = pad_boundaries(m, boundary_widths, mode='edge')  # pad m using edge values
    m = w * m + a  # add absorption to m
    return m


def pad_boundaries(x, boundary_widths, boundary_post=None, mode='constant'):
    """ Pad 'x' with boundaries in all dimensions using numpy pad (if x is np.ndarray) or PyTorch nn.functional.pad
    (if x is torch.Tensor).
    If boundary_post is specified separately, pad with boundary_widths (before) and boundary_post (after)
    :param x: Input array
    :param boundary_widths: Boundary widths for padding before and after (or just before if boundary_post not None)
    :param boundary_post: Boundary widths for padding after
    :param mode: Padding mode
    :return: Padded array """
    if boundary_post is None:
        boundary_post = boundary_widths

    if isinstance(x, np.ndarray):
        pad_width = tuple(zip(boundary_widths, boundary_post))  # pairs ((a0, b0), (a1, b1), (a2, b2))
        return np.pad(x, pad_width, mode)
    elif torch.is_tensor(x):
        t = zip(boundary_widths[::-1], boundary_post[::-1])  # reversed pairs (a2, b2) (a1, b1) (a0, b0)
        pad_width = tuple(chain.from_iterable(t))  # flatten to (a2, b2, a1, b1, a0, b0)
        return torch.nn.functional.pad(x, pad_width, mode)
    else:
        raise ValueError("Input must be a numpy array or a torch tensor")


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function
    :param x: Size of the ARL
    :return boundary_: Boundary"""
    return ((np.arange(1, x + 1) - 0.21).T / (x + 0.66)).astype(np.float32)


# Used in tests
def full_matrix(operator):
    """ Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d) 
    :param operator: Operator to convert to a matrix. This function must be able to accept a 0 scalar, and
                     return a vector of the size and data type of the domain.
    """
    y = operator(0.0)
    n_size = y.shape
    nf = np.prod(n_size)
    M = torch.zeros((nf, nf), dtype=y.dtype, device=y.device)
    b = torch.zeros(n_size, dtype=y.dtype, device=y.device)
    for i in range(nf):
        b.view(-1)[i] = 1
        M[:, i] = torch.ravel(operator(b))
        b.view(-1)[i] = 0

    return M


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


def is_zero(x):
    """ Check if x is zero

    Some functions allow specifying 0 or 0.0 instead of a torch tensor, to indicate that the array should be cleared.
    This function returns True if x is a scalar 0 or 0.0. It raises an error if x is a scalar that is not equal to 0 or
    0.0, and returns False otherwise.
    """
    if isinstance(x, float) or isinstance(x, int):
        if x != 0:
            raise ValueError("Cannot set a field to a scalar to a field, only scalar 0.0 is supported")
        return True
    else:
        return False


def normalize(x, min_val=None, max_val=None, a=0, b=1):
    """ Normalize x to the range [a, b]
    :param x: Input array
    :param min_val: Minimum value (of x)
    :param max_val: Maximum value (of x)
    :param a: Lower bound for normalization
    :param b: Upper bound for normalization
    :return: Normalized x
    """
    if min_val is None:
        min_val = np.min(x)
    if max_val is None:
        max_val = np.max(x)
    normalized_x = (x - min_val) / (max_val - min_val) * (b - a) + a
    return normalized_x
