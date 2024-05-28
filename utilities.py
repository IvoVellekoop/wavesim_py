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


def preprocess(refractive_index, source, boundary_widths=10):
    """ Preprocess the input parameters for the simulation
    :param refractive_index: Refractive index distribution 
    :param source: Direct source term instead of amplitude and location
    :return: Preprocessed refractive index (with boundaries and absorption) and padded source """
    refractive_index = check_input_dims(refractive_index.astype(np.complex64))  # Ensure refractive_index is a 3-d array
    n_dims = get_dims(refractive_index)  # Number of dimensions in simulation
    n_roi = np.array(refractive_index.shape)  # Num of points in ROI (Region of Interest)

    boundary_widths = check_input_len(boundary_widths, 0, n_dims)  # Ensure it's a 3-element array with 0s after n_dims
    # Separate into _pre (before) and _post (after) boundaries, as _post can be changed to satisfy domain
    # decomposition conditions (max subdomain size, all subdomains same size, domain size is int)
    boundary_pre = np.floor(boundary_widths).astype(int)
    boundary_post = np.ceil(boundary_widths).astype(int)

    # Pad source to the size of n_ext = n_roi + boundary_pre + boundary_post
    source = check_input_dims(source)  # Ensure source term is a 3-d array
    if source.shape != refractive_index.shape:  # If source term is not given, pad to the size of
        source = pad_boundaries(source, (0, 0, 0), np.array(refractive_index.shape) - np.array(source.shape),
                                mode='constant')
    source = torch.tensor(source, dtype=torch.complex64)
    source = pad_boundaries_torch(source, boundary_pre, boundary_post, mode='constant')  # pad source term (scale later)

    refractive_index = add_absorption(refractive_index ** 2, boundary_pre, boundary_post, n_roi, n_dims)  # add absorption to refractive_index^2
    # refractive_index = torch.tensor(refractive_index, dtype=torch.complex64)

    return refractive_index, source


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
def full_matrix(operator):
    """ Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d) 
    :param operator: Operator to convert to a matrix. This function must be able to accept a 0 scalar, and return a vector of the size and data type of the domain.
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
    This function returns True if x is a scalar 0 or 0.0. It raises an error if x is a scalar that is not equal to 0 or 0.0,
    and returns False otherwise.
    """
    if isinstance(x, float) or isinstance(x, int):
        if x != 0:
            raise ValueError("Cannot set a field to a scalar to a field, only scalar 0.0 is supported")
        return True
    else:
        return False
