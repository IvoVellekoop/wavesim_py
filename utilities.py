import numpy as np
from scipy.sparse import dok_matrix


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function """
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


def check_input_dims(a):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source) """
    for _ in range(3 - a.ndim):
        a = np.expand_dims(a, axis=-1)
    return a


def check_input_len(x, e, n_dims):
    """ Convert 'x' to a 3-element numpy array, appropriately, i.e., either repeat, or add 0 or 1. """
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
    """ Create a discrete Fourier transform matrix. Faster than scipy dft function """
    r = np.arange(n)
    omega = np.exp((-2 * np.pi * 1j) / n)  # remove the '-' for inverse fourier
    return np.vander(omega ** r, increasing=True).astype(np.complex64)  # faster than meshgrid


def full_matrix(operator, d):
    """ Converts operator to an 2D square matrix of size d.
    (operator should be a function taking a single column vector as input?) """
    nf = np.prod(d)
    m = dok_matrix((nf, nf), dtype=np.complex64)
    b = np.zeros(d, dtype=np.complex64)
    b[*(0,)*b.ndim] = 1
    for i in range(nf):
        m[:, i] = operator(np.roll(b, i)).ravel()
    return m


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


def overlap_decay(x):
    """ Linear decay from 0 to 1 of size x """
    return np.interp(np.arange(x), [0, x - 1], [0, 1])


def pad_func(m, boundary_pre, boundary_post, n_roi, n_dims):
    """ Apply Anti-reflection boundary layer (ARL) filter on the boundaries """
    for i in range(n_dims):
        left_boundary = boundary_(boundary_pre[i])
        right_boundary = np.flip(boundary_(boundary_post[i]))
        full_filter = np.concatenate((left_boundary, np.ones(n_roi[i]), right_boundary))
        m = np.moveaxis(m, i, -1) * full_filter
        m = np.moveaxis(m, -1, i)
    return m.astype(np.complex64)


def relative_error(e, e_true):
    """ Relative error ⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩ """
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)
