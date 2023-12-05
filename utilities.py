import numpy as np
from scipy.sparse import dok_matrix
import matplotlib
matplotlib.use("TkAgg")


def boundary_(x):
    """ Anti-reflection boundary layer (ARL). Linear window function """
    return np.interp(np.arange(x), [0, x - 1], [0.04981993, 0.95018007])


def check_input_dims(a):
    """ Expand arrays to 3 dimensions (e.g. refractive index distribution (n) or source) """
    for _ in range(3 - a.ndim):
        a = np.expand_dims(a, axis=-1)
    return a


def dft_matrix(N):
    """ Create a discrete Fourier transform matrix. Faster than scipy dft function """
    r = np.arange(N)
    omega = np.exp((-2 * np.pi * 1j) / N)  # remove the '-' for inverse fourier
    return np.vander(omega ** r, increasing=True)  # faster than meshgrid


def full_matrix(operator, d):
    """ Converts operator to an 2D square matrix of size d.
    (operator should be a function taking a single column vector as input?) """
    shape = list(d)
    nf = np.prod(d)
    # m = np.zeros((nf, nf), dtype=np.complex64)
    b = np.zeros((nf, 1), dtype=np.complex64)
    m = dok_matrix((nf, nf), dtype=np.complex64)
    # b_ = csr_matrix(([1], ([0],[0])), shape=(nf, 1), dtype=np.complex64)
    b[0] = 1
    # b_[0] = 1
    for i in range(nf):
        # print(f'{i}.', end='\r')
        m[:, i] = np.reshape(operator(np.reshape(b, shape)), (-1,))
        b = np.roll(b, (1, 0), axis=(0, 1))
        # b_.indices = (b_.indices+1)%b.shape[0]
    return m


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
