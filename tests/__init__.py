import numpy as np
from typing import Sequence
from scipy.special import exp1
import matplotlib.pyplot as plt

from wavesim.engine.array import array_like, Array, subtract
from wavesim.domain import Domain
from wavesim.engine.index_utils import shape_like
from wavesim.iteration import forward, preconditioned_operator, preconditioner, preconditioned_iteration
from wavesim.engine.numpyarray import NumpyArray
from wavesim.utilities import normalize

import os
if os.path.basename(os.getcwd()) == "tests":
    os.chdir("..")


def random_vector(shape: shape_like) -> Array:
    """Construct a random vector for testing operators"""
    scale = np.sqrt(0.5)
    return NumpyArray(
        (np.random.normal(scale=scale, size=shape) + 1.0j * np.random.normal(scale=scale, size=shape)).astype(
            np.complex64
        ),
    )


def random_permittivity(shape: shape_like) -> Array:
    """Construct a random permittivity (n^2) between 1 and 4 with a small positive imaginary part"""
    return NumpyArray(
        (
            np.random.uniform(low=1.0, high=4.0, size=shape) + 1.0j * np.random.uniform(low=0.0, high=0.4, size=shape)
        ).astype(np.complex64),
    )


def all_close(a: Array | array_like, b: Array | array_like, rtol=0.0, atol=0.0, ulptol=100, plot=True):
    """Check if two tensors are close to each other.

    Condition: |a-b| <= atol + rtol * maximum(|b|,|a|) + ulptol * ulp
    Where ulp is the size of the smallest representable difference between two numbers of magnitude ~max(|b[...]|)
    Args:
        a: First tensor
        b: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        ulptol: Tolerance in units of the smallest representable difference between two numbers
        plot: If True, plot the difference between the two arrays if it exceeds the tolerances
    """
    a = a.gather() if isinstance(a, Array) else np.asarray(a)
    b = b.gather() if isinstance(b, Array) else np.asarray(b)
    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        print(f"Shapes do not match: {a.shape} != {b.shape}")
        return False
    if a.size == 0 and b.size == 0:
        return True

    # compute the size of a single ULP
    ab_max = np.maximum(np.abs(a), np.abs(b))  # max |a|, |b| for each element
    ulp = np.maximum(_ulp(a), _ulp(b))
    tolerance = atol + rtol * ab_max + ulptol * ulp  # tolerance per element
    diff = np.abs(a - b)
    if np.any(diff > tolerance):
        abs_err = diff.max().item()
        rel_err = (diff / ab_max).max()
        print(f"\nabsolute error {abs_err} = {abs_err / ulp} ulp\nrelative error {rel_err}")
        if plot:
            plot_difference(a, b, tolerance)
        return False
    else:
        return True


def _ulp(a: np.ndarray) -> float:
    """Returns the epsilon (machine precision error) for the data type of `a`"""
    try:
        finfo = np.finfo(a.dtype)
        exponent = np.ceil(np.log2(np.abs(a).max()))
        return float(finfo.eps) * 2.0 ** np.clip(exponent, finfo.minexp, finfo.maxexp)
    except ValueError:
        return 0.0


def plot_difference(a: np.ndarray, b: np.ndarray, tolerance: float):
    """Plot the difference between two arrays and the tolerance"""
    section = list(s // 2 for s in a.shape)
    if len(section) > 0:
        section[0] = slice(None)
    a = a[*section]
    b = b[*section]
    tolerance = tolerance[*section]

    plt.subplot(2, 1, 1)
    plt.plot(a.real, label="a.real")
    plt.plot(b.real, label="b.real")
    plt.plot(a.imag, label="a.imag")
    plt.plot(b.imag, label="b.imag")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.abs(a - b), label="|a-b|")
    plt.plot(np.arange(a.size), tolerance, label="tolerance")
    plt.legend()
    plt.show()


def domain_operator(domain: Domain, function: str, **kwargs):
    """Constructs various operators by combining calls to 'medium', 'propagator', etc."""

    def potential_(dom, x, *, out):
        dom.medium(x, out=out)
        subtract(x, out, out=out)

    fn = {
        "medium": lambda x, out: domain.medium(x, out=out),
        "propagator": lambda x, out: domain.propagator(x, out=out),
        "inverse_propagator": domain.inverse_propagator,
        "potential": potential_,
        "forward": lambda x, out: forward(domain, x, out=out),
        "preconditioned_operator": lambda x, out: preconditioned_operator(domain, x, out=out),
        "preconditioner": lambda x, out: preconditioner(domain, x, out=out),
        "richardson": lambda x, out: preconditioned_iteration(domain, x, out=out, tmp=out, **kwargs),
    }[function]

    def operator_(x):
        a = domain.allocate(x)
        out = domain.allocate()
        fn(a, out=out)
        return out

    return operator_


def full_matrix(operator):
    """Converts operator to a 2D square matrix of size np.prod(d) x np.prod(d)

    :param operator: Operator to convert to a matrix. This function must be able to accept a 0 scalar, and
                     return a vector of the size and data type of the domain.
    """
    # apply the operator to 0 so that we can get size and data type
    test = operator(0.0)
    shape = test.shape
    nf = np.prod(shape)

    # construct matrix and source
    M = np.zeros((nf, nf), dtype=test.dtype)
    b_data = np.zeros(shape, dtype=test.dtype)
    b = NumpyArray(b_data, dtype=test.dtype)
    for i in range(nf):
        b_data.ravel()[i] = 1
        M[:, i] = operator(b).gather().ravel()
        b_data.ravel()[i] = 0

    return M


def max_abs_error(e, e_true):
    """(Normalized) Maximum Absolute Error (MAE) ||e-e_true||_{inf} / ||e_true||

    :param e: Computed field
    :param e_true: True field
    :return: (Normalized) MAE
    """
    return np.max(np.abs(e - e_true)) / np.linalg.norm(e_true)


def max_relative_error(e, e_true):
    """Computes the maximum error, normalized by the rms of the true field

    :param e: Computed field
    :param e_true: True field
    :return: (Normalized) Maximum Relative Error
    """
    return np.max(np.abs(e - e_true)) / np.sqrt(np.mean(np.abs(e_true) ** 2))


def relative_error(e, e_true):
    """Relative error ``⟨|e-e_true|^2⟩ / ⟨|e_true|^2⟩``

    :param e: Computed field
    :param e_true: True field
    :return: Relative Error
    """
    return np.nanmean(np.abs(e - e_true) ** 2) / np.nanmean(np.abs(e_true) ** 2)


def relative_error_check(e, e_true, threshold=1.0e-3, plot=True):
    """Check if the relative error exceeds the threshold

    :param e: Computed field
    :param e_true: True field
    :param threshold: Threshold for the relative error
    :param plot: If True, plot the two arrays if the relative error exceeds the threshold
    :return: True if the relative error exceeds the threshold
    """
    re = relative_error(e, e_true)
    if re > threshold:
        print(f"Relative error {re:.2e} higher than threshold {threshold:.2e}")
        if plot:
            plot_fields(e, e_true, normalize_u=False)
        return False
    else:
        return True


def plot_fields(u, u_ref, re=None, normalize_u=True):
    """Plot the computed field u and the reference field u_ref.
    If u and u_ref are 1D arrays, the real and imaginary parts are plotted separately.
    If u and u_ref are 2D arrays, the absolute values are plotted.
    If u and u_ref are 3D arrays, the central slice is plotted.
    If normalize_u is True, the values are normalized to the same range.
    The relative error is (computed, if needed, and) displayed.
    """

    re = relative_error(u, u_ref) if re is None else re

    if u.ndim == 1 and u_ref.ndim == 1:
        plt.subplot(211)
        plt.plot(u_ref.real, label="Reference")
        plt.plot(u.real, label="Computed")
        plt.plot((u_ref.real - u.real)*10, label="Difference x 10")
        plt.legend()
        plt.title(f"Real part (RE = {relative_error(u.real, u_ref.real):.2e})")
        plt.grid()

        plt.subplot(212)
        plt.plot(u_ref.imag, label="Reference")
        plt.plot(u.imag, label="Computed")
        plt.plot((u_ref.imag - u.imag)*10, label="Difference x 10")
        plt.legend()
        plt.title(f"Imaginary part (RE = {relative_error(u.imag, u_ref.imag):.2e})")
        plt.grid()

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()
        plt.show()
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
            a = 0
            b = 1
            u = normalize(u, min_val, max_val, a, b)
            u_ref = normalize(u_ref, min_val, max_val, a, b)
        else:
            a = None
            b = None

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(u_ref, cmap="hot_r", vmin=a, vmax=b)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Reference")

        plt.subplot(122)
        plt.imshow(u, cmap="hot_r", vmin=a, vmax=b)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Computed")

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()
        plt.show()


def analytical_solution(x: Sequence[float], wavelength: float):
    """Compute analytic solution for 1D propagation in a homogeneous medium with refractive index 1

    The source is a sinc function centered at x=0

    Args:
        x: 1D array of positions to compute the solution at. Must be equally spaced and increasing
        wavelength: Wavelength
    """
    x = np.abs(np.asarray(x))
    h = np.abs(x[1] - x[0])  # pixel_size
    k = 2.0 * np.pi / wavelength
    phi = k * x

    u_theory = 1.0j * h / (2 * k) * np.exp(1.0j * phi) - h / (4 * np.pi * k) * (  # propagating plane wave
        np.exp(1.0j * phi) * (exp1(1.0j * (k - np.pi / h) * x) - exp1(1.0j * (k + np.pi / h) * x))
        - np.exp(-1.0j * phi) * (-exp1(-1.0j * (k - np.pi / h) * x) + exp1(-1.0j * (k + np.pi / h) * x))
    )
    small = np.abs(k * x) < 1.0e-10  # special case for values close to 0
    u_theory[small] = 1.0j * h / (2 * k) * (1 + 2j * np.arctanh(h * k / np.pi) / np.pi)  # exact value at 0.
    return u_theory
