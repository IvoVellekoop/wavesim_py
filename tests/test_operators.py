import numpy as np
import pytest

from wavesim.helmholtzdomain import Helmholtz
from tests import domain_operator, full_matrix
from . import random_vector, random_permittivity, all_close
from wavesim.engine import add, subtract, BlockArray, clone, scale

""" Performs checks on the operators represented as matrices (accretivity, norm)."""

parameters = [
    {"n_size": (1, 1, 12), "n_domains": None, "n_boundary": 0, "periodic": (True, True, True)},  # params0
    {"n_size": (1, 1, 12), "n_domains": (1, 1, 1), "n_boundary": 0, "periodic": (True, True, True)},  # params1
    {"n_size": (1, 1, 12), "n_domains": (1, 1, 1), "n_boundary": 5, "periodic": (True, True, False)},  # params2
    {"n_size": (1, 1, 32), "n_domains": (1, 1, 2), "n_boundary": 5, "periodic": (True, True, True)},  # params3
    {"n_size": (1, 1, 32), "n_domains": (1, 1, 2), "n_boundary": 5, "periodic": (True, True, False)},  # params4
    {"n_size": (1, 32, 1), "n_domains": (1, 2, 1), "n_boundary": 5, "periodic": (True, True, True)},  # params5
    {"n_size": (1, 32, 1), "n_domains": (1, 2, 1), "n_boundary": 5, "periodic": (True, False, True)},  # params6
    {"n_size": (32, 1, 1), "n_domains": (2, 1, 1), "n_boundary": 5, "periodic": (False, True, True)},  # params7
    {"n_size": (32, 1, 1), "n_domains": (2, 1, 1), "n_boundary": 5, "periodic": (True, True, True)},  # params8
    # test different-sized domains
    {"n_size": (23, 1, 1), "n_domains": (3, 1, 1), "n_boundary": 2, "periodic": (True, True, True)},  # params9
    {"n_size": (23, 1, 1), "n_domains": (3, 1, 1), "n_boundary": 3, "periodic": (False, True, True)},  # params10
    {"n_size": (1, 5, 19), "n_domains": (1, 1, 2), "n_boundary": 3, "periodic": (True, True, True)},  # params11
    {"n_size": (1, 5, 19), "n_domains": (1, 1, 3), "n_boundary": 3, "periodic": (True, True, True)},  # params12
    {"n_size": (1, 14, 19), "n_domains": (1, 2, 2), "n_boundary": 3, "periodic": (True, False, True)},  # params13
    {"n_size": (8, 12, 8), "n_domains": (2, 3, 2), "n_boundary": 2, "periodic": (True, True, True)},  # params14
    # these parameters are very slow for test_accretivity and should be run only when needed
    # {"n_size": (12, 12, 12), "n_domains": (2, 2, 2), "n_boundary": 3, "periodic": (False, False, True)},  # params16
    # {"n_size": (17, 30, 1), "n_domains": (2, 3, 1), "n_boundary": 3, "periodic": (True, True, True)},  # params14
    # {"n_size": (8, 8, 8), "n_domains": (2, 2, 2), "n_boundary": 2, "periodic": (False, False, True)},  # params14
    # {'n_size': (18, 24, 18), 'n_domains': (2, 3, 2), 'n_boundary': 3, 'periodic': (True, True, True)}, # params17
    # {'n_size': (17, 23, 19), 'n_domains': (2, 3, 2), 'n_boundary': 3, 'periodic': (True, True, True)}, # params18
]


def construct_domain(n_size, n_domains, n_boundary, periodic=(False, False, True)):
    """Construct a domain or multi-domain"""
    permittivity = random_permittivity(n_size)
    if n_domains is not None:
        permittivity = BlockArray(permittivity, n_domains=n_domains)
    return Helmholtz(
        permittivity=permittivity, pixel_size=0.25, wavelength=1.0, periodic=periodic, boundary_width=n_boundary
    )


@pytest.mark.parametrize("params", parameters)
def test_operators(params):
    """Check that operator definitions are consistent:
    - forward = inverse_propagator - medium: A= L + 1 - B
    - preconditioned_operator = preconditioned(operator)
    - richardson = x + α (Γ⁻¹b - Γ⁻¹A x)
    """
    np.random.seed(1234)
    domain = construct_domain(**params)
    x_orig = random_vector(domain.shape)
    x = clone(x_orig)
    B = domain_operator(domain, "medium")
    L1 = domain_operator(domain, "inverse_propagator")
    A = domain_operator(domain, "forward")
    Ax = A(x)
    L1x = L1(x)
    Bx = B(x)
    # original should remain unchanged, except that the data type is changed from complex128 to complex64
    assert all_close(x, x_orig)
    assert all_close(domain.scale * Ax.gather(), L1x.gather() - Bx.gather())

    Γ = domain_operator(domain, "preconditioner")
    ΓA = domain_operator(domain, "preconditioned_operator")
    ΓAx = ΓA(x)
    assert all_close(ΓAx, Γ(Ax))

    α = 0.1
    b = random_vector(domain.shape)
    Γb = Γ(b)
    M = domain_operator(domain, "richardson", source=b, alpha=α)
    assert all_close(M(0), -α * Γb.gather())

    # compute Richardson iteration x + α(Γb - ΓAx)
    residual = domain.allocate()
    scale(-1.0, Γb, out=residual)  # negative source term in the Helmholtz equation
    subtract(residual, ΓAx, out=residual)
    scale(α, residual, out=residual)
    add(x, residual, out=residual)
    assert all_close(M(x), residual)  # x.to(residual.device) + α * residual)


@pytest.mark.parametrize("params", parameters)
def test_accretivity(params):
    """Checks norm and lower bound of real part for various operators

     B (medium) should have real part between -0.05 and 1.0 (if we don't put the absorption in V0. If we do, the upper
     limit may be 1.95)
        The operator B-1 should have a norm of less than 0.95

    L + 1 (inverse propagator) should be accretive with a real part of at least 1.0
    (L+1)^-1 (propagator) should be accretive with a real part of at least 0.0
    A (forward) should be accretive with a real part of at least 0.0
    ΓA (preconditioned_operator) should be such that 1-ΓA is a contraction (a norm of less than 1.0)
    """
    domain = construct_domain(**params)
    source = domain.allocate(0)
    assert_accretive(domain_operator(domain, "medium"), "B", norm_max=0.96, norm_offset=1.0)
    assert_accretive(domain_operator(domain, "inverse_propagator"), "L + 1", real_min=1.0)
    assert_accretive(domain_operator(domain, "propagator"), "(L + 1)^-1", real_min=0.0)
    assert_accretive(domain_operator(domain, "preconditioned_operator"), "ΓA", norm_max=1.0, norm_offset=1.0)
    assert_accretive(domain_operator(domain, "richardson", alpha=0.75, source=source), "1- α ΓA", norm_max=1.0)
    assert_accretive(domain_operator(domain, "forward"), "A", real_min=-1e-6, pre_factor=domain.scale)


def assert_accretive(operator, name, *, real_min=None, real_max=None, norm_max=None, norm_offset=None, pre_factor=None):
    """Helper function to check if an operator is accretive, and to compute the norm around a given offset.
    This function constructs a full matrix from the operator, so it only works if the domain is not too large.
    """
    M = full_matrix(operator)
    if pre_factor is not None:
        M *= pre_factor

    if norm_max is not None:
        if norm_offset is not None:
            np.fill_diagonal(M, M.diagonal() - norm_offset)
        norm = np.linalg.norm(M, ord=2)
        print(f"norm {norm:.2e}")
        assert norm <= norm_max, f"operator {name} has norm {norm} > {norm_max}"

    if real_min is not None or real_max is not None:
        M += M.conj().T  # this is 2 × the hermitian part of the operator
        eigs = 0.5 * np.linalg.eigvalsh(M)
        if norm_offset is not None:
            eigs += norm_offset
        if real_min is not None:
            acc = eigs.min()
            print(f"acc {acc:.2e}")
            assert acc >= real_min, f"operator {name} is not accretive, min λ_(A+A*) = {acc} < {real_min}"
        if real_max is not None:
            acc = eigs.max()
            print(f"acc {acc:.2e}")
            assert acc <= real_max, (
                f"operator {name} has eigenvalues that are too large, " f"max λ_(A+A*) = {acc} > {real_max}"
            )
