import numpy as np

from wavesim.engine.array import Array, subtract
from wavesim.domain import Domain
from wavesim.engine.index_utils import shape_like
from wavesim.iteration import forward, preconditioned_operator, preconditioner, preconditioned_iteration
from wavesim.engine.numpyarray import NumpyArray


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
