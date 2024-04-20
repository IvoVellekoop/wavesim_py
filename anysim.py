import torch
import numpy as np
from torch.linalg import norm
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from state import State


def run_algorithm(base: HelmholtzBase, source, alpha=0.75, max_iterations=1000):
    """ AnySim update
    :param base: Helmholtz base parameters
    :return: u (computed field), state (object) """

    # Reset the field u to zero
    base.clear(0)

    # Split the source into subdomains and store in the domain state
    base.set_source(source)

    # state = State(base)
    residual_norm = 0.0

    for i in range(max_iterations):
        print(f'Iteration {i + 1}', end='\r')
        residual_norm += precon_iteration(base, alpha, compute_norm2=True)

    # return u and u_iter cropped to roi, residual arrays, and state object with information on run
    return base.get(0)


def domain_decomp_operators(base):
    """ Construct restriction and extension operators """
    restrict = [[] for _ in range(3)]
    extend = [[] for _ in range(3)]

    one = torch.tensor([[1.]], dtype=torch.complex64)
    if base.total_domains == 1:
        [restrict[dim].append(1.) for dim in range(3)]
        [extend[dim].append(1.) for dim in range(3)]
    else:
        ones = torch.eye(base.domain_size[0], dtype=torch.complex64)
        restrict0_ = []
        n_ext = base.n_roi + base.boundary_pre + base.boundary_post
        [restrict0_.append(torch.zeros((base.domain_size[dim], n_ext[dim]), dtype=torch.complex64))
         for dim in range(base.n_dims)]
        for dim in range(3):
            if base.domain_size[dim] == 1:
                for patch in range(base.n_domains[dim]):
                    restrict[dim].append(one)
                    extend[dim].append(one)
            else:
                for patch in range(base.n_domains[dim]):
                    restrict_mid_ = restrict0_[dim].clone()
                    restrict_mid_[:, slice(patch * base.domain_size[dim],
                                           patch * base.domain_size[dim] + base.domain_size[dim])] = ones
                    restrict[dim].append(restrict_mid_.T)
                    extend[dim].append(restrict_mid_)
    return restrict, extend


def map_domain(x, map_operator, patch):
    """ Map x to extended domain or restricted subdomain """
    if isinstance(map_operator[0][0], float):
        pass
    else:
        for dim in range(3):  # For applying in every dimension
            x = torch.moveaxis(x, dim, -1)  # Transpose to last dimension
            # Apply mapping operator to x
            x = torch.tensordot(x, map_operator[dim][patch[dim]].to(x.device), ([-1, ], [0, ]))
            x = torch.moveaxis(x, -1, dim)  # Transpose back to original dimension
    return x  # Return mapped x


def precon_iteration(base, alpha=0.75, compute_norm2=False):
    """ Run one preconditioned iteration

    x -> x - alpha * ((1-B) x - B(L+1)^-1 (Bx + y)
    = (1-alpha) * x + alpha * (B x - B(L+1)^-1 (Bx + y))

    x_0 = x
    x_1 = B · x_0 = B·x
    x_0 = (1-alpha) · x_0 + alpha · x_1 = (1-alpha) · x + alpha · B·x
    x_1 = x_1 + y = B·x + y
    x_1 = (L+1)^-1 x_1 = (L+1)^-1 (B·x + y)
    x_1 = B x_1 = B (L+1)^-1 (B·x + y)
    x_0 = x_0 - alpha · x_1 = (1-alpha) * x + alpha * (B x - B(L+1)^-1 (Bx + y))
    :param: base: domain or multi-domain to operate on
    :param: alpha: relaxation parameter for the RIchardson iteration
    :param: compute_norm2: when True, returns the squared norm of the residual. Otherwise returns 0.0
    """
    base.medium(0, 1)  # x_1 = B·x
    base.mix(1.0 - alpha, 0, alpha, 1, 0)  # x_0 = (1-alpha) · x + alpha · B·x
    base.add_source(1)  # x_1 = B·x + y
    base.propagator(1, 1)  # x_1 = (L+1)^-1 (B·x + y)
    base.medium(1, 1)
    retval = base.inner_product(1, 1) if compute_norm2 else 0.0
    base.mix(1.0, 0, -alpha, 1, 0)
    return retval
