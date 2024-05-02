import torch
from wavesim.domain import Domain


def run_algorithm(domain: Domain, source, alpha=0.75, max_iterations=1000):
    """ AnySim update
    :param domain: Helmholtz base parameters
    :return: u (computed field), state (object) """

    # Reset the field u to zero
    domain.clear(0)

    # Split the source into subdomains and store in the domain state
    domain.set_source(source)

    # state = State(base)
    residual_norm = 0.0

    for i in range(max_iterations):
        print(f'Iteration {i + 1}', end='\r')
        residual_norm += preconditioned_iteration(domain, alpha, compute_norm2=True)

    # return u and u_iter cropped to roi, residual arrays, and state object with information on run
    return domain.get(0)


# def domain_decomp_operators(base):
#     """ Construct restriction and extension operators """
#     restrict = [[] for _ in range(3)]
#     extend = [[] for _ in range(3)]
#
#     one = torch.tensor([[1.]], dtype=torch.complex64)
#     if base.total_domains == 1:
#         [restrict[dim].append(1.) for dim in range(3)]
#         [extend[dim].append(1.) for dim in range(3)]
#     else:
#         ones = torch.eye(base.domain_size[0], dtype=torch.complex64)
#         restrict0_ = []
#         n_ext = base.n_roi + base.boundary_pre + base.boundary_post
#         [restrict0_.append(torch.zeros((base.domain_size[dim], n_ext[dim]), dtype=torch.complex64))
#          for dim in range(base.n_dims)]
#         for dim in range(3):
#             if base.domain_size[dim] == 1:
#                 for patch in range(base.n_domains[dim]):
#                     restrict[dim].append(one)
#                     extend[dim].append(one)
#             else:
#                 for patch in range(base.n_domains[dim]):
#                     restrict_mid_ = restrict0_[dim].clone()
#                     restrict_mid_[:, slice(patch * base.domain_size[dim],
#                                            patch * base.domain_size[dim] + base.domain_size[dim])] = ones
#                     restrict[dim].append(restrict_mid_.T)
#                     extend[dim].append(restrict_mid_)
#     return restrict, extend
#
#
# def map_domain(x, map_operator, patch):
#     """ Map x to extended domain or restricted subdomain """
#     if isinstance(map_operator[0][0], float):
#         pass
#     else:
#         for dim in range(3):  # For applying in every dimension
#             x = torch.moveaxis(x, dim, -1)  # Transpose to last dimension
#             # Apply mapping operator to x
#             x = torch.tensordot(x, map_operator[dim][patch[dim]].to(x.device), ([-1, ], [0, ]))
#             x = torch.moveaxis(x, -1, dim)  # Transpose back to original dimension
#     return x  # Return mapped x
#

def preconditioned_iteration(domain, alpha=0.75, compute_norm2=False):
    """ Run one preconditioned iteration

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x

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
    domain.medium(0, 1)  # x_1 = B·x
    domain.mix(1.0 - alpha, 0, alpha, 1, 0)  # x_0 = (1-alpha) · x + alpha · B·x
    domain.add_source(1)  # x_1 = B·x + y
    domain.propagator(1, 1)  # x_1 = (L+1)^-1 (B·x + y)
    domain.medium(1, 1)
    retval = domain.inner_product(1, 1) if compute_norm2 else 0.0
    domain.mix(1.0, 0, -alpha, 1, 0)
    return retval


def forward(domain: Domain, slot_in: int, slot_out: int):
    """ Evaluates the forward operator A= L + 1 - B

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x
     """
    if slot_in == slot_out:
        raise ValueError("slot_in and slot_out must be different")

    domain.medium(slot_in, slot_out)  # (1-V) x
    domain.inverse_propagator(slot_in, slot_in)  # (L+1) x
    domain.mix(1.0, slot_in, -1.0, slot_out, slot_out)  # (L+V) x


def preconditioned_operator(domain: Domain, slot_in: int, slot_out: int):
    """ Evaluates the preconditioned operator B(L+1)^-1 A = B - B (L+1)^-1 B

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x
    """
    if slot_in == slot_out:
        raise ValueError("slot_in and slot_out must be different")

    domain.medium(slot_in, slot_in)  # B x
    domain.propagator(slot_in, slot_out)  # (L+1)^-1 B x
    domain.medium(slot_out, slot_out)  # B (L+1)^-1 B x
    domain.mix(1.0, slot_in, -1.0, slot_out, slot_out)  # B - B (L+1)^-1 B x


def preconditioner(domain: Domain, slot_in: int, slot_out: int):
    """ Evaluates Γ^-1 = B(L+1)^-1

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x
    """
    domain.propagator(slot_in, slot_in)  # (L+1)^-1 x
    domain.medium(slot_in, slot_out)  # B (L+1)^-1 x


def domain_operator(domain: Domain, function: str):
    """Constructs an operator that takes a field and returns the medium operator 1-V applied to it

    todo: this is currently very inefficient because of the overhead of copying data to and from the device
    """

    def potential_(domain, slot_in, slot_out):
        domain.medium(slot_in, slot_out)
        domain.mix(1.0, slot_in, -1.0, slot_out, slot_out)

    fn = {
        'medium': domain.medium,
        'propagator': domain.propagator,
        'inverse_propagator': domain.inverse_propagator,
        'potential': potential_,
        'forward': lambda slot_in, slot_out: forward(domain, slot_in, slot_out),
        'preconditioned_operator': lambda slot_in, slot_out: preconditioned_operator(domain, slot_in, slot_out),
        'preconditioner': lambda slot_in, slot_out: preconditioner(domain, slot_in, slot_out),
    }[function]

    def operator_(x):
        if isinstance(x, float) or isinstance(x, int):
            if x != 0:
                raise ValueError("Cannot set a field to a scalar to a field, only scalar 0.0 is supported")
            domain.clear(0)
        else:
            domain.set(0, x)
        fn(0, 1)
        return domain.get(1, copy=True)

    return operator_
