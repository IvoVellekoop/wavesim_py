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


def preconditioned_iteration(domain, slot_in: int = 0, slot_out: int = 0, slot_tmp: int = 1, alpha=0.75,
                             compute_norm2=False):
    """ Run one preconditioned iteration.

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_tmp: slot for temporary storage. Cannot be equal to slot_in, may be equal to slot_out
        slot_out: output slot that will receive the result

    Richardson iteration:
        x -> x + α (y - A x)

    Preconditioned Richardson iteration:
        x -> x + α (Γ⁻¹ y - Γ⁻¹ A x)
            = x + α B (L+1)⁻¹ (y - A x)
            = x + α B (Α+Β)⁻¹ (y - A x)
            = x + α B (Α+Β)⁻¹ (y - [A+B] x + B x)
            = x + α B [(Α+Β)⁻¹ (y + B x) - x]
            = [x - α B x] + α B (Α+Β)⁻¹ (y + B x)

    :param: base: domain or multi-domain to operate on
    :param: alpha: relaxation parameter for the RIchardson iteration
    :param: compute_norm2: when True, returns the squared norm of the residual. Otherwise, returns 0.0
    """
    if slot_tmp == slot_in:
        raise ValueError("slot_in and slot_tmp should be different")

    domain.medium(slot_in, slot_tmp)  # [tmp] = B·x
    domain.mix(1.0, slot_in, -alpha, slot_tmp, slot_in)  # [in] = x - α B x
    domain.add_source(slot_tmp)  # [tmp] = B·x + y
    domain.propagator(slot_tmp, slot_tmp)  # [tmp] = (L+1)⁻¹ (B·x + y)
    domain.medium(slot_tmp, slot_tmp)  # [tmp] = B (L+1)⁻¹ (B·x + y)
    # optionally compute norm of residual of preconditioned system
    retval = domain.inner_product(slot_tmp, slot_tmp) if compute_norm2 else 0.0
    domain.mix(1.0, slot_in, alpha, slot_tmp, slot_out)  # [out] = x - α B x + α B (L+1)⁻¹ (B·x + y)
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
    """ Evaluates the preconditioned operator B(L+1)^-1 c A = B - B (L+1)^-1 B

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


def domain_operator(domain: Domain, function: str, **kwargs):
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
        'richardson': lambda slot_in, slot_out: preconditioned_iteration(domain, slot_in, slot_out, slot_out, **kwargs)
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
