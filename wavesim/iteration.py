from .domain import Domain
from .utilities import is_zero


def run_algorithm(domain: Domain, source, alpha=0.75, max_iterations=1000, threshold=1.e-6, full_residuals=False):
    """ WaveSim update

    :param domain: Helmholtz base parameters
    :param source: source field
    :param alpha: relaxation parameter for the Richardson iteration
    :param max_iterations: maximum number of iterations
    :param threshold: threshold for the residual norm
    :param full_residuals: when True, returns list of residuals for all iterations. Otherwise, returns final residual
    :return: u, iteration count, residuals """

    # Reset the field u to zero
    slot_x = 0
    slot_tmp = 1
    domain.clear(slot_x)
    domain.set_source(source)

    # compute initial residual norm (with preconditioned source) for normalization
    domain.add_source(slot_x, weight=1.)  # [x] = y
    preconditioner(domain, slot_x, slot_x)  # [x] = B(L+1)⁻¹y
    init_norm_inv = 1 / domain.inner_product(slot_x, slot_x)  # inverse of initial norm: 1 / norm([x])
    domain.clear(slot_x)  # Clear [x]

    # save list of residuals if requested
    residuals = [] if full_residuals else None

    for i in range(max_iterations):
        residual_norm = preconditioned_iteration(domain, slot_x, slot_x, slot_tmp, alpha, compute_norm2=True)
        # normalize residual norm with preconditioned source (i.e., with norm of B(L+1)⁻¹y)
        residual_norm = residual_norm * init_norm_inv  # norm(B(x - (L+1)⁻¹ (B·x + c·y))) / norm(B(L+1)⁻¹y)
        print('.', end='', flush=True) if (i + 1) % 100 == 0 else None
        residuals.append(residual_norm) if full_residuals else None
        if residual_norm < threshold:
            break

    # return u and u_iter cropped to roi, residual arrays, and state object with information on run
    return domain.get(slot_x), (i + 1), residuals if full_residuals else residual_norm


def preconditioned_iteration(domain, slot_in: int = 0, slot_out: int = 0, slot_tmp: int = 1, alpha=0.75,
                             compute_norm2=False):
    """ Run one preconditioned iteration.

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive the result
        slot_tmp: slot for temporary storage. Cannot be equal to slot_in, may be equal to slot_out
        alpha: relaxation parameter for the Richardson iteration
        compute_norm2: when True, returns the squared norm of the residual. Otherwise, returns 0.0

    Richardson iteration:
        x -> x + α (y - A x)

    Preconditioned Richardson iteration:
        x -> x + α Γ⁻¹ (y - A x)
            = x + α c B (L+1)⁻¹ (y - A x)
            = x + α c B (L+1)⁻¹ (y - c⁻¹ [L+V] x)
            = x + α c B (L+1)⁻¹ (y + c⁻¹ [1-V] x - c⁻¹ [L+1] x)
            = x + α B [(L+1)⁻¹ (c y + B x) - x]
            = x - α B x + α B (L+1)⁻¹ (c y + B x)
    """
    if slot_tmp == slot_in:
        raise ValueError("slot_in and slot_tmp should be different")

    domain.medium(slot_in, slot_tmp, mnum=0)  # [tmp] = B·x
    domain.add_source(slot_tmp, domain.scale)  # [tmp] = B·x + c·y
    domain.propagator(slot_tmp, slot_tmp)  # [tmp] = (L+1)⁻¹ (B·x + c·y)
    domain.mix(1.0, slot_in, -1.0, slot_tmp, slot_tmp)  # [tmp] = x - (L+1)⁻¹ (B·x + c·y)
    domain.medium(slot_tmp, slot_tmp, mnum=1)  # [tmp] = B(x - (L+1)⁻¹ (B·x + c·y))
    # optionally compute norm of residual of preconditioned system
    retval = domain.inner_product(slot_tmp, slot_tmp) if compute_norm2 else 0.0
    domain.mix(1.0, slot_in, -alpha, slot_tmp, slot_out)  # [out] = x - α B x + α B (L+1)⁻¹ (B·x + c·y)
    return retval


def forward(domain: Domain, slot_in: int, slot_out: int):
    """ Evaluates the forward operator A= c⁻¹ (L + V)

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x
     """
    if slot_in == slot_out:
        raise ValueError("slot_in and slot_out must be different")

    domain.medium(slot_in, slot_out)  # (1-V) x
    domain.inverse_propagator(slot_in, slot_in)  # (L+1) x
    domain.mix(1.0 / domain.scale, slot_in, -1.0 / domain.scale, slot_out, slot_out)  # c⁻¹ (L+V) x


def preconditioned_operator(domain: Domain, slot_in: int, slot_out: int):
    """ Evaluates the preconditioned operator Γ⁻¹ A

    Where Γ⁻¹ = c B (L+1)⁻¹

    Note: the scale factor c that makes A accretive and V a contraction is
       included in the preconditioner. The Richardson step size is _not_.

    Operator A is the original non-scaled operator, and we have (L+V) = c A
    Then:

        Γ⁻¹ A    = c B (L+1)⁻¹ A
                    = c B (L+1)⁻¹ c⁻¹ (L+V)
                    = B (L+1)⁻¹ (L+V)
                    = B (L+1)⁻¹ ([L+1] - [1-V])
                    = B - B (L+1)⁻¹ B

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x
    """
    if slot_in == slot_out:
        raise ValueError("slot_in and slot_out must be different")

    domain.medium(slot_in, slot_in)  # B x
    domain.propagator(slot_in, slot_out)  # (L+1)⁻¹ B x
    domain.medium(slot_out, slot_out)  # B (L+1)⁻¹ B x
    domain.mix(1.0, slot_in, -1.0, slot_out, slot_out)  # B - B (L+1)⁻¹ B x


def preconditioner(domain: Domain, slot_in: int, slot_out: int):
    """ Evaluates Γ⁻¹ = c B(L+1)⁻¹

    Args:
        domain: Domain object
        slot_in: slot holding input x. This slot will be overwritten!
        slot_out: output slot that will receive A x
    """
    domain.propagator(slot_in, slot_in)  # (L+1)⁻¹ x
    domain.medium(slot_in, slot_out)  # B (L+1)⁻¹ x
    domain.mix(0.0, slot_out, domain.scale, slot_out, slot_out)  # c B (L+1)⁻¹ x


def domain_operator(domain: Domain, function: str, **kwargs):
    """Constructs various operators by combining calls to 'medium', 'propagator', etc.

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
        if is_zero(x):
            domain.clear(0)
        else:
            domain.set(0, x)
        fn(0, 1)
        return domain.get(1, copy=True)

    return operator_
