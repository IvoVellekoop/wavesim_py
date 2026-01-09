from typing import Callable, Optional

from wavesim.engine.array import Array, mix, scale, norm_squared, subtract
from .domain import Domain


def preconditioned_richardson(
    domain: Domain,
    source: Array,
    /,
    *,
    alpha: float = 0.75,
    max_iterations: int = 1000,
    threshold: float = 1.0e-6,
    full_residuals: bool = False,
    callback: Optional[Callable] = None,
    **kwargs,
):
    """
    WaveSim algorithm

    Args:
        domain (Domain): Helmholtz base parameters
        source (Array): source field
        alpha (float): relaxation parameter for the Richardson iteration
        max_iterations (int): maximum number of iterations
        threshold (float): threshold for the residual norm
        full_residuals (bool): when True, returns list of residuals for all iterations. 
                               Otherwise, returns final residual
        callback (Callable): callback function that is called after each iteration.
            The callback function should accept the following arguments:
            - The domain
            - The iteration number. At the end of the iteration, will be called with -1
            - The current field
            - Keyword only arguments:
                - tmp: A temporary array that can be used by the callback
                - residual_norm: The relative residual norm squared
                - max_iterations: the maximum number of iterations
                - threshold: the threshold for the residual norm squared
                - anything passed in **kwargs
            The callback function should return a boolean value:
            - True: Continue the iteration.
            - False: Stop the iteration.

    Returns:
        tuple: u, iteration count, residuals
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    # Reset the field u to zero
    x = domain.allocate(0.0)
    tmp = domain.allocate()
    callback_kwargs = dict(
        kwargs,
        tmp=tmp,
        max_iterations=max_iterations,
        threshold=threshold,
    )

    # compute initial residual norm (with preconditioned source) for normalization
    preconditioner(domain, source, out=tmp)  # tmp = B(L+1)⁻¹y
    init_norm = norm_squared(tmp)  # |tmp|²

    # save list of residuals if requested
    residuals = [] if full_residuals else None

    for i in range(max_iterations):
        # x = x + α Γ⁻¹ (y - A x)
        # normalize residual norm with preconditioned source (i.e., with norm of B(L+1)⁻¹y)
        residual_norm = (
            preconditioned_iteration(domain, x, source=source, out=x, tmp=tmp, alpha=alpha, compute_norm2=True)
            / init_norm
        )

        # callback
        if callback:
            continue_loop = callback(domain, i + 1, x, residual_norm=residual_norm, **callback_kwargs)
            if not continue_loop:
                break

        residuals.append(residual_norm) if full_residuals else None
        if residual_norm < threshold:
            break

    if callback:
        callback(domain, -1, x, residual_norm=residual_norm, **callback_kwargs)  # noqa

    # return u and u_iter cropped to roi, residual arrays, and state object with information on run
    return x.gather(), i + 1, residuals if full_residuals else residual_norm  # noqa


def preconditioned_iteration(
    domain, x: Array, *, source: Array, out: Array, tmp: Array, alpha=0.75, compute_norm2=False
) -> float:
    """Run one preconditioned iteration.

    Args:
        domain: Domain object
        x: input vector. *This array will be overwritten!*
        out: output slot that will receive the result, may be equal to x
        source: slot holding the source term
        tmp: slot for temporary storage, will be overwritten. Cannot be equal to x, may be equal to slot_out
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
    if tmp is x:
        raise ValueError("x and slot_tmp should be different")

    medium, propagator = domain.medium, domain.propagator

    medium(x, out=tmp)  # [tmp] = B·x
    scale(-domain.scale, source, offset=tmp, out=tmp)  # [tmp] = B·x + c·y
    propagator(tmp, out=tmp)  # [tmp] = (L+1)⁻¹ (B·x + c·y)
    subtract(x, tmp, out=tmp)  # [tmp] = x - (L+1)⁻¹ (B·x + c·y)
    medium(tmp, out=tmp)  # [tmp] = B(x - (L+1)⁻¹ (B·x + c·y))

    # optionally compute norm of residual of preconditioned system
    retval = norm_squared(tmp) if compute_norm2 else 0.0

    scale(-alpha, tmp, offset=x, out=out)  # [out] = x - α B x + α B (L+1)⁻¹ (B·x + c·y)
    return retval


def forward(domain: Domain, x: Array, *, out: Array):
    """Evaluates the forward operator A= c⁻¹ (L + V)

    Args:
        domain: Domain object
        x: Input.This array will be overwritten!
        out: output slot that will receive A x
    """
    if x is out:
        raise ValueError("input and output must be different")

    domain.medium(x, out=out)  # (1-V) x
    domain.inverse_propagator(x, out=x)  # (L+1) x
    mix(1.0 / domain.scale, x, -1.0 / domain.scale, out, out=out)  # c⁻¹ (L+V) x


def preconditioned_operator(domain: Domain, x: Array, *, out: Array):
    """Evaluates the preconditioned operator Γ⁻¹ A

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
        x: slot holding input x. This slot will be overwritten!
        out: output slot that will receive A x
    """
    if x is out:
        raise ValueError("x and slot_out must be different")

    domain.medium(x, out=x)  # B x
    domain.propagator(x, out=out)  # (L+1)⁻¹ B x
    domain.medium(out, out=out)  # B (L+1)⁻¹ B x
    subtract(x, out, out=out)  # B - B (L+1)⁻¹ B x


def preconditioner(domain: Domain, x: Array, *, out: Array):
    """Evaluates Γ⁻¹ = c B(L+1)⁻¹

    Args:
        domain: Domain object
        x: Input.
        out: output array that will receive A x
    """
    domain.propagator(x, out=out)  # (L+1)⁻¹ x
    domain.medium(out, out=out)  # B (L+1)⁻¹ x
    scale(domain.scale, out, out=out)  # c B (L+1)⁻¹ x
