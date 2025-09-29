import numpy as np
from typing import Callable, Optional, Sequence

from .engine import BlockArray, CupyArray, NumpyArray, SparseArray, Array
from .utilities import add_absorbing_boundaries
from .helmholtzdomain import Helmholtz
from .maxwelldomain import Maxwell
from .iteration import preconditioned_richardson


def simulate(
    permittivity: np.ndarray | NumpyArray,
    sources: Sequence[tuple[np.ndarray, Sequence[int]]],
    wavelength: float,
    pixel_size: float,
    boundary_width: float = 1.,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    use_gpu: bool = True,
    n_domains: tuple[int, int, int] = None,
    max_iterations: int = 100000,
    threshold: float = 1.0e-6,
    alpha: float = 0.75,
    full_residuals: bool = False,
    crop_boundaries: bool = True,
    callback: Optional[Callable] = None,
    **kwargs,
):
    """Simulate the Helmholtz equation or time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials in 3D 
    with a given permittivity grid and source(s). This is an advanced version of the simulate function for defining custom sources.
    Args:
        permittivity: np.ndarray, the permittivity grid. Must be a 3D array of complex numbers.
        sources: list of sources. Each source is a tuple of (array of complex numbers containing source values, position).
            array of complex numbers: the source data. Must be a 3D array of complex numbers, and smaller than or equal to permittivity.shape.
            position: a tuple of 3 or 4 integers, the position of the source in pixels. 
                3 integers for solving the Helmholtz equation (scalar).
                4 integers (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector).
        wavelength: float, wavelength in micrometer (μm).
        pixel_size: float, pixel size in micrometer (μm) (the same for x, y, and z dimensions). Pixel size must be < wavelength/2, but we recommend using a pixel size of wavelength/4.
        boundary_width: float, boundary width in micrometer (μm). Default is 1.0 μm.
        periodic: tuple[bool, bool, bool], periodicity in each direction. Default is (False, False, False).
            This means that the simulation domain is not periodic in any direction and a wrapping correction is performed in each direction.
            If True in a direction, the simulation domain is periodic in that direction and no wrapping correction is performed.
        use_gpu: bool, if True, use CupyArray for GPU acceleration, else NumpyArray. Default is True.
        n_domains: tuple[int, int, int], number of domains in each direction (None for single domain). Default is None.
        max_iterations: int, maximum number of iterations. Default is 100000.
        threshold: float, tolerance for convergence. Default is 1.0e-6.
        alpha: float, relaxation factor for the preconditioned Richardson method. Default is 0.75.
        full_residuals: bool, if True, return full residuals, and if False, return only the final residual norm. Default is False.
        crop_boundaries: bool, if True, crop the boundaries of the field to remove the absorbing boundaries. Default is True.
        callback: Optional[Callable], callback function that is called after each iteration.
    Returns:
        u: np.ndarray, the field in the simulation domain
        iterations: int, number of iterations taken to converge
        residual_norm: float, norm of the residual at convergence
    """
    # Validate input
    if not isinstance(permittivity, np.ndarray) and not isinstance(permittivity, Array):
        raise TypeError("permittivity must be a numpy array or WaveSim Array")
    if pixel_size >= wavelength / 2:
        raise ValueError("Pixel size must be < wavelength/2")

    factory = CupyArray if use_gpu else NumpyArray  # resolve to NumpyArray for CPU, CupyArray for GPU acceleration

    # Parameters
    boundary_width = (np.asarray(boundary_width) / pixel_size).astype(int)  # Boundary width in pixels

    # permittivity shape adjusted to include the absorbing boundaries
    permittivity, roi = add_absorbing_boundaries(factory(permittivity), boundary_width, strength=1.0, periodic=periodic)

    # Create a SparseArray for the sources
    top_left = [s.start for s in roi]
    shape = permittivity.shape

    # Check if problem is scalar or vector (3 or 4 elements in source position tuple, respectively)
    if any(len(s[1])==4 for s in sources):  # sources[i][0] is the source data and sources[i][1] is the source position
        vectorial = True  # Solve time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials
        # Add polarization dimension to the following variables 
        roi = (slice(None),) + roi
        top_left = [0] + top_left
        shape = (3,) + shape
    else:
        vectorial = False  # Solve the Helmholtz equation
    data = [factory(s[0], dtype=permittivity.dtype) for s in sources]
    positions = [np.asarray(s[1]) + np.asarray(top_left) for s in sources]
    source = SparseArray(data, at=positions, shape=shape)

    if n_domains is not None:
        permittivity = BlockArray(permittivity, n_domains=n_domains, factories=factory)

    if vectorial:
        domain = Maxwell
    else:
        domain = Helmholtz
    domain = domain(
        permittivity=permittivity, 
        pixel_size=pixel_size, 
        wavelength=wavelength, 
        periodic=periodic
    )

    u, iterations, residual_norm = preconditioned_richardson(
        domain,
        source,
        alpha=alpha,
        max_iterations=max_iterations,
        threshold=threshold,
        full_residuals=full_residuals,
        callback=callback,
        **kwargs,
    )
    # crop the field
    if crop_boundaries:
        u = u[*roi].squeeze()

    return u, iterations, residual_norm
