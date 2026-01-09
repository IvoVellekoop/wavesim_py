from . import create_medium
from . import create_source
from . import plotting
from . import utilities
from .create_medium import (
    random_permittivity, 
    sphere_permittivity, 
    cuboids_permittivity
)
from .create_source import (
    point_source,
    plane_wave,
    gaussian_beam,
    source_angled,
    expand_source_shape,
    check_source_parameters,
    polarization_
)
from .plotting import plot_computed, plot_computed_and_reference
from .utilities import (
    normalize,
    add_absorbing_boundaries,
    laplace_kernel_1d,
    diff_kernel_1d,
    _ulp,
    plot_difference,
    all_close,
    full_matrix,
    max_abs_error,
    max_relative_error,
    relative_error,
    plot_fields,
    relative_error_check,
    analytical_solution
)
