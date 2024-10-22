<a id="root-label"></a>

# Wavesim

<!-- NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY. -->

## What is Wavesim?

Wavesim is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method [[1](#id25), [2](#id19), [3](#id14), [4](#id13)], Wavesim is based on the Modified Born Series approach, which has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD  [[5](#id15), [6](#id21)].

This package [[7](#id23)] is a Python implementation of the Modified Born Series (MBS) approach for solving the Helmholtz equation in arbitrarily large media through domain decomposition [[8](#id11)]. With this new framework, we simulated a complex 3D structure of a remarkable $315\times 315\times 315$ wavelengths $\left( 3.1\cdot 10^7 \right)$ in size in just $379$ seconds by solving over two GPUs. This represents a factor of $1.93$ increase over the largest possible simulation on a single GPU without domain decomposition.

When using Wavesim in your work, please cite [[5](#id15), [8](#id11)], and [[7](#id23)] for the code. Examples and documentation for this project are available at [Read the Docs](https://wavesim.readthedocs.io/en/latest/) [[9](#id22)]. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website [www.wavesim.org](https://www.wavesim.org/).

## Installation

Wavesim requires [Python 3.11.0 and above](https://www.python.org/downloads/) and uses [PyTorch](https://pytorch.org/) for GPU acceleration.

We recommend using [Miniconda](https://docs.anaconda.com/miniconda/) (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in [environment.yml](https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml)) within a conda environment. If you prefer to create a virtual environment without using Miniconda/Anaconda, you can use [requirements.txt](https://github.com/IvoVellekoop/wavesim_py/blob/main/requirements.txt) for dependencies. The steps that follow are for a Miniconda installation.

1. **Download Miniconda**, choosing the appropriate [Python installer](https://docs.anaconda.com/miniconda/) for your operating system (Windows/macOS/Linux).
2. **Install Miniconda**, following the [installation instructions](https://docs.anaconda.com/miniconda/miniconda-install/) for your OS. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. (If you cannot immediately activate conda, close and re-open your terminal window to make the changes take effect).
3. **Test your installation**. Open Anaconda Prompt and run the below command. Alternatively, open an editor like [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/), select the Python interpreter in the `miniconda3/` directory with the label `('base')`, and run the command:
   ```default
   conda list
   ```

> A list of installed packages appears if it has been installed correctly.
1. **Set up a conda environment**. Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don’t work. Create a new environment using [environment.yml](https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml) and activate.:
   ```default
   conda env create -f environment.yml
   conda activate wavesim
   ```

> The [Miniconda environment management guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more details if you need them.

## Running the code

Once the virtual environment is set up with all the required packages, you are ready to run the code. You can go through any of the scripts in the `examples` [directory](https://github.com/IvoVellekoop/wavesim_py/tree/main/examples) for the basic steps needed to run a simulation. The directory contains two examples each of 1D, 2D, and 3D problems.

You can run the code with just three inputs:

* `permittivity`, i.e. refractive index distribution squared (a 4-dimensional array on a regular grid),
* `periodic`, a tuple of three booleans to indicate whether the domain is periodic in each dimension [`True`] or not [`False`], and
* `source`, the same size as permittivity.

[Listing 1.1](#helmholtz-1d-analytical) shows a simple example of a 1D problem with a homogeneous medium ([helmholtz_1d_analytical.py](https://github.com/IvoVellekoop/wavesim_py/blob/main/examples/helmholtz_1d_analytical.py)) to explain these and other inputs.

<a id="helmholtz-1d-analytical"></a>
```python
"""
Helmholtz 1D analytical test
============================
Test to compare the result of Wavesim to analytical results. 
Compare 1D free-space propagation with analytic solution.
"""

import torch
import numpy as np
from time import time
import sys
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain  # when number of domains is 1
from wavesim.multidomain import MultiDomain  # for domain decomposition, when number of domains is >= 1
from wavesim.iteration import run_algorithm  # to run the wavesim iteration
from wavesim.utilities import analytical_solution, preprocess, relative_error
from __init__ import plot


# Parameters
wavelength = 1.  # wavelength in micrometer (um)
n_size = (256, 1, 1)  # size of simulation domain (in pixels in x, y, and z direction)
n = np.ones(n_size, dtype=np.complex64)  # permittivity (refractive index²) map
boundary_widths = 16  # width of the boundary in pixels

# return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n**2, boundary_widths)  # permittivity is n², but uses the same variable n

# Source term. This way is more efficient than dense tensor
indices = torch.tensor([[0 + boundary_array[i] for i, v in enumerate(n_size)]]).T  # Location: center of the domain
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# Set up the domain operators (HelmholtzDomain() or MultiDomain() depending on number of domains)
# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
domain = HelmholtzDomain(permittivity=n, periodic=periodic, wavelength=wavelength)
# # OR. Uncomment to test domain decomposition
# periodic = (False, True, True)  # wrapping correction
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, n_domains=(3, 1, 1))

# Run the wavesim iteration and get the computed field
start = time()
u_computed, iterations, residual_norm = run_algorithm(domain, source, max_iterations=2000)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')
u_computed = u_computed.squeeze().cpu().numpy()[boundary_widths:-boundary_widths]
u_ref = analytical_solution(n_size[0], domain.pixel_size, wavelength)

# Compute relative error with respect to the analytical solution
re = relative_error(u_computed, u_ref)
print(f'Relative error: {re:.2e}')
threshold = 1.e-3
assert re < threshold, f"Relative error higher than {threshold}"

# Plot the results
plot(u_computed, u_ref, re)
```

Apart from the inputs `permittivity`, `periodic`, and `source`, all other parameters have defaults. Details about these are given below (with the default values, if defined).

Parameters in the `Domain` class: `HelmholtzDomain` or `MultiDomain`

* `permittivity`: 4-dimensional array with refractive index-squared distribution in x, y, and z direction, and a polarization dimension (unused in Helmholtz case). To set up a 1 or 2-dimensional problem, leave the other dimension(s) as 1.
* `periodic`: indicates for each dimension whether the simulation is periodic (`True`) or not (`False`). For periodic dimensions, i.e., `periodic` `= [True, True, True]`, the field is wrapped around the domain.
* `pixel_size` `:float = 0.25`: points per wavelength.
* `wavelength` `:float = None`: wavelength: wavelength in micrometer (um). If not given, i.e. `None`, it is calculated as `1/pixel_size = 4 um`.
* `n_domains` `: tuple[int, int, int] = (1, 1, 1)`: number of domains to split the simulation into. If the domain size is not divisible by n_domains, the last domain will be slightly smaller than the other ones. If `(1, 1, 1)`, indicates no domain decomposition.
* `n_boundary` `: int = 8`: number of points used in the wrapping and domain transfer correction. Applicable when `periodic` is False in a dimension, or `n_domains > 1` in a dimension.
* `device` `: str = None`:
  > * `'cpu'` to use the cpu,
  > * `'cuda:x'` to use a specific cuda device
  > * `'cuda'` or a list of strings, e.g., `['cuda:0', 'cuda:1']`, to distribute the simulation over the available/given cuda devices in a round-robin fashion
  > * `None`, which is equivalent to `'cuda'` if cuda devices are available, and `'cpu'` if they are not.
* `debug` `: bool = False`: set to `True` for testing to return `inverse_propagator_kernel` as output.

Parameters in the `run_algorithm()` function

* `domain`: the domain object created by HelmholtzDomain() or MultiDomain()
* `source`: source term, a 4-dimensional array, with the same size as permittivity. Set up amplitude(s) at the desired location(s), following the same principle as permittivity for 1, 2, or 3-dimensional problems.
* `alpha` `: float = 0.75`: relaxation parameter for the Richardson iteration
* `max_iterations` `: int = 1000`: maximum number of iterations
* `threshold` `: float = 1.e-6`: threshold for the residual norm for stopping the iteration

## Acknowledgements

This work was supported by the European Research Council’s Proof of Concept Grant n° [101069402].

## Conflict of interest statement

The authors declare no conflict of interest.

## References

<a name="id25"></a>1

Kane Yee. Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media. *IEEE Transactions on Antennas and Propagation*, 14(3):302–307, 1966. [doi:10.1109/TAP.1966.1138693](https://doi.org/10.1109/TAP.1966.1138693).

<a name="id19"></a>2

Allen Taflove and Susan C Hagnes. *Computational electrodynamics: The Finite-Difference Time-Domain Method*. Artech House, 1995.

<a name="id14"></a>3

Ardavan F. Oskooi, David Roundy, Mihai Ibanescu, Peter Bermel, J.D. Joannopoulos, and Steven G. Johnson. MEEP: a flexible free-software package for electromagnetic simulations by the FDTD method. *Computer Physics Communications*, 181(3):687–702, 2010. [doi:10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).

<a name="id13"></a>4

Majid Nabavi, M.H. Kamran Siddiqui, and Javad Dargahi. A new 9-point sixth-order accurate compact finite-difference method for the Helmholtz equation. *Journal of Sound and Vibration*, 307(3):972–982, 2007. [doi:10.1016/j.jsv.2007.06.070](https://doi.org/10.1016/j.jsv.2007.06.070).

<a name="id15"></a>5

Gerwin Osnabrugge, Saroch Leedumrongwatthanakun, and Ivo M Vellekoop. A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. *Journal of Computational Physics*, 322:113–124, 2016. [doi:10.1016/j.jcp.2016.06.034](https://doi.org/10.1016/j.jcp.2016.06.034).

<a name="id21"></a>6

Tom Vettenburg and Ivo M. Vellekoop. A universal matrix-free split preconditioner for the fixed-point iterative solution of non-symmetric linear systems. *arXiv preprint*, 2023. [arXiv:2207.14222](https://arxiv.org/abs/2207.14222).

<a name="id23"></a>7

Swapnil Mache and Ivo M. Vellekoop. Wavesim - a python package for wave propagation simulation. URL: [https://github.com/IvoVellekoop/wavesim_py](https://github.com/IvoVellekoop/wavesim_py).

<a name="id11"></a>8

Swapnil Mache and Ivo M. Vellekoop. Domain decomposition of the modified born series approach for large-scale wave propagation simulations. *arXiv preprint*, 2024. [arXiv:2410.02395](https://arxiv.org/abs/2410.02395).

<a name="id22"></a>9

Wavesim documentation. URL: [https://wavesim.readthedocs.io/en/latest/](https://wavesim.readthedocs.io/en/latest/).
