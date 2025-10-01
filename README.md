<a id="root-label"></a>

# Wavesim

<!-- NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY. -->

## What is Wavesim?

Wavesim is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method [[1](#id27), [2](#id21), [3](#id16), [4](#id15)], Wavesim is based on the modified Born series (MBS) approach, which has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD  [[5](#id17), [6](#id23)].

This package [[7](#id25)] is a Python implementation of the MBS approach for solving the Helmholtz equation in arbitrarily large media through domain decomposition [[8](#id13)], and time-harmonic Maxwell’s equations for non-magnetic and non-birefringent materials. With this new framework, we simulated a complex 3D structure of a remarkable $315\times 315\times 315$ wavelengths $\left( 3.1\cdot 10^7 \right)$ in size in just $1.4$ hours by solving over two GPUs. This represents a factor of $1.93$ increase over the largest possible simulation on a single GPU without domain decomposition.

When using Wavesim in your work, please cite:

> [[5](#id17)] [Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. *Journal of computational physics, 322*, 113-124.](https://doi.org/10.1016/j.jcp.2016.06.034)

> [[8](#id13)] [Mache, S., & Vellekoop, I. M. (2024). Domain decomposition of the modified Born series approach for large-scale wave propagation simulations. *arXiv preprint arXiv:2410.02395*.](https://arxiv.org/abs/2410.02395)

If you use the code in your research, please cite this repository as well [[7](#id25)].

Examples and documentation for this project are available at [Read the Docs](https://wavesim.readthedocs.io/en/latest/) [[9](#id24)]. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website [www.wavesim.org](https://www.wavesim.org/).

## Installation

Wavesim requires [Python 3.11.0 and above](https://www.python.org/downloads/) and uses [CuPy](https://cupy.dev/) for GPU acceleration.

First, clone the repository and navigate to the directory:

```default
git clone https://github.com/IvoVellekoop/wavesim_py.git
cd wavesim_py
```

Then, you can install the dependencies in a couple of ways:

[1. Using pip](#pip-installation)

[2. Using conda](#conda-installation)

[3. Using Poetry](#poetry-installation)

We recommend working with a virtual environment to avoid conflicts with other packages.

<a id="pip-installation"></a>

### 1. **Using pip**

If you prefer to use pip, you can install the required packages using [requirements.txt](https://github.com/IvoVellekoop/wavesim_py/blob/main/requirements.txt):

1. **Create a virtual environment and activate it** (optional but recommended)
   * First, [create a virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) using the following command:
     ```default
     python -m venv path/to/venv
     ```
   * Then, activate the virtual environment. The command depends on your operating system and shell ([How venvs work](https://docs.python.org/3/library/venv.html#how-venvs-work)):
     ```default
     source path/to/venv/bin/activate    # for Linux/macOS
     path/to/venv/Scripts/activate.bat   # for Windows (cmd)
     path/to/venv/Scripts/Activate.ps1   # for Windows (PowerShell)
     ```
2. **Install packages**:
   ```default
   pip install -r requirements.txt
   ```

<a id="conda-installation"></a>

### 2. **Using conda**

We recommend using [Miniconda](https://docs.anaconda.com/miniconda/) (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in [environment.yml](https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml)) within a conda environment.

1. **Download Miniconda**, choosing the appropriate [Python installer](https://docs.anaconda.com/miniconda/) for your operating system (Windows/macOS/Linux).
2. **Install Miniconda**, following the [installation instructions](https://docs.anaconda.com/miniconda/miniconda-install/) for your OS. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. (If you cannot immediately activate conda, close and re-open your terminal window to make the changes take effect).
3. **Test your installation**. Open Anaconda Prompt and run the below command. Alternatively, open an editor like [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/), select the Python interpreter in the `miniconda3/` directory with the label `('base')`, and run the command:
   ```default
   conda list
   ```

   A list of installed packages appears if it has been installed correctly.
4. **Set up a conda environment**. Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don’t work. Create a new environment using [environment.yml](https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml) and activate:
   ```default
   conda env create -f environment.yml
   conda activate wavesim
   ```

   The [Miniconda environment management guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more details if you need them.

   Alternatively, you can create a conda environment with a specific Python version, and then use the [requirements.txt](https://github.com/IvoVellekoop/wavesim_py/blob/main/requirements.txt) file to install the dependencies:
   ```default
   conda create -n wavesim python'>=3.11.0,<3.13'
   conda activate wavesim
   pip install -r requirements.txt
   ```

<a id="poetry-installation"></a>

### 3. **Using Poetry**

1. Install [Poetry](https://python-poetry.org/).
2. Install dependencies by running the following command:
   ```default
   poetry install
   ```

   To run tests using pytest, you can install the development dependencies as well:
   ```default
   poetry install --with dev
   ```
3. [Activate](https://python-poetry.org/docs/managing-environments/#activating-the-environment) the virtual environment created by Poetry.

## Running the code

Once the virtual environment is set up with all the required packages, you are ready to run the code. You can go through any of the scripts in the `examples` [directory](https://github.com/IvoVellekoop/wavesim_py/tree/main/examples) for the basic steps needed to run a simulation. The directory contains examples of 1D, 2D, and 3D problems, for the Helmholtz equation and Maxwell’s equations for non-magnetic and non-birefringent materials.

You can run the code with just four inputs to the `simulate` function:

* `permittivity`, i.e. refractive index distribution squared (a 3-dimensional array on a regular grid),
* `sources`, list of sources.
* `wavelength` `:float`: wavelength in micrometer (μm).
* `pixel_size` `:float`: pixel size in micrometer (μm).

[Listing 1.1](#helmholtz-1d-analytical) shows a simple example of a 1D problem with a homogeneous medium ([helmholtz_1d_analytical.py](https://github.com/IvoVellekoop/wavesim_py/blob/main/examples/helmholtz_1d_analytical.py)) to explain these and other inputs.

<a id="helmholtz-1d-analytical"></a>
```python
"""
Helmholtz 1D analytical test
============================
Test to compare the result of Wavesim to analytical results. 
Compare 1D free-space propagation with analytic solution.
"""

import numpy as np
from time import time

import sys
sys.path.append(".")
from wavesim.utilities.create_source import point_source
from wavesim.simulate import simulate
from tests import analytical_solution, all_close, relative_error
from examples import plot_computed_and_reference

# Parameters
wavelength = 0.5  # wavelength in micrometer (μm)
pixel_size = wavelength / 10  # pixel size in micrometer (μm)

# Create a refractive index map
sim_size = 128  # size of simulation domain in x direction in micrometer (μm)
n_size = (int(sim_size / pixel_size), 1, 1)  # We want to set up a 1D simulation, so y and z are 1.
permittivity = np.ones(n_size, dtype=np.complex64)  # permittivity (refractive index squared) of 1

# Create a point source at the center of the domain
source_values, source_position = point_source(
    position=[sim_size//2, 0, 0],  # source center position in the center of the domain in micrometer (μm)
    pixel_size=pixel_size
)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=5,  # Boundary width in micrometer (μm) 
    periodic=(False, True, True)  # Periodic boundary conditions in the y and z directions
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# Compute the analytical solution
c = np.arange(0, sim_size, pixel_size)
c = c - c[source_position[0]]
u_ref = analytical_solution(c, wavelength)

# Compute relative error with respect to the analytical solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re)

threshold = 1.0e-3
assert re < threshold, f"Relative error higher than {threshold}"
assert all_close(u, u_ref, rtol=4e-2)
```

Apart from the inputs `permittivity`, `sources`, `pixel_size`, and `wavelength`, all other parameters have defaults. Details about all parameters are given below (with the default values, if defined).

* `permittivity`: 3-dimensional array with refractive index-squared distribution in x, y, and z direction. To set up a 1 or 2-dimensional problem, leave the other dimension(s) as 1.
* `sources`: list of sources, where each source is a tuple of (array of complex numbers containing source values, position). The array of complex numbers is the source data. Must be a 3D array of complex numbers, and smaller than or equal to permittivity.shape. The position should be a tuple of 3 or 4 integers, the position of the source in pixels (3 integers for solving the Helmholtz equation (scalar), and 4 integers (polarization axis, x, y, z) for solving time-harmonic Maxwell’s equations for non-magnetic and non-birefringent materials (vector)).
* `wavelength` `:float`: wavelength in micrometer (μm).
* `pixel_size` `:float`: pixel size in micrometer (μm). Pixel size must be < wavelength/2, but we recommend using a pixel size of wavelength/4.
* `boundary_width` `: float = 1.`: width of the absorbing boundaries in micrometer (μm). The boundaries are placed on the outside of the domain defined by permittivity.
* `periodic` `: tuple[bool, bool, bool] = (False, False, False)`: indicates for each dimension whether the simulation is periodic (`True`) or not (`False`). For periodic dimensions, i.e., `periodic` `= [True, True, True]`, the field is wrapped around the domain.
* `use_gpu` `: bool = True`: if true use CupyArray for GPU acceleration, else NumpyArray.
* `n_domains` `: tuple[int, int, int] = None`: number of domains in each direction (None for single domain).
* `max_iterations` `: int = 100000`: maximum number of iterations.
* `threshold` `: float = 1.e-6`: threshold for the residual norm for stopping the iteration.
* `alpha` `: float = 0.75`: relaxation parameter for the preconditioned Richardson method.
* `full_residuals` `: bool = False`: when True, returns list of residuals for all iterations. Otherwise, only returns the residual for the final iteration.
* `crop_boundaries` `: bool = True`: if True, crop the boundaries of the field to remove the absorbing boundaries.
* `callback` `: Optional[Callable]`: callback function that is called after each iteration.

The `simulate` function returns the field, the number of iterations, and the residual norm.

* `u` `: np.ndarray`: the field in the simulation domain.
* `iterations` `: int`: number of iterations taken to converge.
* `residual_norm` `: float|[float, ...]`: norm of the residual at convergence if `full_residuals` `= False`, or list of the norms of the reisduals at every iteration.

## Acknowledgements

This work was supported by the European Research Council’s Proof of Concept Grant n° [101069402].

## Conflict of interest statement

The authors declare no conflict of interest.

## References
