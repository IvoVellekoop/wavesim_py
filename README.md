# What is [Wavesim](https://www.wavesim.org/)?
[Wavesim](https://www.wavesim.org/) is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method, Wavesim is based on the Modified Born Series approach. This method has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website [www.wavesim.org](https://www.wavesim.org/).

## If you use Wavesim and publish your work, please cite us:

* Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. _Journal of computational physics, 322_, 113-124. [[Link]](https://doi.org/10.1016/j.jcp.2016.06.034)
* Osnabrugge, G., Benedictus, M., & Vellekoop, I. M. (2021). Ultra-thin boundary layer for high-accuracy simulations of light propagation. _Optics express, 29_(2), 1649-1658. [[Link]](https://doi.org/10.1364/OE.412833)

# Installation

Wavesim requires [Python 3.12](https://www.python.org/downloads/release/python-3120/) and uses [PyTorch](https://pytorch.org/) for GPU acceleration.

We recommend using [Miniconda](https://docs.anaconda.com/miniconda/) (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in [environment.yml](environment.yml)) within a conda environment. If you prefer to create a virtual environment without using Miniconda/Anaconda, you can use [requirements.txt](requirements.txt) for dependencies. The steps that follow are for a Miniconda installation.

1. **Download Miniconda**, choosing the [Python 3.12 installer](https://docs.anaconda.com/miniconda/miniconda-other-installer-links/) for your operating system (Windows/macOS/Linux).

2. **Install Miniconda**, following the [installation instructions](https://docs.anaconda.com/miniconda/miniconda-install/) for your OS. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. (If you cannot immediately activate conda, close and re-open your terminal window to make the changes take effect).

3. **Test your installation**. Open Anaconda Prompt and run the below command. Alternatively, open an editor like [Visual Studio Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/), select the Python interpreter in the `miniconda3/` directory with the label `('base')`, and run the command:

    ``` 
    conda list
    ``` 

   A list of installed packages appears if it has been installed correctly.

4. **Set up a conda environment**. Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don't work. Create a new environment using the [environment.yml](environment.yml) and activate.
    ```
    conda env create -f environment.yml
    conda activate wavesim
    ```

    The [Miniconda environment management guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more details if you need them.

# Running the code

Once the virtual environment is setup with all the required packages, you are ready to run the code. You can go through any of the scripts in the [examples directory](examples) for the basic steps needed to run a simulation. The directory contains 2 examples each of 1D, 2D, and 3D problems. 

You can run the code with just three inputs:
* `permittivity`, i.e. refractive index distribution squared (a 3-dimensional array on a regular grid),
* `periodic`, a tuple of three booleans to indicate whether the domain is periodic in each dimension [True] or not [False], and
* `source` (same size as permittivity)

Below a simple example, [helmholtz_1d_analytical.py](examples/helmholtz_1d_analytical.py), is shown to explain these and the other optional inputs.

## Simple example (1D, homogeneous medium)

    ```
    import numpy as np
    from wavesim.multidomain import MultiDomain     # to set up medium, propagation operators, and scaling
    from wavesim_iteration import run_algorithm     # to run the wavesim iteration
    from utilities import preprocess                # to pad refractive_index and square to give permittivity

    permittivity = np.ones((256, 1, 1))
    periodic = (True, True, True)
    source = np.zeros_like(permittivity)
    source[0] = 1.                                  # Amplitude 1 at location [0]

    domain = MultiDomain(permittivity, periodic)    # to set up the domain operators
    u = run_algorithm(domain, source)               # Field u
    ```

All other parameters have defaults. Details about permittivity, source, and the other parameters are given below (with the default values, if defined).

### MultiDomain()

`permittivity`: 3-dimensional array with refractive index-squared distribution in x, y, and z direction. To set up a 1 or 2-dimensional problem, leave the other dimension(s) as 1.

`periodic`: indicates for each dimension whether the simulation is periodic [True] or not [False]. For periodic dimensions, i.e., `periodic = [True, True, True]`, the field is wrapped around the domain.

`pixel_size: float = 0.25`: points per wavelength.

`wavelength: float = None`: wavelength: wavelength in micrometer (um).

`n_domains: tuple[int, int, int] = (1, 1, 1)`: number of domains to split the simulation into. If the domain size is not divisible by n_domains, the last domain will be slightly smaller than the other ones. If `(1, 1, 1)`, indicates no domain decomposition.

`n_boundary: int = 8`: number of points used in the wrapping and domain transfer correction. Applicable when `periodic` is False in a dimension, or `n_domains` > 1 in a dimension.

`device: str = None`: 
*  `'cpu'` to use the cpu, 
* `'cuda:x'` to use a specific cuda device
* `'cuda'` or a list of strings, e.g., `['cuda:0', 'cuda:1']`, to distribute the simulation over the available/given cuda devices in a round-robin fashion
* `None`, which is equivalent to `'cuda'` if cuda devices are available, and `'cpu'` if they are not.

`debug: bool = False`: set to `True` for testing to return inverse_propagator_kernel as output.

### run_algorithm()

`domain`: the domain object created by MultiDomain

`source`: source term, a 3-dimensional array, with the same size as permittivity. Set up amplitude(s) at the desired location(s), following the same principle as permittivity for 1, 2, or 3-dimensional problems.

`alpha: float = 0.75`: relaxation parameter for the Richardson iteration

`max_iterations: int = 1000`: maximum number of iterations

`threshold: float = 1.e-6`: threshold for the residual norm for stopping the iteration
