# What is [Wavesim](https://www.wavesim.org/)?
[Wavesim](https://www.wavesim.org/) is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method, Wavesim is based on the Modified Born Series approach. This method has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website [www.wavesim.org](https://www.wavesim.org/).

## If you use Wavesim and publish your work, please cite us:

* Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. _Journal of computational physics, 322_, 113-124. [[Link]](https://doi.org/10.1016/j.jcp.2016.06.034)
* Osnabrugge, G., Benedictus, M., & Vellekoop, I. M. (2021). Ultra-thin boundary layer for high-accuracy simulations of light propagation. _Optics express, 29_(2), 1649-1658. [[Link]](https://doi.org/10.1364/OE.412833)

# Installation

Wavesim requires [Python 3.11.0 and above](https://www.python.org/downloads/) and uses [PyTorch](https://pytorch.org/) for GPU acceleration.

We recommend using [Miniconda](https://docs.anaconda.com/miniconda/) (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in [environment.yml](environment.yml)) within a conda environment. If you prefer to create a virtual environment without using Miniconda/Anaconda, you can use [requirements.txt](requirements.txt) for dependencies. The steps that follow are for a Miniconda installation.

1. **Download Miniconda**, choosing the appropriate [Python installer](https://docs.anaconda.com/miniconda/) for your operating system (Windows/macOS/Linux).

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

Once the virtual environment is set up with all the required packages, you are ready to run the code. You can go through any of the scripts in the [examples directory](examples) for the basic steps needed to run a simulation. The directory contains two examples each of 1D, 2D, and 3D problems. 

You can run the code with just three inputs:
* `permittivity`, i.e. refractive index distribution squared (a 3-dimensional array on a regular grid),
* `periodic`, a tuple of three booleans to indicate whether the domain is periodic in each dimension [True] or not [False], and
* `source` (same size as permittivity)

Below a simple example, [helmholtz_1d_analytical.py](examples/helmholtz_1d_analytical.py), is shown to explain these and the other optional inputs.

## Simple example (1D, homogeneous medium)

Import basic packages and internal functions

    ```
    import numpy as np
    from torch.nn.functional import pad
    from wavesim_iteration import run_algorithm         # to run the wavesim iteration
    from utilities import preprocess                    # to pad refractive_index and square to give permittivity
    ```

To set up medium, propagation operators, and scaling and create a domain object

    ```
    from wavesim.helmholtzdomain import HelmholtzDomain # when number of domains = 1
    from wavesim.multidomain import MultiDomain         # for domain decomposition, when number of domains >= 1
    ```
Set up problem parameters
    
    ```
    wavelength = 1.                                     # wavelength in micrometer (um)
    n_size = (256, 1, 1)                                # size of simulation domain (in pixels in x, y, and z direction)
    n = np.ones(n_size, dtype=np.complex64)             # refractive index map
    boundary_widths = 50                                # padding

    # add boundary conditions and return permittivity (n²)
    n = preprocess(n, boundary_widths)[0]               # n is actually n², but uses the same variable

    # Source term
    source = torch.zeros(n_size, dtype=torch.complex64) # create source array
    source[0] = 1.                                      # Amplitude 1 at location [0]
    # Pad source. torch needs padding width as a tuple with order (z1, z2, y1, y2, x1, x2) for x, y, and z axes. 1 indicates before, and 2 indicates after.
    source = pad(source, pad = (0, 0, 
                               0, 0, 
                               boundary_widths, boundary_widths))

    periodic = (True, True, True)                       # periodic boundaries, wrapped field.
    ```

Set up the domain operators (HelmholtzDomain() or MultiDomain() depending on number of domains)

    ```
    domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength)  # This particular scenario is equivalent to setting HelmholtzDomain()
    ```

Run the wavesim iteration to get the field u

    ```
    u = run_algorithm(domain, source)[0]
    ```

All other parameters have defaults. Details about permittivity, source, and the other parameters are given below (with the default values, if defined).

### HelmholtzDomain() or MultiDomain()

`permittivity`: 3-dimensional array with refractive index-squared distribution in x, y, and z direction. To set up a 1 or 2-dimensional problem, leave the other dimension(s) as 1.

`periodic`: indicates for each dimension whether the simulation is periodic [True] or not [False]. For periodic dimensions, i.e., `periodic = [True, True, True]`, the field is wrapped around the domain.

`pixel_size: float = 0.25`: points per wavelength.

`wavelength: float = None`: wavelength: wavelength in micrometer (um). If not given, i.e. `= None`, it is calculated as `1/pixel_size = 4 um`.

`n_domains: tuple[int, int, int] = (1, 1, 1)`: number of domains to split the simulation into. If the domain size is not divisible by n_domains, the last domain will be slightly smaller than the other ones. If `(1, 1, 1)`, indicates no domain decomposition.

`n_boundary: int = 8`: number of points used in the wrapping and domain transfer correction. Applicable when `periodic` is False in a dimension, or `n_domains` > 1 in a dimension.

`device: str = None`: 
*  `'cpu'` to use the cpu, 
* `'cuda:x'` to use a specific cuda device
* `'cuda'` or a list of strings, e.g., `['cuda:0', 'cuda:1']`, to distribute the simulation over the available/given cuda devices in a round-robin fashion
* `None`, which is equivalent to `'cuda'` if cuda devices are available, and `'cpu'` if they are not.

`debug: bool = False`: set to `True` for testing to return inverse_propagator_kernel as output.

### run_algorithm()

`domain`: the domain object created by HelmholtzDomain() or MultiDomain()

`source`: source term, a 3-dimensional array, with the same size as permittivity. Set up amplitude(s) at the desired location(s), following the same principle as permittivity for 1, 2, or 3-dimensional problems.

`alpha: float = 0.75`: relaxation parameter for the Richardson iteration

`max_iterations: int = 1000`: maximum number of iterations

`threshold: float = 1.e-6`: threshold for the residual norm for stopping the iteration
