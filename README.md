# WaveSim

### What is WaveSim?
[WaveSim](https://www.wavesim.org/) is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method, WaveSim is based on the Modified Born Series approach. This method has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website [www.wavesim.org](https://www.wavesim.org/).

### If you use WaveSim and publish your work, please cite us:

* Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. _Journal of computational physics, 322_, 113-124. [[Link]](https://doi.org/10.1016/j.jcp.2016.06.034)
* Osnabrugge, G., Benedictus, M., & Vellekoop, I. M. (2021). Ultra-thin boundary layer for high-accuracy simulations of light propagation. _Optics express, 29_(2), 1649-1658. [[Link]](https://doi.org/10.1364/OE.412833)

## Installation

WaveSim requires [Python 3.12](https://www.python.org/downloads/release/python-3120/) and uses [PyTorch](https://pytorch.org/) for GPU acceleration.

We recommend using [Miniconda](https://docs.anaconda.com/miniconda/) (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in [environment.yml](environment.yml)) within a conda environment. If you prefer to create a virtual environment without using Miniconda/Anaconda, you can use [requirements.txt](requirements.txt) for dependencies. The steps that follow are for a Miniconda installation.

1. **Download Miniconda**, choosing the [Python 3.12 installer](https://docs.anaconda.com/miniconda/miniconda-other-installer-links/) for your operating system (Windows/macOS/Linux).

2. **Install Miniconda**, following the [installation instructions](https://docs.anaconda.com/miniconda/miniconda-install/) for your OS. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. (If you cannot immediately activate conda, close and re-open your terminal window to make the changes take effect).

3. **Test your installation**. In the terminal, run the below command.

        conda list
   A list of installed packages appears if it has been installed correctly.

6. **Set up a conda environment**. Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don't work.

   If you see (base) before all the information before \$ on a command line, you are already in the base conda environment. If not, run

        conda activate
   Create a new environment by running

        conda env create -f environment.yml
   

    * To update the current conda environment from a .yml file:
    
            conda env update --name anysim_cluster --file environment.yml --prune
    
    * To export the current environment to a .yml file:
    
            conda env export > <filename>.yml
    
    * To install any packages within an environment, first go into the environment and then install the package:
    
            conda activate <environment name>
            conda install <package name>
    
    * If conda does not have the package, and googling it suggests installing it via pip, use this command to install it specifically within the current environment and not globally (always prefer conda over pip. Only go to pip if the package is not available through conda):
    
            python -m pip install <package name>
    
    * After updating conda, setting up a new environment, installing packages, it is a nice idea to clean up any installation packages or tarballs as they are not needed anymore:
    
            conda clean --all

The [Miniconda environment management guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more details if you need them.

---

# Running the code

test/test_examples.py contains 2 examples each of 1D, 2D, and 3D problems. The minimum 2 inputs that all require are n (refractive index distribution, a 3-dimensional array) and source (source term, of the same size as n). 

## Simple example (1D, homogeneous medium)

        import numpy as np
        from helmholtzbase import HelmholtzBase  # to set up medium, propagation operators, and scaling
        from anysim import run_algorithm  # to run the anysim iteration

        n = np.ones((256, 1, 1))        # Refractive index distribution
        source = np.zeros_like(n)       # Source term
        source[0] = 1.                  # Amplitude 1. at location [0]
        base = HelmholtzBase(n, source) # to set up medium and propagation operators, and scaling
        u, state = run_algorithm(base)  # Field u and state object with information about the run

All other parameters have defaults. Details about n, source, and the other parameters are given below, with the default values defined given in the headers:

## n = np.ones((1, 1, 1))
3-dimensional array with refractive index distribution in x, y, and z direction. To set up a 1 or 2-dimensional problem, simply fill the first 1 or 2 dimensions with values > 1 and leave the other dimension(s) as 1.

## source=np.zeros((1, 1, 1))
Source term, a 3-dimensional array, with the same size as n, and default as 0 everywhere. Set up amplitude(s) at the desired location(s), following the same principle as n for 1, 2, or 3-dimensional problems.

## wavelength = 1.
Wavelength in um (micron)

## ppw = 4
points per wavelength

## boundary_widths = (20, 20, 20)
Width of absorbing boundaries. 3-element tuple indicating boundaries in x, y, and z dimensions.

## n_domains = (1, 1, 1)
Number of subdomains to decompose the problem into. 3-element tuple indicating number of domains in x, y, and z dimensions.

## wrap_correction = None
None
    (Eliminate wrap-around effects with absorbing boundaries), OR

'L_omega'
    (Do the fast convolution over a much larger domain such that there are no wrap-around effects in the Laplacian), OR

'wrap_corr'
    (Add the wrapping correction term to the medium operator to correct for the wrap-around effects, allowing for smaller absorbing boundaries only to tackle reflections)

wrap_correction defaults to 'wrap_corr' when n_domains > 1.

## omega = 10
Compute the fft over omega times the domain size. The fft is used for implementing the Laplacian in the wrap_correction='L_omega' case, or the wrapping corrections in the wrap_correction='wrap_corr' case, or in the communication between subdomains when n_domains > 1.

## n_correction = 8
Number of points used in the wrapping correction in the wrap_correction='wrap_corr' case, or in the communication between subdomains when n_domains > 1.

## max_iterations = 10000
[int] Maximum number of iterations

## setup_operators = True
Boolean for whether to set up Medium (+corrections) and Propagator operators, and scaling

---

# If problem with specifying fonts in matplotlib.rc 

Example of an error: "findfont: Generic family 'sans-serif' not found because none of the following families were found: Time New Roman"

1. Check if 'mscorefonts' package installed in conda (using conda list). If not,

        conda install -c conda-forge mscorefonts

2. Clear matplotlib cache. An equally important step.

        rm ~/.cache/matplotlib -rf

---

# If problem with tex in matplotlib

        sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

        python -m pip install latex

# For animations, ffmpeg package needed (below command for Linux)

        sudo apt-get install ffmpeg
