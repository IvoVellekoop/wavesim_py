.. _root_label:

Wavesim
=====================================

..
    NOTE: README.MD IS AUTO-GENERATED FROM DOCS/SOURCE/README.RST. DO NOT EDIT README.MD DIRECTLY.

.. only:: html

    .. image:: https://readthedocs.org/projects/wavesim/badge/?version=latest
       :target: https://wavesim.readthedocs.io/en/latest/?badge=latest
       :alt: Documentation Status


What is Wavesim?
----------------------------------------------

Wavesim is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method :cite:`yee1966numerical, taflove1995computational, oskooi2010meep, nabavi2007new`, Wavesim is based on the Modified Born Series approach, which has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD  :cite:`osnabrugge2016convergent, vettenburg2023universal`.

This package :cite:`wavesim_py` is a Python implementation of the Modified Born Series (MBS) approach for solving the Helmholtz equation in arbitrarily large media through domain decomposition :cite:`mache2024domain`. With this new framework, we simulated a complex 3D structure of a remarkable :math:`315\times 315\times 315` wavelengths :math:`\left( 3.1\cdot 10^7 \right)` in size in just :math:`379` seconds by solving over two GPUs. This represents a factor of :math:`1.93` increase over the largest possible simulation on a single GPU without domain decomposition. 

When using Wavesim in your work, please cite :cite:`mache2024domain, osnabrugge2016convergent`, and :cite:`wavesim_py` for the code. Examples and documentation for this project are available at `Read the Docs <https://wavesim.readthedocs.io/en/latest/>`_ :cite:`wavesim_documentation`. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website `www.wavesim.org <https://www.wavesim.org/>`_.

Installation
----------------------------------------------

Wavesim requires `Python 3.11.0 and above <https://www.python.org/downloads/>`_ and uses `PyTorch <https://pytorch.org/>`_ for GPU acceleration.

We recommend using `Miniconda <https://docs.anaconda.com/miniconda/>`_ (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in `environment.yml <https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml>`_) within a conda environment. If you prefer to create a virtual environment without using Miniconda/Anaconda, you can use `requirements.txt <https://github.com/IvoVellekoop/wavesim_py/blob/main/requirements.txt>`_ for dependencies. The steps that follow are for a Miniconda installation.

1. **Download Miniconda**, choosing the appropriate `Python installer <https://docs.anaconda.com/miniconda/>`_ for your operating system (Windows/macOS/Linux).

2. **Install Miniconda**, following the `installation instructions <https://docs.anaconda.com/miniconda/miniconda-install/>`_ for your OS. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. (If you cannot immediately activate conda, close and re-open your terminal window to make the changes take effect).

3. **Test your installation**. Open Anaconda Prompt and run the below command. Alternatively, open an editor like `Visual Studio Code <https://code.visualstudio.com/>`_ or `PyCharm <https://www.jetbrains.com/pycharm/>`_, select the Python interpreter in the ``miniconda3/`` directory with the label ``('base')``, and run the command::

    conda list

  A list of installed packages appears if it has been installed correctly.

4. **Set up a conda environment**. Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don't work. Create a new environment using `environment.yml <https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml>`_ and activate.::

    conda env create -f environment.yml
    conda activate wavesim

  The `Miniconda environment management guide <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ has more details if you need them.

Running the code
----------------

Once the virtual environment is set up with all the required packages, you are ready to run the code. You can go through any of the scripts in the ``examples`` `directory <https://github.com/IvoVellekoop/wavesim_py/tree/main/examples>`_ for the basic steps needed to run a simulation. The directory contains two examples each of 1D, 2D, and 3D problems. 

You can run the code with just three inputs:

* :attr:`~.Domain.permittivity`, i.e. refractive index distribution squared (a 4-dimensional array on a regular grid),

* :attr:`~.Domain.periodic`, a tuple of three booleans to indicate whether the domain is periodic in each dimension [``True``] or not [``False``], and

* :attr:`~.Domain.source`, the same size as permittivity.

:numref:`helmholtz_1d_analytical` shows a simple example of a 1D problem with a homogeneous medium (`helmholtz_1d_analytical.py <https://github.com/IvoVellekoop/wavesim_py/blob/main/examples/helmholtz_1d_analytical.py>`_) to explain these and other inputs.

.. _helmholtz_1d_analytical:
.. literalinclude:: ../../examples/helmholtz_1d_analytical.py
    :language: python
    :caption: ``helmholtz_1d_analytical.py``. A simple example of a 1D problem with a homogeneous medium.

Apart from the inputs :attr:`~.Domain.permittivity`, :attr:`~.Domain.periodic`, and :attr:`~.Domain.source`, all other parameters have defaults. Details about these are given below (with the default values, if defined).

Parameters in the :class:`~.domain.Domain` class: :class:`~.helmholtzdomain.HelmholtzDomain` or :class:`~.multidomain.MultiDomain`

* :attr:`~.Domain.permittivity`: 4-dimensional array with refractive index-squared distribution in x, y, and z direction, and a polarization dimension (unused in Helmholtz case). To set up a 1 or 2-dimensional problem, leave the other dimension(s) as 1.

* :attr:`~.Domain.periodic`: indicates for each dimension whether the simulation is periodic (``True``) or not (``False``). For periodic dimensions, i.e., :attr:`~.Domain.periodic` ``= [True, True, True]``, the field is wrapped around the domain.

* :attr:`~.pixel_size` ``:float = 0.25``: points per wavelength.

* :attr:`~.wavelength` ``:float = None``: wavelength: wavelength in micrometer (um). If not given, i.e. ``None``, it is calculated as ``1/pixel_size = 4 um``.

* :attr:`~.n_domains` ``: tuple[int, int, int] = (1, 1, 1)``: number of domains to split the simulation into. If the domain size is not divisible by n_domains, the last domain will be slightly smaller than the other ones. If ``(1, 1, 1)``, indicates no domain decomposition.

* :attr:`~.n_boundary` ``: int = 8``: number of points used in the wrapping and domain transfer correction. Applicable when :attr:`~.Domain.periodic` is False in a dimension, or ``n_domains > 1`` in a dimension.

* :attr:`~.device` ``: str = None``: 

    *  ``'cpu'`` to use the cpu, 

    * ``'cuda:x'`` to use a specific cuda device

    * ``'cuda'`` or a list of strings, e.g., ``['cuda:0', 'cuda:1']``, to distribute the simulation over the available/given cuda devices in a round-robin fashion
    
    * ``None``, which is equivalent to ``'cuda'`` if cuda devices are available, and ``'cpu'`` if they are not.

* :attr:`~.debug` ``: bool = False``: set to ``True`` for testing to return :attr:`~.inverse_propagator_kernel` as output.

Parameters in the :func:`run_algorithm` function

* :attr:`~.domain`: the domain object created by HelmholtzDomain() or MultiDomain()

* :attr:`~.Domain.source`: source term, a 4-dimensional array, with the same size as permittivity. Set up amplitude(s) at the desired location(s), following the same principle as permittivity for 1, 2, or 3-dimensional problems.

* :attr:`~.alpha` ``: float = 0.75``: relaxation parameter for the Richardson iteration

* :attr:`~.max_iterations` ``: int = 1000``: maximum number of iterations

* :attr:`~.threshold` ``: float = 1.e-6``: threshold for the residual norm for stopping the iteration

%endmatter%
