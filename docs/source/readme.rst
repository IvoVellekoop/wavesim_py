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

Wavesim is a tool to simulate the propagation of waves in complex, inhomogeneous structures. Whereas most available solvers use the popular finite difference time domain (FDTD) method :cite:`yee1966numerical, taflove1995computational, oskooi2010meep, nabavi2007new`, Wavesim is based on the modified Born series (MBS) approach, which has lower memory requirements, no numerical dispersion, and is faster as compared to FDTD  :cite:`osnabrugge2016convergent, vettenburg2023universal`.

This package :cite:`wavesim_py` is a Python implementation of the MBS approach for solving the Helmholtz equation in arbitrarily large media through domain decomposition :cite:`mache2024domain`. With this new framework, we simulated a complex 3D structure of a remarkable :math:`315\times 315\times 315` wavelengths :math:`\left( 3.1\cdot 10^7 \right)` in size in just :math:`1.4` hours by solving over two GPUs. This represents a factor of :math:`1.93` increase over the largest possible simulation on a single GPU without domain decomposition. 

When using Wavesim in your work, please cite:

    :cite:`osnabrugge2016convergent` |osnabrugge2016|_

    :cite:`mache2024domain` |mache2024|_
    
.. _osnabrugge2016: https://doi.org/10.1016/j.jcp.2016.06.034
.. |osnabrugge2016| replace:: Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. *Journal of computational physics, 322*\ , 113-124.

.. _mache2024: https://arxiv.org/abs/2410.02395
.. |mache2024| replace:: Mache, S., & Vellekoop, I. M. (2024). Domain decomposition of the modified Born series approach for large-scale wave propagation simulations. *arXiv preprint arXiv:2410.02395*.

If you use the code in your research, please cite this repository as well :cite:`wavesim_py`.

Examples and documentation for this project are available at `Read the Docs <https://wavesim.readthedocs.io/en/latest/>`_ :cite:`wavesim_documentation`. For more information (and to participate in the forum for discussions, queries, and requests), please visit our website `www.wavesim.org <https://www.wavesim.org/>`_.

Installation
----------------------------------------------

Wavesim requires `Python >=3.11.0 and <3.13.0 <https://www.python.org/downloads/>`_ and uses `PyTorch <https://pytorch.org/>`_ for GPU acceleration.

First, clone the repository and navigate to the directory::

    git clone https://github.com/IvoVellekoop/wavesim_py.git
    cd wavesim_py

Then, you can install the dependencies in a couple of ways:

:ref:`pip_installation`

:ref:`conda_installation`

:ref:`poetry_installation`

We recommend working with a virtual environment to avoid conflicts with other packages.

.. _pip_installation:

1. **Using pip**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to use pip, you can install the required packages using `requirements.txt <https://github.com/IvoVellekoop/wavesim_py/blob/main/requirements.txt>`_:

1. **Create a virtual environment and activate it** (optional but recommended)
    
   * First, `create a virtual environment <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_ using the following command::
        
        python -m venv path/to/venv

   * Then, activate the virtual environment. The command depends on your operating system and shell (`How venvs work <https://docs.python.org/3/library/venv.html#how-venvs-work>`_)::
        
        source path/to/venv/bin/activate    # for Linux/macOS
        path/to/venv/Scripts/activate.bat   # for Windows (cmd)
        path/to/venv/Scripts/Activate.ps1   # for Windows (PowerShell)

2. **Install packages**::

    pip install -r requirements.txt

.. _conda_installation:

2. **Using conda**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend using `Miniconda <https://docs.anaconda.com/miniconda/>`_ (a much lighter counterpart of Anaconda) to install Python and the required packages (contained in `environment.yml <https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml>`_) within a conda environment.

1. **Download Miniconda**, choosing the appropriate `Python installer <https://docs.anaconda.com/miniconda/>`_ for your operating system (Windows/macOS/Linux).

2. **Install Miniconda**, following the `installation instructions <https://docs.anaconda.com/miniconda/miniconda-install/>`_ for your OS. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later. (If you cannot immediately activate conda, close and re-open your terminal window to make the changes take effect).

3. **Test your installation**. Open Anaconda Prompt and run the below command. Alternatively, open an editor like `Visual Studio Code <https://code.visualstudio.com/>`_ or `PyCharm <https://www.jetbrains.com/pycharm/>`_, select the Python interpreter in the ``miniconda3/`` directory with the label ``('base')``, and run the command::

    conda list

   A list of installed packages appears if it has been installed correctly.

4. **Set up a conda environment**. Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don't work. Create a new environment using `environment.yml <https://github.com/IvoVellekoop/wavesim_py/blob/main/environment.yml>`_ and activate::

    conda env create -f environment.yml
    conda activate wavesim

   The `Miniconda environment management guide <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ has more details if you need them.

   Alternatively, you can create a conda environment with a specific Python version, and then use the `requirements.txt <https://github.com/IvoVellekoop/wavesim_py/blob/main/requirements.txt>`_ file to install the dependencies::

    conda create -n wavesim python'>=3.11.0,<3.13'
    conda activate wavesim
    pip install -r requirements.txt

.. _poetry_installation:

3. **Using Poetry**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install `Poetry <https://python-poetry.org/>`_.
2. Install dependencies by running the following command::

    poetry install

   To run tests using pytest, you can install the development dependencies as well::
    
        poetry install --with dev

3. `Activate <https://python-poetry.org/docs/managing-environments/#activating-the-environment>`_ the virtual environment created by Poetry.

Running the code
----------------

Once the virtual environment is set up with all the required packages, you are ready to run the code. You can go through any of the scripts in the ``examples`` `directory <https://github.com/IvoVellekoop/wavesim_py/tree/main/examples>`_ for the basic steps needed to run a simulation. The directory contains examples of 1D, 2D, and 3D problems. 

You can run the code with just three inputs:

* :attr:`~.Domain.permittivity`, i.e. refractive index distribution squared (a 3-dimensional array on a regular grid),

* :attr:`~.Domain.periodic`, a tuple of three booleans to indicate whether the domain is periodic in each dimension [``True``] or not [``False``], and

* :attr:`~.Domain.source`, the same size as permittivity.

:numref:`helmholtz_1d_analytical` shows a simple example of a 1D problem with a homogeneous medium (`helmholtz_1d_analytical.py <https://github.com/IvoVellekoop/wavesim_py/blob/main/examples/helmholtz_1d_analytical.py>`_) to explain these and other inputs.

.. _helmholtz_1d_analytical:
.. literalinclude:: ../../examples/helmholtz_1d_analytical.py
    :language: python
    :caption: ``helmholtz_1d_analytical.py``. A simple example of a 1D problem with a homogeneous medium.

Apart from the inputs :attr:`~.Domain.permittivity`, :attr:`~.Domain.source`, and :attr:`~.Domain.periodic`, all other parameters have defaults. Details about all parameters are given below (with the default values, if defined).

Parameters in the :class:`~.domain.Domain` class: :class:`~.helmholtzdomain.HelmholtzDomain` (for a single domain without domain decomposition) or :class:`~.multidomain.MultiDomain` (to solve a problem with domain decomposition)

* :attr:`~.Domain.permittivity`: 3-dimensional array with refractive index-squared distribution in x, y, and z direction. To set up a 1 or 2-dimensional problem, leave the other dimension(s) as 1.

* :attr:`~.Domain.periodic`: indicates for each dimension whether the simulation is periodic (``True``) or not (``False``). For periodic dimensions, i.e., :attr:`~.Domain.periodic` ``= [True, True, True]``, the field is wrapped around the domain.

* :attr:`~.pixel_size` ``:float = 0.25``: points per wavelength.

* :attr:`~.wavelength` ``:float = None``: wavelength in micrometer (um). If not given, i.e. ``None``, it is calculated as ``1/pixel_size = 4 um``.

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

* :attr:`~.Domain.source`: source term, a 3-dimensional array, with the same size as permittivity. Set up amplitude(s) at the desired location(s), following the same principle as permittivity for 1, 2, or 3-dimensional problems.

* :attr:`~.alpha` ``: float = 0.75``: relaxation parameter for the Richardson iteration

* :attr:`~.max_iterations` ``: int = 1000``: maximum number of iterations

* :attr:`~.threshold` ``: float = 1.e-6``: threshold for the residual norm for stopping the iteration

%endmatter%
