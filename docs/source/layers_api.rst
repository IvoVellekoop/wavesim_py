.. _section-layers_api:

API Layers
==========

The WaveSim package is organized in several layers, each of which is responsible for a specific aspect of the simulation. The layers are designed to be modular and extensible, so that users can easily add new functionality or modify existing functionality. The layers are organized in a hierarchical structure, with each layer building on the functionality provided by the layers below it. The layers are as follows:

1. **The engine layer.** This layer is responsible for the low-level computational operations, such as addition, multiplication, Fourier transforms, etc. The key class is the ``Array``, which is an abstract representation of a multi-dimensional array of numbers. There are different backends that implement the functionality of the ``Array`` class, currently CuPy, and Numpy/Numba. ``Array`` provides a level of abstraction that allows working with blocked arrays, sparse arrays, etc. without changing the code that uses the arrays. Future plans include adding support for distributed arrays, arrays that are evaluated lazily from geometrical data, and arrays that implement indexed or compressed data storage. All data in WaveSim is passed around as ``Array`` objects.

  - Dependency on other layers: None.
  - Exposed interface: ``Array`` class, and the functions that operate on ``Array`` objects. These functions are automatically forwarded to their implementations using a multi-dispatch mechanism.
  - Usage in client code: all data is passed to and from WaveSim in the form of an ``Arrays`` from data. Client code will need to construct ``Arrays`` from data, and extract data from ``Arrays`` when needed. Except when constructing arrays, the client code should not need to know any details of the backend implementation.

2. **The physics layer.** This layer contains the implementations of the physical problems to be solved: ``Helmholtz``, and later ``MaxwellSolver``, etc.

  - Dependency on other layers: Engine
  - Exposed interface: ``Helmholtz`` class is intended to be used in client code. It exposes the ``medium`` ``propagator`` and related functions, which are used in the algorithm layer.
  - Usage in client code: The client code will create a ``Helmholtz`` object, pass the medium properties as an ``Array``, as well as options (wavelength, grid spacing, boundary types, etc. see documentation fo the ``Helmholtz``).

3. **The algorithm layer.** This layer contains the implementations of the algorithms used to solve the physical problems. It contains functions for the preconditioned Richardson iteration, as well as lower-level functions for evaluating just the forward operator, the preconditioned operator, the preconditioner itself, etc. Client code will typically only call the preconditioned Richardson iteration function, but the lower-level functions are exposed for advanced users who want to experiment with different algorithms.

4. **The client code.** (*not* part of the WaveSim package) This layer is the top layer, where the user interacts with the package. It contains the code that sets up the physical problem to be solved, calls the iterative solver, and processes, stores and/or displays the results. The client code may be a script, a Jupyter notebook, a GUI application, a web server, etc.

  - Dependency on other layers: Engine, Physics, Algorithm.
    - The client code is responsible for selecting a backend, and constructing an ``Array`` that represents the permittivity distribution.
    - The client code is responsible for constructing a ``Helmholtz`` object, and passing the permittivity distribution and simulation options (grid spacing, wavelength) to it.
    - Finally, the client code is responsible for calling the iterative solver function, optionally providing a callback to report progress or visualise intermediate results.
  - Exposed interface: None

5. **Utilities.** (does not exist yet). A subpackage containing functions that simplify working with the data: loading and saving `Array` objects, converting 3-D models to `Array` objects, etc.

