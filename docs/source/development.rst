.. _section-development:

WaveSim Development
==============================================

Running the tests and examples
--------------------------------------------------
To download the source code, including tests and examples, clone the repository from GitHub :cite:`wavesim_py` and use any PEP 621-compatible package manater to create a virtual environment and install all dependencies. The examples below also install the package in editable mode, so that the tests and examples use the current version of the code rather than a version installed from PyPi.

For `uv`

.. code-block:: shell

    git clone https://github.com/IvoVellekoop/wavesim_py
    cd wavesim_py
    uv venv
    uv sync --all-extras
    uv run pytest tests

uv installs the `wavesim` package in editable mode. This means that changes in the source code of the package are automatically seen by the tests and the examples. However, in this mode the package metadata is not updated automatically. If you change `pyproject.toml`, make sure to run `uv build` to update the metadata.
In PyCharm, open the project. You should have wavesim_py as root, and `examples, tests, src`, etc. as subfolders.
Select `Add new interpreter...` --> `Add local interpreter...`--> Generate New, `Type: uv`.


The examples are located in the ``examples`` directory. Note that a lot of functionality is also demonstrated in the automatic tests located in the ``tests`` directory. As an alternative to downloading the source code, the samples can also be copied directly from the example gallery on the documentation website :cite:`readthedocs_WaveSim`.

Building the documentation
--------------------------------------------------

.. only:: html or markdown

    The html, and pdf versions of the documentation, as well as the ``README.md`` file in the root directory of the repository, are automatically generated from the docstrings in the source code and reStructuredText source files in the repository.

.. only:: latex

    The html version of the documentation, as well as the ``README.md`` file in the root directory of the repository, and the pdf document you are currently reading are automatically generated from the docstrings in the source code and reStructuredText source files in the repository.

Note that for building the pdf version of the documentation, you need to have ``xelatex`` installed, which comes with the MiKTeX distribution of LaTeX :cite:`MiKTeX`. Then, run the following commands to build the html and pdf versions of the documentation, and to auto-generate ``README.md``.

.. code-block:: shell

    .venv\Scripts\activate
    cd docs
    make clean
    make html
    make markdown
    tex


Reporting bugs and contributing
--------------------------------------------------
Bugs can be reported through the GitHub issue tracking system. Better than reporting bugs, we encourage users to *contribute bug fixes, optimizations, and other improvements*. These contributions can be made in the form of a pull request :cite:`zandonellaMassiddaOpenScience2022`, which will be reviewed by the development team and integrated into the package when appropriate. Please contact the current development team through GitHub :cite:`wavesim_py` or the `www.wavesim.org <https://www.wavesim.org/>`_ forum to coordinate such contributions.
