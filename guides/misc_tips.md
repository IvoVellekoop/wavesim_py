### Some useful conda environment management commands

The [Miniconda environment management guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) has more details if you need them.

* To update the current conda environment from a .yml file:

    ``` 
    conda env update --name wavesim --file environment.yml --prune
    ``` 

* To export the current environment to a .yml file:

    ``` 
    conda env export > <filename>.yml
    ``` 

* To install any packages within an environment, first go into the environment and then install the package:

    ``` 
    conda activate <environment name>
    conda install <package name>
    ``` 

* If conda does not have the package, and googling it suggests installing it via pip, use this command to install it specifically within the current environment and not globally (always prefer conda over pip. Only go to pip if the package is not available through conda):

    ``` 
    python -m pip install <package name>
    ``` 

* After updating conda, setting up a new environment, installing packages, it is a nice idea to clean up any installation packages or tarballs as they are not needed anymore:

    ``` 
    conda clean --all
    ``` 

### If problem with specifying fonts in matplotlib.rc 

Example of an error: "findfont: Generic family 'sans-serif' not found because none of the following families were found: Time New Roman"

1. Check if 'mscorefonts' package installed in conda (using conda list). If not,

    ``` 
    conda install -c conda-forge mscorefonts
    ``` 

2. Clear matplotlib cache. An equally important step.

    ``` 
    rm ~/.cache/matplotlib -rf
    ``` 

### If problem with tex in matplotlib

``` 
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super

python -m pip install latex
``` 

### For animations, ffmpeg package needed (below command for Linux)

``` 
sudo apt-get install ffmpeg
``` 
