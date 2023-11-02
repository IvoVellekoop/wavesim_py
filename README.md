## Running the code

To run with the default options/flags:

        python anysim_combined.py

The flags that can be specified, along with the options for each of them are given below:

topic = 'Helmholtz'

""" Helmholtz (the only one for now) or Maxwell """

# test = 'FreeSpace'
""" 
'FreeSpace'
    (Simulates free-space propagation and compares the result to the analytical solution), OR
'1D'
    (Simulates 1-D propagation of light through a slab of glass), OR
'2D'
    (Simulates propagation of light in a 2-D structure made of iron (uses the scalar wave equation)), OR 
'2D_low_contrast'
    (Simulates propagation of light in a 2-D structure made of fat and water (uses the scalar wave equation)) 
"""

# smallest_circle_problem = False
""" True (V0 as in AnySim) or False (V0 as in WaveSim) """

# absorbing_boundaries = False
""" True (add absorbing boundaries) or False (don't add) """

# wrap_correction = 'None'
""" 
'None'
    (Use the usual Laplacian and eliminate wrap-around effects with absorbing boundaries), OR
'L_omega'
    (Do the fast convolution over a much larger domain to eliminate wrap-around effects without absorbing boundaries), OR
'L_corr'
    (Add the wrap-aroound correction term to V to correct for the wrap-around effects without absorbing boundaries)
"""


---

Below are the instructions for setting up Miniconda (a much lighter counterpart of Anaconda) to work in conda environments, and the conda environment corresponding to this project.

---
## Installing Miniconda on a Linux system (or Windows Subsystem for Linux)

1. Download the appropriate (i.e. Linux/Windows/MacOS) installer from https://docs.conda.io/en/latest/miniconda.html

Assuming Linux for the step ahead. Instructions also available at https://conda.io/projects/conda/en/stable/user-guide/install/linux.html

2. In the directory where the installer was downloaded, in the terminal window, run:

        bash Miniconda3-latest-Linux-x86_64.sh

3. Follow the prompts on the installer screens. If you are unsure about any setting, accept the defaults. You can change them later.

4. To make the changes take effect, close and then re-open your terminal window.

5. Test your installation. In your terminal window or Anaconda Prompt, run the command

        conda list

A list of installed packages appears if it has been installed correctly.

---
## Setting up a conda environment

Instructions also available at https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Avoid using the base environment altogether. It is a good backup environment to fall back on if and when the other environments are corrupted/don't work.

1. If you see (base) before all the information before \$ on a command line, you are already in the base conda environment. If not, run:

        conda activate

2. Update conda:

                conda update conda
                conda update --all

3. New conda environment from:

    * scratch, just with commands
        
                conda create --name <environment name>

        OR

    * a .yml file in the current directory

                conda env create -f setting_up/anysim_cluster.yml

4. To update the current conda environment from a .yml file:

        conda env update --name anysim_cluster --file setting_up/anysim_cluster.yml --prune

4. To export current environment to a .yml file:

        conda env export > <filename>.yml

5. To install any packages within an environment, first go into the environment and then install the package:

        conda activate <environment name>
        conda install <package name>

6. If conda does not have the package, and googling it suggests installing it via pip, use this command to install it specifically within the current environment and not globally (always prefer conda over pip. Only go to pip if the package is not available through conda):

        python -m pip install <package name>

7. After updating conda, setting up a new environment, installing packages, it is a nice idea to clean-up any installation packages or tarballs as they are not needed anymore:

        conda clean --all

---

## If problem with specifying fonts in matplotlib.rc 

Example of an error: "findfont: Generic family 'sans-serif' not found because none of the following families were found: Time New Roman"

1. Check if 'mscorefonts' package installed in conda (using conda list). If not,

        conda install -c conda-forge mscorefonts

2. Clear matplotlib cache. An equally important step.

        rm ~/.cache/matplotlib -rf