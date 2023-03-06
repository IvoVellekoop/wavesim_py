Below are the instructions for setting up Miniconda (a much lighter counterpart of Anaconda) to work in conda environments.

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

                conda env create -f <filename>.yml

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
## Working with conda and jupyter notebooks (.ipynb)

We need to add the conda environment to jupyter notebooks so that it can be selected in the Select Kernel option in a jupyter notebook

1. Activate the desired conda environment

        conda activate <environment name>

2. Install the ipykernel package

        conda install ipykernel

3. Add/install the environment to the ipykernel

        python -m ipykernel install --user --name=<environment name>

### Additional step for Visual Studio code

4. Install the Jupyter extension through the gui

### To open in browser (google chrome)

4. Although these packages should already be installed through the .yml file while setting up the environment, if the current working directory does not open jupyter in a browser window after entering the command:

        jupyter notebook
        
    1. Set up jupyterlab and jupyter notebook with the following commands

            conda install -c conda-forge jupyterlab
            conda install -c anaconda notebook

    2. Run the below command again, and now jupyter should open in a browser window:

            jupyter notebook

5. The environment should now be visible in the Select Kernel dropdown.


### To convert the current jupyter notebook into a presentation (plotly plots stay interactive)

1. Open the jupter notebook in a browser window and check that running all cells gives the expected output

2. In the toolbar at the top, Click **View** --> **Cell Toolbar** ---> **Slideshow**

3. Each cell in the notebook will now have a toolbar at the top with a dropdown named **Slide Type**. In the dropdown, select **Slide** for all the cells you want to include in the presentation.

4. Convert the .ipynb notebook to a .html presentation with the command(s and the options as below)

    * Default options

            jupyter nbconvert --to slides <filename>.ipynb

    * If you don't want to show the code, and just the outputs

            jupyter nbconvert --to slides --no-input <filename>.ipynb

    * In addition to above, if you don't want any transitions

            jupyter nbconvert --to slides --no-input <filename>.ipynb --SlidesExporter.reveal_transition=none

    * In addition to above, if you want a specific theme (here, serif)

            jupyter nbconvert --to slides --no-input <filename>.ipynb --SlidesExporter.reveal_transition=none --SlidesExporter.reveal_theme=serif

6. Double-click the .html presentation \<filename\>.html that should now be in the current working directory.

7. To convert the .html slides to pdf, add _?print-pdf_ in the URL in the web browser between _html_ and _#_.

## If problem with specifying fonts in matplotlib.rc 

Example of an error: "findfont: Generic family 'sans-serif' not found because none of the following families were found: Time New Roman"

1. Check if 'mscorefonts' package installed in conda (using conda list). If not,

        conda install -c conda-forge mscorefonts

2. Clear matplotlib cache. An equally important step.

        rm ~/.cache/matplotlib -rf