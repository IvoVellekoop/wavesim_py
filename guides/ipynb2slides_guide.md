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

5. Although these packages should already be installed through the .yml file while setting up the environment, if the current working directory does not open jupyter in a browser window after entering the command:

        jupyter notebook
        
    1. Set up jupyterlab and jupyter notebook with the following commands

            conda install -c conda-forge jupyterlab
            conda install -c anaconda notebook

    2. Run the below command again, and now jupyter should open in a browser window:

            jupyter notebook

6. The environment should now be visible in the Select Kernel dropdown.


### To convert the current jupyter notebook into a presentation (plotly plots stay interactive)

1. Open the jupyter notebook in a browser window and check that running all cells gives the expected output

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

5. Double-click the .html presentation \<filename\>.html that should now be in the current working directory.

6. To convert the .html slides to pdf, add _?print-pdf_ in the URL in the web browser between _html_ and _#_.