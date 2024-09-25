import os
import numpy as np
from __init__ import sim_3d_random, plot_validation

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_data', exist_ok=True)
    os.makedirs('paper_figures', exist_ok=True)
    filename = 'paper_data/fig5_dd_validation_'
    figname = 'paper_figures/fig5_dd_validation.pdf'
else:
    try:
        os.makedirs('examples/paper_data', exist_ok=True)
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_data/fig5_dd_validation_'
        figname = 'examples/paper_figures/fig5_dd_validation.pdf'
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'paper_code' directory.")

sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers (excluding boundaries)
full_residuals = True

# Run the simulations
sim_ref = sim_3d_random(filename, sim_size, n_domains=None, full_residuals=full_residuals)
sim = sim_3d_random(filename, sim_size, n_domains=(3, 1, 1), full_residuals=full_residuals)

# plot the field
plot_validation(figname, sim_ref, sim, plt_norm='log')
