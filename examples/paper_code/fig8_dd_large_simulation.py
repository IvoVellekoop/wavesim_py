import os
import numpy as np
from __init__ import sim_3d_random, plot_validation

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    filename = 'paper_data/fig8_dd_large_simulation_'
    figname = 'paper_figures/fig8_dd_large_simulation.pdf'
else:
    try:
        filename = 'examples/paper_data/fig8_dd_large_simulation_'
        figname = 'examples/paper_figures/fig8_dd_large_simulation.pdf'
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'paper_code' directory.")

sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers (excluding boundaries)
full_residuals = True

# Run the simulations
sim_cpu = sim_3d_random(filename, sim_size, n_domains=None, full_residuals=full_residuals, 
                        device='cpu')
sim_gpu = sim_3d_random(filename, sim_size, n_domains=(2, 1, 1), full_residuals=full_residuals)

# plot the field
plot_validation(figname, sim_cpu, sim_gpu, plt_norm='log')