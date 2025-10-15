import os
import numpy as np
from __init__ import sim_3d_random, plot_validation

n_domains = (2, 1, 1)

if os.path.basename(os.getcwd()) == 'paper_code':
    os.chdir('..')
    os.makedirs('paper_data', exist_ok=True)
    os.makedirs('paper_figures', exist_ok=True)
    filename = 'paper_data/fig9_dd_large_simulation_'
    figname = f'paper_figures/fig9_dd_large_simulation_domains'
    for i in range(3):
        figname += f'{n_domains[i]}'
    figname += '.pdf'
else:
    try:
        os.makedirs('examples/paper_data', exist_ok=True)
        os.makedirs('examples/paper_figures', exist_ok=True)
        filename = 'examples/paper_data/fig9_dd_large_simulation_'
        figname = 'examples/paper_figures/fig9_dd_large_simulation_domains'
        for i in range(3):
            figname += f'{n_domains[i]}'
        figname += '.pdf'
    except FileNotFoundError:
        print("Directory not found. Please run the script from the 'paper_code' directory.")

sim_size = 320 * np.array([1, 1, 1])  # Simulation size in micrometers (excluding boundaries)
full_residuals = True

# Run the simulations
sim_gpu = sim_3d_random(filename, sim_size, n_domains=n_domains, r=24, clearance=24, full_residuals=full_residuals)
sim_cpu = sim_3d_random(filename, sim_size, n_domains=None, n_boundary=0, r=24, clearance=24, full_residuals=full_residuals, device='cpu')

# plot the field
plot_validation(figname, sim_cpu, sim_gpu, plt_norm='log', inset=True)
print('Done.')
