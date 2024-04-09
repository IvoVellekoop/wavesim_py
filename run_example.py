# %% import packages
import numpy as np
from helmholtzbase import HelmholtzBase  # to set up medium, propagation operators, and scaling
from anysim import run_algorithm  # to run the anysim iteration
from save_details import LogPlot  # to log and plot the results

# %% generate a refractive index map
n_size = (3000, 6000)  # Size of the simulation domain
# Random refractive index distribution
np.random.seed(0)
n = (np.random.normal(1.3, 0.1, n_size) + 1j * np.random.normal(0.05, 0.02, n_size)).astype(np.complex64)

# %% set up source, with size same as n, and some amplitude [here, a point source at the center of the domain]
source = np.zeros_like(n)  # Source term
source[tuple(i // 2 for i in n_size)] = 1.  # Source term at the center of the domain

# %% set up scaling, and medium, propagation, and if required, correction (wrapping and transfer) operators
base = HelmholtzBase(n, source, n_domains=(1, 2))

# %% run the algorithm
u, state = run_algorithm(base)  # Field u and state object with information about the run

# %% log, plot, and save the results
LogPlot(base, state, u).log_and_plot(save=False)
