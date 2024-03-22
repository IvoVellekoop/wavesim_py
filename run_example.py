#%% import packages
import numpy as np
from helmholtzbase import HelmholtzBase  # to set up medium, propagation operators, and scaling
from anysim import run_algorithm  # to run the anysim iteration
from save_details import LogPlot  # to log and plot the results

#%% generate a refractive index map
n_size = (60, 120)  # Size of the simulation domain
# n = np.ones(n_size, dtype=np.complex64)  # Refractive index distribution
np.random.seed(0)
n = np.random.normal(1.3, 0.1, n_size) + 1j*np.random.normal(0.05, 0.02, n_size)  # Random refractive index distribution

#%% set up source, with size same as n, and some amplitude [here, a point source at the center of the domain]
source = np.zeros_like(n)  # Source term
source[tuple(i//2 for i in n_size)] = 1.  # Source term at the center of the domain

#%% set up scaling, and medium, propagation, and if required, correction (wrapping and transfer) operators
base = HelmholtzBase(n, source)

#%% run the algorithm
u, state = run_algorithm(base)  # Field u and state object with information about the run

#%% log and plot the results
LogPlot(base, state, u).log_and_plot()
