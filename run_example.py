# %% import packages
import torch
from torch import tensor
import numpy as np
from wavesim.multidomain import MultiDomain  # to set up medium, propagation operators, and scaling
from anysim import run_algorithm  # to run the anysim iteration
from save_details import LogPlot  # to log and plot the results

# generate a refractive index map
n_size = (3000, 6000, 1)  # Size of the simulation domain
np.random.seed(0)
n = np.random.normal(1.3, 0.1, n_size) + 1j * np.maximum(np.random.normal(0.05, 0.02, n_size), 0.0)

# set up source, with size same as n, and a point source at the center of the domain
center_index = tensor([[n_size[0] // 2], [n_size[1] // 2], [0]])
source = torch.sparse_coo_tensor(center_index, tensor([1]), n_size, dtype=torch.complex64)

# other parameters
periodic = (True, True, True)
pixel_size = 0.25

# set up scaling, and medium, propagation, and if required, correction (wrapping and transfer) operators
base = MultiDomain(n, n_domains=(1, 2, 1), pixel_size=pixel_size, periodic=periodic)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True,
#             record_shapes=True) as prof:
#    with record_function("RunAlg"):

#  run the algorithm
u = run_algorithm(base, source)  # Field u and state object with information about the run

# output = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30)
# prof.export_stacks('profiler.stacks', 'self_cuda_time_total')
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
# with open('./output.txt', 'w') as file:
#     file.write(output)
# print(output)

# %% log, plot, and save the results
LogPlot(base, state, u).log_and_plot(save=False)
