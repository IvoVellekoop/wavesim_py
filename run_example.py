import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from anysim import run_algorithm  # to run the anysim iteration
from utilities import preprocess
from torch.profiler import profile, record_function, ProfilerActivity  # to profile the code


# generate a refractive index map
n_size = (4000, 4000, 1)  # (3000, 6000, 1)  # (50, 120, 1)  # Size of the simulation domain
np.random.seed(0)
n = np.random.normal(1.3, 0.1, n_size) + 1j * np.maximum(np.random.normal(0.05, 0.02, n_size), 0.0)

# # set up source, with size same as n, and a point source at the center of the domain
# center_index = torch.tensor([[n_size[0] // 2], [n_size[1] // 2], [0]])
# source = torch.sparse_coo_tensor(center_index, torch.tensor([1]), n_size, dtype=torch.complex64)
# set up source, with size same as n, and some amplitude [here, a point source at the center of the domain]
source = np.zeros_like(n)  # Source term
source[n_size[0] // 2, n_size[1] // 2, 0] = 1.  # Source term at the center of the domain

boundary_widths = 50
n, source = preprocess(n, source, boundary_widths)  # add boundary conditions and return permittivity and source

# other parameters
periodic = (True, True, True)
wavelength = 1.
pixel_size = 0.25

# set up scaling, and medium, propagation, and if required, correction (wrapping and transfer) operators
domain = HelmholtzDomain(permittivity=n, wavelength=wavelength, pixel_size=pixel_size, periodic=periodic)
# domain = MultiDomain(n, n_domains=(1, 2, 1), wavelength=wavelength, pixel_size=pixel_size, periodic=periodic)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True,
#             record_shapes=True) as prof:
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:  #, profile_memory=True
    with record_function("RunAlg"):
        #  run the algorithm
        u = run_algorithm(domain, source, max_iterations=100)  # Field u and state object with information about the run

# output = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30)
# prof.export_stacks('profiler.stacks', 'self_cuda_time_total')
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
# with open('./output.txt', 'w') as file:
#     file.write(output)
# print(output)
output = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=100)  
# sort_by='self_cpu_time_total'
# sort_by='self_cuda_memory_usage'
with open('./output.txt', 'w') as file:
    file.write(output)
print(output)

# crop the field to the region of interest
u = u.squeeze()[*([slice(boundary_widths, -boundary_widths)] * 2)]

# plot the field
# plt.figure(figsize=(10, 5))
plt.imshow(np.abs(u.cpu().numpy()), cmap='hot_r', norm=LogNorm())
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()