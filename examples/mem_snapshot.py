"""
PyTorch memory snapshot example
===============================
This script captures a memory snapshot of the GPU memory usage during the simulation.
"""

import os
import sys
import torch
import platform
import numpy as np
from time import time
from paper_code.__init__ import random_refractive_index, construct_source
sys.path.append(".")
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.multidomain import MultiDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
if os.path.basename(os.getcwd()) == 'examples':
    os.chdir('..')
os.makedirs("logs", exist_ok=True)


def is_supported_platform():
    return platform.system().lower() == "linux" and sys.maxsize > 2**32


if is_supported_platform():
    torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100000, 
                                             trace_alloc_record_context=True)
else:
    print(f"Pytorch emory snapshot functionality is not supported on {platform.system()} (non-linux non-x86_64 platforms).")
    # On Windows, gives "RuntimeError: record_context_cpp is not supported on non-linux non-x86_64 platforms"

# generate a refractive index map
sim_size = 100 * np.array([2, 1, 1])  # Simulation size in micrometers
wavelength = 1.
pixel_size = 0.25
boundary_widths = 20
n_dims = len(sim_size.squeeze())

# Size of the simulation domain in pixels
n_size = sim_size * wavelength / pixel_size
n_size = n_size - 2 * boundary_widths  # Subtract the boundary widths
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

n = random_refractive_index(n_size)
print(f"Size of n: {n_size}")
print(f"Size of n in GB: {n.nbytes / (1024**3):.2f}")
assert n.imag.min() >= 0, 'Imaginary part of n is negative'

# return permittivity (n²) with boundaries, and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n**2, boundary_widths)  # permittivity is n², but uses the same variable n
assert n.imag.min() >= 0, 'Imaginary part of n² is negative'

source = construct_source(n_size, boundary_array)

n_domains = (2, 1, 1)  # number of domains in each direction
periodic = (False, True, True)  # True for 1 domain in that direction, False otherwise
domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
                     n_domains=n_domains)

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = run_algorithm(domain, source, max_iterations=5)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')

if is_supported_platform():
    try:
        torch.cuda.memory._dump_snapshot(f"logs/mem_snapshot_cluster.pickle")
        # To view memory snapshot, got this link in a browser window: https://pytorch.org/memory_viz
        # Then drag and drop the file "mem_snapshot.pickle" into the browser window.
        # From the dropdown menus in the top left corner, open the second one and select "Allocator State History".
    except Exception as e:
        # logger.error(f"Failed to capture memory snapshot {e}")
        print(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history
    torch.cuda.memory._record_memory_history(enabled=None)


# %% Postprocessing

file_name = 'logs/size'
for i in range(n_dims):
    file_name += f'{n_size[i]}_'
file_name += f'bw{boundary_widths}_domains'
for i in range(n_dims):
    file_name += f'{n_domains[i]}'

output = (f'Size {n_size}; Boundaries {boundary_widths}; Domains {n_domains}; '
          + f'Time {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e} \n')
with open('logs/output.txt', 'a') as file:
    file.write(output)
