import os
import sys
import torch
import platform
import numpy as np
from time import time
from wavesim.helmholtzdomain import HelmholtzDomain
from wavesim.iteration import run_algorithm
from wavesim.utilities import preprocess

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def is_supported_platform():
    return platform.system().lower() == "linux" and sys.maxsize > 2**32


if is_supported_platform():
    torch.cuda.memory._record_memory_history()
else:
    print("This functionality is not supported on your platform.")
    # On Windows, gives "RuntimeError: record_context_cpp is not support on non-linux non-x86_64 platforms"

# generate a refractive index map
sim_size = 50 * np.array([1, 1, 1])  # Simulation size in micrometers
wavelength = 1.
pixel_size = 0.25
boundary_widths = 20
n_dims = len(sim_size.squeeze())

# Size of the simulation domain
# Size of the simulation domain in pixels
n_size = sim_size * wavelength / pixel_size
n_size = n_size - 2 * boundary_widths  # Subtract the boundary widths
n_size = tuple(n_size.astype(int))  # Convert to integer for indexing

torch.manual_seed(0)  # Set the random seed for reproducibility
n = (torch.normal(mean=1.3, std=0.1, size=n_size, dtype=torch.float32)
     + 1j * abs(torch.normal(mean=0.05, std=0.02, size=n_size, dtype=torch.float32))).numpy()
print(f"Size of n: {n_size}")
print(f"Size of n in GB: {n.nbytes / (1024**3):.2f}")
assert n.imag.min() >= 0, 'Imaginary part of n is negative'

# add boundary conditions and return permittivity (n²) and boundary_widths in format (ax0, ax1, ax2)
n, boundary_array = preprocess(n, boundary_widths)
assert n.imag.min() >= 0, 'Imaginary part of n² is negative'

# set up source, with size same as n + 2*boundary_widths, and a point source at the center of the domain
# Location: center of the domain
indices = torch.tensor([[v // 2 + boundary_array[i]
                       for i, v in enumerate(n_size)]]).T
values = torch.tensor([1.0])  # Amplitude: 1
n_ext = tuple(np.array(n_size) + 2*boundary_array)
source = torch.sparse_coo_tensor(indices, values, n_ext, dtype=torch.complex64)

# 1-domain, periodic boundaries (without wrapping correction)
periodic = (True, True, True)  # periodic boundaries, wrapped field.
n_domains = (1, 1, 1)  # number of domains in each direction
domain = HelmholtzDomain(permittivity=n, periodic=periodic,
                         wavelength=wavelength, pixel_size=pixel_size)

# # OR. Uncomment to test domain decomposition
# n_domains = np.array([2, 1, 1])  # number of domains in each direction
# periodic = np.where(n_domains == 1, True, False)  # True for 1 domain in that direction, False otherwise
# n_domains = tuple(n_domains)
# periodic = tuple(periodic)
# domain = MultiDomain(permittivity=n, periodic=periodic, wavelength=wavelength, pixel_size=pixel_size,
#                      n_domains=n_domains)


start = time()
# Field u and state object with information about the run
u, iterations, residual_norm = run_algorithm(domain, source, max_iterations=5)
end = time() - start
print(f'\nTime {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e}')

if is_supported_platform():
    try:
        torch.cuda.memory._dump_snapshot(f"mem_snapshot.pickle")
        # To view memory snapshot, got this link in a browser window: https://pytorch.org/memory_viz
        # Then drag and drop the file "mem_snapshot.pickle" into the browser window.
        # From the dropdown menus in the top left corner, open the second one and select "Allocator State History".
    except Exception as e:
        # logger.error(f"Failed to capture memory snapshot {e}")
        print(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history
    torch.cuda.memory._record_memory_history(enabled=None)


# %% Postprocessing

file_name = './logs/size'
for i in range(n_dims):
    file_name += f'{n_size[i]}_'
file_name += f'bw{boundary_widths}_domains'
for i in range(n_dims):
    file_name += f'{n_domains[i]}'

output = (f'Size {n_size}; Boundaries {boundary_widths}; Domains {n_domains}; '
          + f'Time {end:2.2f} s; Iterations {iterations}; Residual norm {residual_norm:.3e} \n')
if not os.path.exists('../logs'):
    os.makedirs('../logs')
with open('../logs/output.txt', 'a') as file:
    file.write(output)
