import torch
import time

## Tests for the fastest way to calculate the Euclidean norm of a complex vector
# It seems that on the CPU, 'norm' is consistently slower than 'vdot', but gives a slightly different, perhaps more accurate value.
# On the GPU, 'norm' is only very slightly faster (1%) than 'vdot' and gives the same result as vdot, indicating that the same
# computations are used (perhaps with one cached memory load less in the 'norm' case)

size = 100000000  # Οne hundred million elements


def norm_using_vdot(vector):
    squared_magnitude = torch.vdot(vector, vector)
    return squared_magnitude.sqrt().real


def norm_using_matmul(vector):
    x = torch.unsqueeze(vector, 0)
    y = torch.unsqueeze(vector, 0)
    squared_magnitude = torch.matmul(x, y.H).item()
    return squared_magnitude.sqrt().real


def norm_using_linalg_norm(vector):
    return torch.linalg.norm(vector)


def timeit(func, name, complex_vector):
    start_time = time.time()
    for x in range(100):
        value = func(complex_vector)
    print(f"Euclidean norm using {name}: {value}, Time taken: {time.time() - start_time} seconds")


for device in ['cpu', 'cuda']:
    complex_vector = torch.randn(size, dtype=torch.float32, device=device) + 1j * torch.randn(size,
                                                                                              dtype=torch.float32,
                                                                                              device=device)
    print(device)
    for repeat in range(3):
        print(f"Repeat {repeat + 1}")
        # timeit(norm_using_matmul, 'torch.matmul', complex_vector) #  always much slower
        timeit(norm_using_linalg_norm, 'torch.linalg.norm', complex_vector)
        timeit(norm_using_vdot, 'torch.vdot', complex_vector)
