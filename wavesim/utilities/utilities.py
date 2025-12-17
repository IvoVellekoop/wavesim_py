import numpy as np


def normalize(x: np.ndarray, min_val: float = None, max_val: float = None, a: float = 0, b: float = 1) -> float:
    """Normalize x to the range [a, b]

    :param x: Input array
    :param min_val: Minimum value (of x)
    :param max_val: Maximum value (of x)
    :param a: Lower bound for normalization
    :param b: Upper bound for normalization
    :return: Normalized x
    """
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    normalized_x = (x - min_val) / (max_val - min_val) * (b - a) + a
    return normalized_x
