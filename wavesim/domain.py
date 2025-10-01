from abc import ABCMeta, abstractmethod
from typing import Iterable

import numpy as np

from wavesim.engine.array import Array, Factory, array_like
from wavesim.engine.index_utils import shape_like


class Domain(metaclass=ABCMeta):
    """Base class for all simulation domains

    This base class defines the interface for operations that are common for all simulation types,
    and for single or multi domain simulations.

    Domain provides an interface for algorithms to use, without knowing anything about the type of simulation
     (e.g. Helmholtz or Maxwell), the domain size, decomposition, floating point accuracy, etc.
    """

    def __init__(
        self,
        *,
        pixel_size: float,
        shape: shape_like,
        allocator: Factory,
        periodic: Iterable[bool],
        wavelength: float,
    ):
        self.pixel_size = float(pixel_size)
        self.shape = tuple(shape)
        self.allocator = allocator
        self.periodic = tuple(periodic)
        self.wavelength = wavelength
        self.k02 = (2.0 * np.pi / self.wavelength) ** 2
        self.scale = 0.0  # to be set by child classes

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def allocate(self, data: array_like | None = None, **kwargs) -> Array:
        return self.allocator.__call__(data, copy=True, **kwargs)

    @abstractmethod
    def medium(self, x: Array, /, *, out: Array):
        """Applies the operator B=1-V.

        Args:
            x: The input array
            out: The output array
        """
        pass

    @abstractmethod
    def propagator(self, x: Array, /, *, out: Array):
        """Applies the operator (L+1)^-1 x."""
        ...

    @abstractmethod
    def inverse_propagator(self, x: Array, /, *, out: Array):
        """Applies the operator (L+1) x .

        This operation is not needed for the Wavesim algorithm, but is provided for testing purposes,
        and can be used to evaluate the residue of the solution.
        """
        ...

    def coordinates_f(self, dim):
        """Returns the Fourier-space coordinates along the specified dimension"""
        # todo: make part of Array (include pixel size property)
        return (2 * np.pi * np.fft.fftfreq(self.shape[dim], self.pixel_size)).reshape(_shapes[dim])

    def coordinates(self, dim, coordinate_type: str = "linear"):
        """Returns the real-space coordinates along the specified dimension, starting at 0"""
        x = self.pixel_size * np.arange(self.shape[dim])
        if coordinate_type == "periodic":
            x -= self.pixel_size * (self.shape[dim] // 2)
            x = np.fft.ifftshift(x)
        elif coordinate_type == "centered":
            x -= self.pixel_size * (self.shape[dim] // 2)
        elif coordinate_type == "linear":
            pass
        else:
            raise ValueError(f"Unknown type {coordinate_type}")

        return x.reshape(_shapes[dim])


_shapes = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
