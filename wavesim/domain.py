import torch
import numpy as np
from torch import tensor
from abc import ABCMeta, abstractmethod


class Domain(metaclass=ABCMeta):
    def __init__(self, pixel_size: float, shape, device):
        self.pixel_size = pixel_size
        self.scale = None
        self.shift = None
        self.shape = shape
        self.device = device

    @abstractmethod
    def add_source(self, slot):
        pass

    @abstractmethod
    def clear(self, slot):
        """Clears the data in the specified slot"""
        pass

    @abstractmethod
    def get(self, slot):
        """Returns the data in the specified slot"""
        pass

    @abstractmethod
    def set(self, slot, data):
        """Copy the date into the specified slot"""
        pass

    @abstractmethod
    def inner_product(self, slot_a, slot_b):
        """Computes the inner product of two data vectors

        Note: the vectors may be represented as multidimensional arrays,
        but these arrays must be contiguous for this operation to work.
        Although it would be possible to use flatten(), this would create a
        copy when the array is not contiguous, causing a hidden performance hit.
        """
        pass

    @abstractmethod
    def medium(self, slot_in, slot_out):
        """Applies the operator 1-Vscat.

        Note: does not apply the wrapping correction.
        """
        pass

    @abstractmethod
    def mix(self, weight_a, slot_a, weight_b, slot_b, slot_out):
        """Mixes two data arrays and stores the result in the specified slot"""
        pass

    @abstractmethod
    def propagator(self, slot_in, slot_out):
        """Applies the operator (L+1)^-1 x.
        """
        pass

    @abstractmethod
    def inverse_propagator(self, slot_in, slot_out):
        """Applies the operator (L+1) x .

        This operation is not needed for the Wavesim algorithm, but is provided for testing purposes,
        and can be used to evaluate the residue of the solution.
        """
        pass

    @abstractmethod
    def set_source(self, source):
        """Sets the source term for this domain.
        """
        pass

    def coordinates_f(self, dim):
        """Returns the Fourier-space coordinates along the specified dimension"""
        shapes = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        return (2 * torch.pi * torch.fft.fftfreq(self.shape[dim], self.pixel_size, device=self.device)).reshape(
            shapes[dim])

    def coordinates(self, dim):
        """Returns the real-space coordinates along the specified dimension, starting at 0"""
        shapes = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        return (torch.arange(self.shape[dim], device=self.device) * self.pixel_size).reshape(shapes[dim])
