import numpy as np
from typing import Sequence
from scipy.signal.windows import gaussian

from wavesim.engine.index_utils import shape_like

def point_source(
    position: Sequence[float],
    pixel_size: float,
    amplitude: complex = 1.0, 
    ):
    """Define a point source with given amplitude (default 1), and position
    Args:
        position: a tuple of 3 or 4 floats, the position of the center of the source in micrometer (μm).
            3 floats (x, y, z) for solving the Helmholtz equation (scalar).
            4 floats (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector). Polarization axis must be 0, 1, or 2 for x, y, or z polarization, respectively.
        pixel_size: pixel size in micrometer (μm)
        amplitude: amplitude to scale the source by. Default is 1.0
    Returns:
        source_values: 3D (4D) array of complex numbers with shape (1, 1, 1) ((1, 1, 1, 1))
        position: a tuple of 3 (4) integers indicating (the polarization and) the position of the source in pixels.
    """
    check_source_parameters(position)
    p = polarization_(position)
    shape = (1, 1, 1, 1) if p else (1, 1, 1)
    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    position = tuple(position.astype(int))
    return np.full(shape=shape, fill_value=amplitude, dtype=np.complex64), position


def plane_wave(
    shape: shape_like, 
    origin: str,
    position: Sequence[float],
    source_plane: str,
    pixel_size: float,
    amplitude: complex = 1.0, 
    theta: float = None, 
    phi: float = None, 
    wavelength: float = None, 
):
    """Define a plane wave source. The source can be given an incident angle theta 
    or azimuthal angle phi in radians.

    Args:
        shape: shape of the source (3D array) in micrometer (μm)
        origin (str): Define the origin as 'center' or 'topleft.' The position of the source is defined with respect to the origin.
        position: a tuple of 3 or 4 floats, the position of the center/top left of the source in micrometer (μm).
            3 floats (x, y, z) for solving the Helmholtz equation (scalar).
            4 floats (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector). Polarization axis must be 0, 1, or 2 for x, y, or z polarization, respectively.
        source_plane: plane to define the source in. should be a string of one or two axes from x, y, and z. 
            For example, for a 2D plane wave source in the xz plane, source_plane='xz'. 
            The string is always sorted by alphabetical order, so 'xy' and 'yx' are both recognized as 'xy'.
        pixel_size: pixel size in micrometer (μm)
        amplitude: amplitude to scale the source by. Default is 1.0
        theta: angle of incidence of the source with respect to the source axis in radians
        phi: angle of incidence of the source in the plane orthogonal to the source axis in radians
        wavelength: wavelength in micrometer (μm). Must be defined when theta or phi is not None.
    Returns:
        source_values: 3D array of complex numbers with shape determined by shape and source_plane
        position: a tuple of 3 (4) integers indicating (the polarization and) the position of the source in pixels.
    """
    if origin not in ['center', 'topleft']:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")
    source_plane = ''.join(sorted(source_plane.lower()))

    check_source_parameters(position, shape, source_plane)
    p = polarization_(position)

    # convert shape from micrometers to pixels
    shape = (np.asarray(shape) / pixel_size).astype(int)
    shape = (shape,) if isinstance(shape, int) or isinstance(shape, np.int64) else tuple(shape)

    source_values = np.ones(shape=shape[0], dtype=np.complex64)
    if theta is not None:
        assert wavelength is not None, "wavelength must be given for an angled source with some theta"
        source_values = source_angled(source_values, theta, wavelength, pixel_size)

    if len(shape) == 2:  # a 2D plane wave
        source_values_ = np.ones(shape=shape[1], dtype=np.complex64)
        if phi is not None:
            assert wavelength is not None, "wavelength must be given for an angled source with some phi"
            source_values_ = source_angled(source_values_, phi, wavelength, pixel_size)
        source_values = source_values[:, None] * source_values_  # 2D plane wave

    source_values *= amplitude
    assert source_values.shape == shape, f"Source shape {source_values.shape} does not match expected shape {shape}"
    source_values = expand_source_shape(source_values, source_plane)
    
    source_values = source_values[None, ...] if p else source_values  # Add polarization dimension if polarization (p) is True
    if p:
        assert source_values.ndim == 4, f"Expected source shape {source_values.shape} to be 4D"
    else:
        assert source_values.ndim == 3, f"Expected source shape {source_values.shape} to be 3D"
    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    if origin == 'center':
        for i, axis in enumerate(source_plane):
            position[p + "xyz".index(axis)] -= shape[i] // 2
    position = tuple(position.astype(int))

    return source_values.astype(np.complex64), position


def gaussian_beam(
    shape: shape_like, 
    origin: str,
    position: Sequence[float],
    source_plane: str,
    pixel_size: float,
    amplitude: complex = 1.0, 
    theta: float = None, 
    phi: float = None, 
    wavelength: float = None, 
    alpha: float = 3, 
    sigma: float = None, 
):
    """Define a plane wave source with a Gaussian intensity profile, given by
    :math:`w(n) = e^{ -\frac{1}{2} { \left( \alpha \frac{n}{(m-1)/2} \right) }^2 } = e^{-n^2 / 2\sigma^2}`

    gaussian(m, sigma) from scipy.signal.windows gives the same output as gausswin(m, alpha) from MATLAB,
    where m is the number of points in the Gaussian window. sigma = (m - 1) / (2 * alpha)

    The Gaussian beam can be given an incident angle theta or azimuthal angle phi in radians.

    Args:
        shape: shape of the source (3D array) in micrometer (μm)
        origin (str): Define the origin as 'center' or 'topleft.' The position of the source is defined with respect to the origin.
        position: a tuple of 3 or 4 floats, the position of the center/top left of the source in micrometer (μm).
            3 floats (x, y, z) for solving the Helmholtz equation (scalar).
            4 floats (polarization axis, x, y, z) for solving time-harmonic Maxwell's equations for non-magnetic and non-birefringent materials (vector). Polarization axis must be 0, 1, or 2 for x, y, or z polarization, respectively.
        source_plane: plane to define the source in. should be a string of one or two axes from x, y, and z. 
            For example, for a 2D Gaussian beam source in the xz plane, source_plane='xz'.
            The string is always sorted by alphabetical order, so 'xy' and 'yx' are both recognized as 'xy'.
        pixel_size: pixel size in micrometer (μm)
        amplitude: amplitude to scale the source by. Default is 1.0
        theta: angle of incidence of the source with respect to the source axis in radians
        phi: angle of incidence of the source in the plane orthogonal to the source axis in radians
        wavelength: wavelength in micrometer (μm). Must be defined when theta or phi is not None.
        alpha: (optional, instead of sigma) width factor for Gaussian window. Default is 3.
        sigma: (optional, instead of alpha) standard deviation
    Returns:
        source_values: 3D array of complex numbers with shape determined by shape and source_plane
        position: a tuple of 3 (4) integers indicating (the polarization and) the position of the source in pixels.
    """
    if origin not in ['center', 'topleft']:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")
    source_plane = ''.join(sorted(source_plane.lower()))

    check_source_parameters(position, shape, source_plane)
    p = polarization_(position)

    # convert shape from micrometers to pixels
    shape = (np.asarray(shape) / pixel_size).astype(int)
    shape = (shape,) if isinstance(shape, int) or isinstance(shape, np.int64) else tuple(shape)

    m = shape[0]  # number of points in the Gaussian window
    if sigma is None:  # alpha is given as input and sigma is computed from it
        sigma = (m - 1) / (2 * alpha)  # Standard deviation for the Gaussian beam
    source_values = gaussian(m, sigma).astype(np.complex64)
    if theta is not None:
        assert wavelength is not None, "wavelength must be given for an angled source with some theta"
        source_values = source_angled(source_values, theta, wavelength, pixel_size)

    if len(shape) == 2:  # a 2D Gaussian beam
        source_values_ = gaussian(shape[1], sigma).astype(np.complex64)
        if phi is not None:
            assert wavelength is not None, "wavelength must be given for an angled source with some phi"
            source_values_ = source_angled(source_values_, phi, wavelength, pixel_size)
        source_values = source_values[:, None] * source_values_  # 2D Gaussian beam

    source_values *= amplitude
    assert source_values.shape == shape, f"Source shape {source_values.shape} does not match expected shape {shape}"
    source_values = expand_source_shape(source_values, source_plane)

    source_values = source_values[None, ...] if p else source_values  # Add polarization dimension if polarization (p) is True
    if p:
        assert source_values.ndim == 4, f"Expected source shape {source_values.shape} to be 4D"
    else:
        assert source_values.ndim == 3, f"Expected source shape {source_values.shape} to be 3D"
    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    if origin == 'center':
        for i, axis in enumerate(source_plane):
            position[p + "xyz".index(axis)] -= shape[i] // 2
    position = tuple(position.astype(int))

    return source_values.astype(np.complex64), position


def source_angled(source_values, angle, wavelength, pixel_size):
    kx = 2 * np.pi / wavelength * np.sin(angle)
    x = np.arange(1, source_values.shape[0] + 1) * pixel_size
    return source_values * np.exp(1j * kx * x)


def expand_source_shape(source_values, plane: str):
    if plane == 'x':
        return np.expand_dims(source_values, axis=(1, 2))
    elif plane == 'y':
        return np.expand_dims(source_values, axis=(0, 2))
    elif plane == 'z':
        return np.expand_dims(source_values, axis=(0, 1))
    elif plane == 'xy':
        return np.expand_dims(source_values, axis=(2))
    elif plane == 'yz':
        return np.expand_dims(source_values, axis=(0))
    elif plane == 'xz':
        return np.expand_dims(source_values, axis=(1))
    else:
        raise ValueError("source_plane should be in ['x', 'y', 'z', 'xy', 'yz', 'xz']")


def check_source_parameters(position, shape=None, source_plane=None):
    if len(tuple(position)) != 3 and len(tuple(position)) != 4:
        raise ValueError(f"Source position {position} invalid, should be a tuple of 3 (or 4) floats indicating the (polarization axis and the) position in x, y, and z axes in micrometer (μm)")

    if shape is not None and source_plane is not None:
        if source_plane in ['x', 'y', 'z']:
            try:
                len(shape) == 1
            except TypeError:
                isinstance(shape, int)
            else:
                print(f"Source shape {shape} invalid, should be 1D for source_plane='{source_plane}'")
        elif source_plane in ['xy', 'yz', 'xz']:
            assert len(shape) == 2, f"Source shape {shape} invalid, should be 2D for source_plane='{source_plane}'"
        else:
            raise ValueError("source_plane should be in ['x', 'y', 'z', 'xy', 'yz', 'xz']")


def polarization_(position):
    """Check whether there is a polarization axis (1) or not (0) and return it"""
    p = int(len(tuple(position)) == 4)  # polarization True (1) or False (0)
    if p:
        assert position[0] in [0, 1, 2], f"polarization axis, i.e., position[0], must be 0, 1, or 2 for x, y, or z polarization, respectively"
    return p
