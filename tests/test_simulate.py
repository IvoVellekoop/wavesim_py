""" Tests to validate the inputs to the simulate function"""

import pytest
import cupy as cp
import numpy as np
from random import choice

from wavesim.utilities.create_source import point_source, gaussian_beam, plane_wave
from wavesim.simulate import simulate
from . import random_permittivity


@pytest.mark.parametrize(
    "n_size",
    [
        (50, 1),
        (50, 50),
        (1, 50)
    ],
)
def test_permittivity_shape(n_size: tuple[int, ...]):
    """Test for permittivity shape. The function should raise an error if the permittivity shape is not valid."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 4  # pixel size in micrometer (μm)

    # Create permittivity map with invalid shape
    permittivity = random_permittivity(n_size)

    source_values, source_position = point_source(
        position=[0, 0, 0],  # source position at the starting edge of the domain in micrometer (μm)
        pixel_size=pixel_size
    )

    # Run the wavesim iteration and get the computed field
    with pytest.raises(ValueError):
        simulate(permittivity=permittivity, 
                 sources=[ (source_values, source_position) ],
                 wavelength=wavelength, 
                 pixel_size=pixel_size
                 )[0]


@pytest.mark.parametrize(
    "permittivity_type",
    [
        "list",
        "tuple",
        "cupy",
    ]
)
def test_permittivity_type(permittivity_type: str):
    """Test for permittivity type. The function should raise an error if the permittivity type is not valid, i.e. not np.ndarray or NumpyArray."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 3  # pixel size in micrometer (μm)
    boundary_width = 2  # boundary width in micrometer (μm)

    # Create permittivity map with invalid type (not a numpy array)
    permittivity = random_permittivity((5, 42, 13))
    if permittivity_type == "list":
        permittivity = permittivity.d.tolist()
    elif permittivity_type == "tuple":
        permittivity = tuple(map(tuple, permittivity.d))
    elif permittivity_type == "cupy":
        permittivity = cp.asarray(permittivity.d)
    
    source_values, source_position = point_source(
        position=[0, 0, 0],  # source position at the starting edge of the domain in micrometer (μm)
        pixel_size=pixel_size
    )

    # Run the wavesim iteration and get the computed field
    with pytest.raises(TypeError):
        simulate(permittivity=permittivity, 
                 sources=[ (source_values, source_position) ],
                 wavelength=wavelength, 
                 pixel_size=pixel_size, 
                 boundary_width=boundary_width,
                 max_iterations=1
                 )[0]


valid_source_types = [point_source, plane_wave, gaussian_beam]  
valid_source_positions = [[0, 0, 0], [4, 5, 6],
                          [0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0],
                          [0, 4, 5, 6], [1, 4, 5, 6], [2, 4, 5, 6]
                         ]
valid_source_shapes = [[(3, 4), 'xy'],  # 2D source in XY plane
                       [(4, 5), 'yz'],  # 2D source in YZ plane
                       [(3, 5), 'xz'],  # 2D source in XZ plane
                       [(3), 'x'],  # 1D source in X direction
                       [(4), 'y'],  # 1D source in Y direction
                       [(5), 'z']  # 1D source in Z direction
                      ]

invalid_source_positions = [[13, 1, 4], [0, 13, 1, 4]]  # out of bounds
invalid_source_shapes = [[(13, 4), 'xz'], [(2, 50), 'xy'], [(2, 25), 'xz'],  # out of bounds (larger than permittivity shape) -> ValueError for simulate()
                         [(2, 3, 4), 'xy']]  # not in a 2D plane (for plane_wave and gaussian_beam) -> IndexError for source generation through create_source.src() where src can be plane wave or gaussian

def src():
    """Choose a random valid source type from the predefined list."""
    return choice(valid_source_types)


def src_():
    """Choose a random valid source type from the predefined list, excluding point_source."""
    return choice(valid_source_types[1:])  # plane_wave or gaussian_beam


def shape_pos():
    """Choose a random valid source shape and position from the predefined lists."""
    return choice(valid_source_shapes), choice(valid_source_positions)


def shape_ipos():
    """Choose a random valid source shape and invalid source position from the predefined lists."""
    return choice(valid_source_shapes), choice(invalid_source_positions)


def ishape_pos(i, j):
    """Choose a random invalid source shape and valid source position from the predefined lists, 
    the former within the given index range."""
    return choice(invalid_source_shapes[i:j]), choice(valid_source_positions)


@pytest.mark.parametrize(
    "src_shape_pos", 
    [
        [ (point_source, *shape_ipos()) ],  # 1 source
        [ (plane_wave, *shape_ipos()) ], 
        [ (gaussian_beam, *shape_ipos()) ], 
        [ (point_source, *shape_ipos()), (src(), *shape_ipos()) ], # 2 sources
        [ (plane_wave, *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (gaussian_beam, *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (src(), *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (point_source, *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ],  # 3 sources
        [ (plane_wave, *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (gaussian_beam, *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (src(), *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ],
        [ (point_source, *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ],  # 4 sources
        [ (plane_wave, *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (gaussian_beam, *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ], 
        [ (src(), *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()), (src(), *shape_ipos()) ] 
    ]
)
def test_source_invalid_position(src_shape_pos):
    """Test for source position. The function should raise an error if the source position(s) are not valid for simulate, but valid for create_source and pass through that without any problems."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 3  # pixel size in micrometer (μm)
    boundary_width = 2  # boundary width in micrometer (μm)

    # Create permittivity map
    permittivity = random_permittivity((5, 42, 13))

    sources = []
    for i in range(len(src_shape_pos)):
        if src_shape_pos[i][0]==point_source:
            sources.append((
                src_shape_pos[i][0](
                    position=src_shape_pos[i][2], 
                    pixel_size=pixel_size
                )
            ))
        else:
            sources.append((
                src_shape_pos[i][0](
                    shape=src_shape_pos[i][1][0], 
                    origin='topleft', 
                    position=src_shape_pos[i][2], 
                    source_plane=src_shape_pos[i][1][1], 
                    pixel_size=pixel_size
                )
            ))

    # Run the wavesim iteration and get the computed field
    with pytest.raises(ValueError):
        simulate(
            permittivity=permittivity, 
            sources=sources, 
            wavelength=wavelength, 
            pixel_size=pixel_size, 
            boundary_width=boundary_width, 
            max_iterations=1
        )[0]


@pytest.mark.parametrize(
    "src_types",
    [
        [ plane_wave ],  # 1 source
        [ gaussian_beam ], 
        [ src_(), src_() ], # 2 sources
        [ plane_wave, src_() ], 
        [ gaussian_beam, src_() ], 
        [ src_(), src_(), src_() ],  # 3 sources
        [ plane_wave, src_(), src_() ], 
        [ gaussian_beam, src_(), src_() ], 
        [ src_(), src_(), src_(), src_() ],  # 4 sources
        [ plane_wave, src_(), src_(), src_() ], 
        [ gaussian_beam, src_(), src_(), src_() ]
    ]
)
@pytest.mark.parametrize(
    "ij",  # index range for invalid source shape
    [
        (0, 3),  # out of bounds (larger than permittivity shape) -> ValueError for simulate()
        (3, 4)  # not in a 2D plane (for plane_wave and gaussian_beam) -> IndexError for simulate(); and no error, so skip test, for simulate() (sources can be 3D or not in a 2D plane)
    ]
)
def test_source_invalid_shape(src_types, ij):
    """Test for source shape. The function should raise an error if the source shape(s) are not valid for simulate, but valid for create_source and pass through that without any problems."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 3  # pixel size in micrometer (μm)
    boundary_width = 2  # boundary width in micrometer (μm)

    # Create permittivity map
    permittivity = random_permittivity((5, 42, 13))

    # Test with primitive sources

    # if ij == (0, 3), simulate() gives ValueError because of out of bounds source shape (larger than permittivity shape)
    # else, i.e., if ij == (3, 4), create_source.src() gives an AssertionError for improper source shape (not 1D or 2D)
    error = ValueError if ij[0] == 0 else AssertionError
    with pytest.raises(error):
        sources = [ (src(
            shape=shape_plane[0], 
            origin='topleft',
            position=position, 
            source_plane=shape_plane[1], 
            pixel_size=pixel_size
        )) 
                    for src, shape_plane, position in [(src_types[0], *ishape_pos(*ij))] ]

        # Run the wavesim iteration and get the computed field
        simulate(
            permittivity=permittivity, 
            sources=sources, 
            wavelength=wavelength, 
            pixel_size=pixel_size, 
            boundary_width=boundary_width, 
            max_iterations=1
        )[0]

    # Test with custom sources

    # if ij == (3, 4), simulate should not give any error, as custom sources can be 3D or not in a 2D plane
    # For ij == (2, 5), simulate() gives ValueError because of out of bounds source shape (larger than permittivity shape)
    try:
        pos_ishape_list = [ ishape_pos(*ij) for _ in src_types ]
        sources_ = [ (np.random.randn(*shp), pos) for pos, shp in pos_ishape_list]
        simulate(
            permittivity=permittivity, 
            sources=sources_, 
            wavelength=wavelength, 
            pixel_size=pixel_size, 
            boundary_width=boundary_width, 
            max_iterations=1
        )[0]
    except:
        pytest.raises(ValueError)
