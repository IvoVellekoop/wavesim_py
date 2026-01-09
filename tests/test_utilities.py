import pytest
import numpy as np
from random import choice

from wavesim.engine import combine, edges, pad, NumpyArray, shape_like, BlockArray, block_edges
from wavesim.utilities.create_source import point_source, gaussian_beam, plane_wave
from wavesim.utilities import add_absorbing_boundaries, all_close, create_source
from . import random_vector

"""Test of the utility functions."""


@pytest.mark.parametrize("size", [(5, 4, 6), (7, 15, 32), (3, 5, 6)])
@pytest.mark.parametrize("n_domains", [(1, 2, 3), (3, 3, 3), (2, 4, 1)])
def test_partition_combine(size: shape_like, n_domains: shape_like):
    x = random_vector(size)

    partitions = BlockArray(x, n_domains=n_domains)
    combined = combine(partitions)
    assert all_close(combined, x)
    assert partitions.blocks.shape == n_domains


@pytest.mark.parametrize("size", [(17, 15, 32), (13, 26, 11)])
@pytest.mark.parametrize("n_domains", [(2, 5, 3)])
@pytest.mark.parametrize("width", [1, 5])
def test_partition_slice(size: shape_like, n_domains: shape_like, width: int):
    # combining slicing and partitioning
    x = random_vector(size)
    x_sub = x[width:, width:-width, width:]
    partitions_sub1 = BlockArray(x_sub, n_domains=n_domains)
    assert partitions_sub1.blocks.shape == n_domains
    assert partitions_sub1.shape == x_sub.shape

    partitions_sub2 = BlockArray(x, n_domains=n_domains)[width:, width:-width, width:]
    assert partitions_sub2.shape == x_sub.shape
    assert all_close(partitions_sub1, partitions_sub2)


@pytest.mark.parametrize("n_domains", [(0, 0, 0), (4, 3, 3)])
def test_partition_with_invalid_input(n_domains: shape_like):
    array = random_vector((3, 3, 3))
    with pytest.raises(ValueError):
        BlockArray(array, n_domains=n_domains)


def test_edges():
    x = random_vector((5, 4, 6))
    widths = ((1, 1), (0, 0), (2, 3))
    edges_ = edges(x, widths)
    for d, e in enumerate(edges_):
        for side in range(2):
            w = widths[d][side]
            slices = [slice(None)] * x.ndim
            slices[d] = (slice(-w, None) if w != 0 else slice(-1, -1)) if side == 1 else slice(None, w)
            shape = [5, 4, 6]
            shape[d] = w
            assert e[side].shape == tuple(shape)
            assert all_close(e[side], x[slices])


def test_block_edges():
    x = random_vector((8, 4, 6))
    bx = BlockArray(x, boundaries=[(2,), (), (1, 3)], copy=True)
    assert bx.blocks.shape == (2, 1, 3)
    widths = ((1, 2), (0, 0), (1, 1))
    edges_ = block_edges(bx, widths)
    assert edges_.shape == (2, 1, 3, 3, 2)
    assert all_close(edges_[0, 0, 0, 0, 0], x[:1, :, 0:1])
    assert all_close(edges_[0, 0, 0, 0, 1], x[:2, :, 0:1])
    assert all_close(edges_[1, 0, 0, 0, 0], x[2:3, :, 0:1])
    assert all_close(edges_[1, 0, 0, 0, 1], x[-2:, :, 0:1])
    assert edges_[0, 0, 0, 1, 0].shape == (2, 0, 1)


def test_pad():
    x = random_vector((5, 4, 6))
    widths = [(1, 2), (0, 0), (3, 1)]
    padded = pad(x, widths)
    assert padded.shape == tuple(x.shape[d] + sum(w) for d, w in enumerate(widths))


@pytest.mark.parametrize(
    "boundary_widths, strength",
    [
        (5, 0.5),
        (((3, 2), (2, 1), (1, 6)), 0.5),
    ],
)
def test_add_absorbing_boundaries(boundary_widths, strength):
    permittivity = NumpyArray(np.ones((4, 4, 4)))
    result, roi = add_absorbing_boundaries(permittivity, boundary_widths, strength)
    if isinstance(boundary_widths, int):
        boundary_widths = ((boundary_widths, boundary_widths),) * 3
    assert result.shape == tuple(permittivity.shape + np.sum(boundary_widths, axis=1))
    p = result.gather()
    assert np.all(p.imag >= 0)
    assert 0.8 * strength < p.imag.max() < strength


def test_construct_source_point():
    source = point_source(position=[0, 0, 0], pixel_size=0.25)[0]
    assert source.shape == (1, 1, 1)
    assert np.all(source == 1.0)
    assert source.dtype == np.complex64


@pytest.mark.parametrize(
    "shape, position, source_plane",
    [
        [(4, 5), (0, 2, 2.5), 'yz'],  # 2D source in YZ plane
        [(3, 5), (1.5, 0, 2.5), 'xz'],  # 2D source in XZ plane
        [(3, 4), (1.5, 2, 0), 'xy'],  # 2D source in XY plane
        [(3), (1.5, 0, 0), 'x'],  # 1D source in X direction
        [(4), (0, 2, 0), 'y'],  # 1D source in Y direction
        [(5), (0, 0, 2.5), 'z'],  # 1D source in Z direction
    ],
)  # 1D source in Z direction
@pytest.mark.parametrize("source_type", ["plane_wave", "gaussian_beam"])
def test_construct_source(source_type, shape, position, source_plane):
    if source_type == "plane_wave":
        source = plane_wave(
            shape=shape, 
            origin='center',
            position=position, 
            source_plane=source_plane,
            pixel_size=0.25
        )[0].squeeze()
        assert np.all(source == 1.0)
    elif source_type == "gaussian_beam":
        source = gaussian_beam(
            shape=shape, 
            origin='center',
            position=position, 
            source_plane=source_plane,
            pixel_size=0.25
        )[0].squeeze()
        # Check that the center is maximum and the edges are minimum
        assert np.all(source[*np.asarray(source.shape) // 2] == source.max())
        if isinstance(shape, int):
            assert np.all(source[0] == source.min())
            assert np.all(source[-1] == source.min())
        elif len(shape) == 2:
            assert np.all(source[0, 0] == source.min())
            assert np.all(source[0, -1] == source.min())
            assert np.all(source[-1, 0] == source.min())
            assert np.all(source[-1, -1] == source.min())
    # assert source.shape == shape


@pytest.mark.parametrize("source_type", [plane_wave, gaussian_beam])
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 4, 5)])
def test_construct_source_invalid_shape(source_type, shape):
    with pytest.raises(AssertionError):
        source_type(shape, origin='topleft', position=[0, 0, 0], source_plane='xy', pixel_size=0.25)


def test_construct_source_invalid_type():
    with pytest.raises(AttributeError):
        create_source.invalid_source_type()


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

invalid_source_positions = [[3], [0, 0], [1, 2]]  # not a list of 3D coordinates
invalid_source_shapes = [[(1, 1, 1), 'yz'], [(2, 3, 4), 'xy']]  # not in a 2D plane (for plane_wave and gaussian_beam) -> AssertionError for source generation through create_source.src() where src can be plane wave or gaussian

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


def ishape_pos():
    """Choose a random invalid source shape and valid source position from the predefined lists, 
    the former within the given index range."""
    return choice(invalid_source_shapes), choice(valid_source_positions)


@pytest.mark.parametrize(
    "src_shape_ipos", 
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
def test_source_invalid_position(src_shape_ipos):
    """Test for source position. The function should raise an error if the source position(s) are not valid."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 3  # pixel size in micrometer (μm)

    with pytest.raises(ValueError):
        sources = []
        for i in range(len(src_shape_ipos)):
            if src_shape_ipos[i][0]==point_source:
                sources.append((
                    src_shape_ipos[i][0](
                        position=src_shape_ipos[i][2], 
                        pixel_size=pixel_size
                    )
                ))
            else:
                sources.append((
                    src_shape_ipos[i][0](
                        shape=src_shape_ipos[i][1][0], 
                        origin='topleft',
                        position=src_shape_ipos[i][2], 
                        source_plane=src_shape_ipos[i][1][1], 
                        pixel_size=pixel_size
                    )
                ))


@pytest.mark.parametrize(
    "src_ishape_pos",
    [
        [ (plane_wave, *ishape_pos()) ],  # 1 source
        [ (gaussian_beam, *ishape_pos()) ], 
        [ (src_(), *ishape_pos()), (src_(), *ishape_pos()) ], # 2 sources
        [ (plane_wave, *ishape_pos()), (src_(), *ishape_pos()) ], 
        [ (gaussian_beam, *ishape_pos()), (src_(), *ishape_pos()) ], 
        [ (src_(), *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()) ],  # 3 sources
        [ (plane_wave, *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()) ], 
        [ (gaussian_beam, *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()) ], 
        [ (src_(), *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()) ],  # 4 sources
        [ (plane_wave, *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()) ], 
        [ (gaussian_beam, *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()), (src_(), *ishape_pos()) ]
    ]
)
def test_source_invalid_shape(src_ishape_pos):
    """Test for source shape. The function should raise an error if the source shape(s) are not valid."""
    # Parameters
    wavelength = 1.0  # wavelength in micrometer (μm)
    pixel_size = wavelength / 3  # pixel size in micrometer (μm)

    with pytest.raises(AssertionError):
        [ (src(
            shape=shape_plane[0], 
            origin='topleft',
            position=position, 
            source_plane=shape_plane[1], 
            pixel_size=pixel_size
        )) for src, shape_plane, position in src_ishape_pos ]
