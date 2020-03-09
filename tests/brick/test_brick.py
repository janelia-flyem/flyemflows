import sys
import pickle
from itertools import chain
from functools import partial

import pytest
import numpy as np
import pandas as pd

from neuclease.util import extract_subvol, box_intersection, Grid

from flyemflows.util import DebugClient, COMPRESSION_METHODS
from flyemflows.brick import ( Brick, BrickWall, generate_bricks_from_volume_source,
                               realign_bricks_to_new_grid, split_brick, assemble_brick_fragments,
                               pad_brick_data_from_volume_source, extract_halos )
from neuclease.util.box import overwrite_subvol

def box_as_tuple(box):
    if isinstance(box, np.ndarray):
        box = box.tolist()
    return (tuple(box[0]), tuple(box[1]))


def test_generate_bricks():
    grid = Grid( (10,20), (12,3) )
    bounding_box = np.array([(15,30), (95,290)])
    volume = np.random.randint(0,10, (100,300) )

    bricks, num_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume), DebugClient() )

    bricks = bricks.compute()
    assert len(bricks) == 9 * 14 == num_bricks
    
    for brick in bricks:
        assert isinstance( brick, Brick )
        assert brick.logical_box.shape == (2,2)
        assert brick.physical_box.shape == (2,2)

        # logical_box must be exactly one block
        assert ((brick.logical_box[1] - brick.logical_box[0]) == grid.block_shape).all()
        
        # Must be grid-aligned
        assert ((brick.logical_box - grid.offset) % grid.block_shape == 0).all()
        
        # Must not exceed bounding box
        assert (brick.physical_box == box_intersection( brick.logical_box, bounding_box )).all()
        
        # Volume shape must match
        assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
        
        # Volume data must match
        assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()

        # __sizeof__ must include the volume
        assert sys.getsizeof(brick) > sys.getsizeof(brick.volume)


def test_split_brick():
    grid = Grid( (10,20), (12,3) )
    volume = np.random.randint(0,10, (100,300) )
    
    # Test with the first brick in the grid
    physical_start = np.array(grid.offset)
    logical_start = physical_start // grid.block_shape * grid.block_shape
    logical_stop = logical_start + grid.block_shape
    
    physical_stop = logical_stop # Not always true, but happens to be true in this case.
    
    logical_box = np.array([logical_start, logical_stop])
    physical_box = np.array([physical_start, physical_stop])
    
    assert (logical_box == [(10,0), (20,20)]).all()
    assert (physical_box == [(12,3), (20,20)]).all()

    original_brick = Brick( logical_box, physical_box, extract_subvol(volume, physical_box) )

    # New grid scheme
    new_grid = Grid((2,10), (0,0))
    fragments = split_brick(new_grid, original_brick)
    boxes = list(box_as_tuple(frag.logical_box) for frag in fragments)
    
    assert boxes == [ # ((10, 0), (14, 10)),  # <--- Not present. These new boxes intersect with the original logical_box,
                      # ((10, 10), (14, 20)), # <--- but there is no physical data for them in the original brick.
                      ((12, 0), (14, 10)),
                      ((12, 10), (14, 20)),
                      ((14, 0), (16, 10)),
                      ((14, 10), (16, 20)),
                      ((16, 0), (18, 10)),
                      ((16, 10), (18, 20)),
                      ((18, 0), (20, 10)),
                      ((18, 10), (20, 20)) ]
    
    for frag in fragments:
        assert (frag.volume == extract_subvol(volume, frag.physical_box)).all()


def test_assemble_brick_fragments():
    volume = np.random.randint(0,10, (100,300) )
    
    logical_box = np.array( [(10, 20), (20, 120)] )

    # Omit the first and last boxes, to prove that the final
    # physical box ends up smaller than the logical box.
    
    # box_0 = np.array( [(10,20), (20,40)] )
    box_1 = np.array( [(10,40), (20,60)] )
    box_2 = np.array( [(10,60), (20,80)] )
    box_3 = np.array( [(10,80), (20,100)] )
    # box_4 = np.array( [(10,100), (20,120)] )

    # frag_0 = Brick( logical_box, box_0, extract_subvol(volume, box_0) ) # omit
    frag_1 = Brick( logical_box, box_1, extract_subvol(volume, box_1) )
    frag_2 = Brick( logical_box, box_2, extract_subvol(volume, box_2) )
    frag_3 = Brick( logical_box, box_3, extract_subvol(volume, box_3) )
    # frag_4 = Brick( logical_box, box_4, extract_subvol(volume, box_4) ) # omit

    assembled_brick = assemble_brick_fragments( [frag_1, frag_2, frag_3] )
    assert (assembled_brick.logical_box == logical_box).all()
    assert (assembled_brick.physical_box == [box_1[0], box_3[1]] ).all()
    
    physical_shape = assembled_brick.physical_box[1] - assembled_brick.physical_box[0]
    assert (assembled_brick.volume.shape == physical_shape).all()
    assert (assembled_brick.volume == extract_subvol(volume, assembled_brick.physical_box)).all()
    

def test_realign_bricks_to_new_grid():
    grid = Grid( (10,20), (12,3) )
    bounding_box = np.array([(15,30), (95,290)])
    volume = np.random.randint(0,10, (100,300) )

    original_bricks, _num_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume), DebugClient() )

    new_grid = Grid((20,10), (0,0))
    new_bricks = realign_bricks_to_new_grid(new_grid, original_bricks).compute()

    new_logical_boxes = list(brick.logical_box for brick in new_bricks)

    assert len(new_bricks) == 5 * 26 # from (0,30) -> (100,290)
    
    for logical_box, brick in zip(new_logical_boxes, new_bricks):
        assert isinstance( brick, Brick )
        assert (brick.logical_box == logical_box).all()

        # logical_box must be exactly one block
        assert ((brick.logical_box[1] - brick.logical_box[0]) == new_grid.block_shape).all()
        
        # Must be grid-aligned
        assert ((brick.logical_box - new_grid.offset) % new_grid.block_shape == 0).all()
        
        # Must not exceed bounding box
        assert (brick.physical_box == box_intersection( brick.logical_box, bounding_box )).all()
        
        # Volume shape must match
        assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
        
        # Volume data must match
        assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()


def test_realign_bricks_to_same_grid():
    """
    The realign function has a special optimization to
    avoid realigning bricks that are already aligned.
    """
    grid = Grid( (10,20), (12,3) )
    bounding_box = np.array([(15,30), (95,290)])
    def assert_if_called(box):
        assert False, ("Shouldn't get here, since the bricks were generated with lazy=True "
                       "and realignment shouldn't have attempted to split any bricks.")

    original_bricks, _num_bricks = generate_bricks_from_volume_source( bounding_box, grid, assert_if_called, DebugClient(), lazy=True )
    new_bricks = realign_bricks_to_new_grid(grid, original_bricks)
    
    import dask.bag
    assert isinstance(new_bricks, dask.bag.Bag)
    
    # If we attempt to realign to a different grid,
    # we'll get an assertion because it will have to call create_brick_volume, above.
    with pytest.raises(AssertionError):
        realign_bricks_to_new_grid(Grid((20,10)), original_bricks).compute()
        

def test_pad_brick_data_from_volume_source():
    source_volume = np.random.randint(0,10, (100,300) )
    logical_box = [(1,0), (11,20)]
    physical_box = [(3,8), (7, 13)]
    brick = Brick( logical_box, physical_box, extract_subvol(source_volume, physical_box) )
    
    padding_grid = Grid( (5,5), offset=(1,0) )
    padded_brick = pad_brick_data_from_volume_source( padding_grid, partial(extract_subvol, source_volume), brick )
    
    assert (padded_brick.logical_box == brick.logical_box).all()
    assert (padded_brick.physical_box == [(1,5), (11, 15)]).all()
    assert (padded_brick.volume == extract_subvol(source_volume, padded_brick.physical_box)).all()


def test_pad_brick_data_from_volume_source_NO_PADDING_NEEDED():
    source_volume = np.random.randint(0,10, (100,300) )
    logical_box = [(1,0), (11,20)]
    physical_box = [(6,10), (11, 15)]
    brick = Brick( logical_box, physical_box, extract_subvol(source_volume, physical_box) )
    
    padding_grid = Grid( (5,5), offset=(1,0) )
    padded_brick = pad_brick_data_from_volume_source( padding_grid, partial(extract_subvol, source_volume), brick )

    assert padded_brick is brick, "Expected to get the same brick back."


def test_generate_bricks_WITH_HALO():
    halo = 1
    halo_shape = np.array([1,1])
    grid = Grid( (10,20), (12,3), halo )
    bounding_box = np.array([(15,30), (95,290)])
    volume = np.random.randint(0,10, (100,300) )

    bricks, num_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume), DebugClient() )
    bricks = bricks.compute()

    assert len(bricks) == 9 * 14 == num_bricks
    
    for brick in bricks:
        assert isinstance( brick, Brick )
        assert brick.logical_box.shape == (2,2)
        assert brick.physical_box.shape == (2,2)

        # logical_box must be exactly one block
        assert ((brick.logical_box[1] - brick.logical_box[0]) == grid.block_shape).all()
        
        # Must be grid-aligned
        assert ((brick.logical_box - grid.offset) % grid.block_shape == 0).all()
        
        # Physical == logical+halo, except for bounding-box edges
        assert (brick.physical_box == box_intersection( brick.logical_box + (-halo_shape, halo_shape), bounding_box )).all()
        
        # Volume shape must match
        assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
        
        # Volume data must match
        assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()

def test_split_brick_WITH_HALO():
    halo = 1
    grid = Grid( (10,20), (12,3), halo )
    volume = np.random.randint(0,10, (100,300) )
    
    # Test with the first brick in the grid
    physical_start = np.array(grid.offset)
    logical_start = physical_start // grid.block_shape * grid.block_shape
    logical_stop = logical_start + grid.block_shape
    
    physical_stop = logical_stop+halo # Not always true, but happens to be true in this case.
    
    logical_box = np.array([logical_start, logical_stop])
    physical_box = np.array([physical_start, physical_stop])
    
    assert (logical_box == [(10,0), (20,20)]).all()
    assert (physical_box == [(12,3), (21,21)]).all()

    original_brick = Brick( logical_box, physical_box, extract_subvol(volume, physical_box) )

    # New grid scheme
    new_grid = Grid((2,10), (0,0))
    
    try:
        _fragments = split_brick(new_grid, original_brick)
    except AssertionError:
        pass # Expected failure: Forbidden to split bricks that have a halo
    else:
        assert False, "Did not encounter the expected assertion.  split_brick() should fail for bricks that have a halo."


def test_realign_bricks_to_new_grid_WITH_HALO():
    grid = Grid( (10,20), (12,3) )
    bounding_box = np.array([(15,30), (95,290)])
    volume = np.random.randint(0,10, (100,300) )

    original_bricks, _num_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume), DebugClient() )

    halo = 1
    halo_shape = np.array([1,1])
    new_grid = Grid((20,10), (0,0), halo)
    new_bricks = realign_bricks_to_new_grid(new_grid, original_bricks).compute()

    new_logical_boxes = list(brick.logical_box for brick in new_bricks)

    assert len(new_bricks) == 5 * 26, f"{len(new_bricks)}" # from (0,30) -> (100,290)
    
    for logical_box, brick in zip(new_logical_boxes, new_bricks):
        assert isinstance( brick, Brick ), f"Got {type(brick)}"
        assert (brick.logical_box == logical_box).all()

        # logical_box must be exactly one block
        assert ((brick.logical_box[1] - brick.logical_box[0]) == new_grid.block_shape).all()
        
        # Must be grid-aligned
        assert ((brick.logical_box - new_grid.offset) % new_grid.block_shape == 0).all()
        
        # Should match logical_box+halo, except for edges
        assert (brick.physical_box == box_intersection( brick.logical_box + (-halo_shape, halo_shape), bounding_box )).all()
        
        # Volume shape must match
        assert (brick.volume.shape == brick.physical_box[1] - brick.physical_box[0]).all()
        
        # Volume data must match
        assert (brick.volume == extract_subvol( volume, brick.physical_box )).all()


def test_compression():
    vol_box = [(0,0,0), (100,100,120)]
    volume = np.random.randint(10, size=vol_box[1], dtype=np.uint64)
    
    for method in COMPRESSION_METHODS:
        wall = BrickWall.from_accessor_func(vol_box, Grid((64,64,128)), lambda box: extract_subvol(volume, box), compression=method)

        # Compress them all
        wall.bricks.map(Brick.compress).compute()
        
        def check_pickle(brick):
            pickle.dumps(brick)

        # Compress them all
        wall.bricks.map(check_pickle).compute()
        
        def check_brick(brick):
            assert (brick.volume.shape == (brick.physical_box[1] - brick.physical_box[0])).all()
            assert (brick.volume == extract_subvol(volume, brick.physical_box)).all()
        
        # Check them all (implicit decompression)
        wall.bricks.map(check_brick).compute()


def test_extract_halos():
    halo = 1
    grid = Grid( (10,20), (0,0), halo )
    bounding_box = np.array([(15,30), (95,290)])
    volume = np.random.randint(0,10, (100,300) )

    bricks, _num_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume), DebugClient() )

    outer_halos = extract_halos(bricks, grid, 'outer').compute()
    inner_halos = extract_halos(bricks, grid, 'inner').compute()

    for halo_type, halo_bricks in zip(('outer', 'inner'), (outer_halos, inner_halos)):
        for hb in halo_bricks:
            # Even bricks on the edge of the volume
            # (which have smaller physical boxes than logical boxes)
            # return halos which correspond to the original
            # logical box (except for the halo axis).
            # (Each halo's "logical box" still corresponds to
            # the brick it was extracted from.)
            if halo_type == 'outer':
                assert (hb.physical_box[0] != hb.logical_box[0]).sum() == 1
                assert (hb.physical_box[1] != hb.logical_box[1]).sum() == 1
            else:
                assert (hb.physical_box != hb.logical_box).sum() == 1

            # The bounding box above is not grid aligned,
            # so blocks on the volume edge will only have partial data
            # (i.e. a smaller physical_box than logical_box)
            # However, halos are always produced to correspond to the logical_box size,
            # and zero-padded if necessary to achieve that size.
            # Therefore, only compare the actually valid portion of the halo here with the expected volume.
            # The other voxels should be zeros.
            valid_box = box_intersection(bounding_box, hb.physical_box)
            halo_vol = extract_subvol(hb.volume, valid_box - hb.physical_box[0])
            expected_vol = extract_subvol(volume, valid_box)
            assert (halo_vol == expected_vol).all()
            
            # Other voxels should be zero
            full_halo_vol = hb.volume.copy()
            overwrite_subvol(full_halo_vol, valid_box - hb.physical_box[0], 0)
            assert (full_halo_vol == 0).all()

    rows = []
    for hb in chain(outer_halos):
        rows.append([*hb.physical_box.flat, hb, 'outer'])

    for hb in chain(inner_halos):
        rows.append([*hb.physical_box.flat, hb, 'inner'])
    
    halo_df = pd.DataFrame(rows, columns=['y0', 'x0', 'y1', 'x1', 'brick', 'halo_type'])
    
    halo_counts = halo_df.groupby(['y0', 'x0', 'y1', 'x1']).size()

    # Since the bricks' physical boxes are all clipped to the overall bounding-box,
    # every outer halo should have a matching inner halo from a neighboring brick.
    # (This would not necessarily be true for Bricks that are initialized from a sparse mask.)
    assert halo_counts.min() == 2
    assert halo_counts.max() == 2
    
    for _box, halos_df in halo_df.groupby(['y0', 'x0', 'y1', 'x1']):
        assert set(halos_df['halo_type']) == set(['outer', 'inner'])

        brick0 = halos_df.iloc[0]['brick']
        brick1 = halos_df.iloc[1]['brick']
        assert (brick0.volume == brick1.volume).all()


def test_extract_halos_subsets():
    halo = 1
    grid = Grid( (10,20), (0,0), halo )
    bounding_box = np.array([(15,30), (95,290)])
    volume = np.random.randint(0,10, (100,300) )

    bricks, _num_bricks = generate_bricks_from_volume_source( bounding_box, grid, partial(extract_subvol, volume), DebugClient() )

    def bricks_to_df(bricks):
        rows = []
        for brick in bricks:
            rows.append([*brick.physical_box.flat, brick.volume])
        df = pd.DataFrame(rows, columns=['y0', 'x0', 'y1', 'x1', 'brickvol'])
        df = df.sort_values(['y0', 'x0', 'y1', 'x1']).reset_index(drop=True)
        return df

    def check(all_halos, lower_halos, upper_halos):
        all_df = bricks_to_df(all_halos)
        lower_df = bricks_to_df(lower_halos)
        upper_df = bricks_to_df(upper_halos)
        
        combined_df = pd.concat([lower_df, upper_df], ignore_index=True).sort_values(['y0', 'x0', 'y1', 'x1'])
        combined_df.reset_index(drop=True, inplace=True)
    
        assert (all_df[['y0', 'x0', 'y1', 'x1']] == combined_df[['y0', 'x0', 'y1', 'x1']]).all().all()
        for a, b in zip(all_df['brickvol'].values, combined_df['brickvol'].values):
            assert (a == b).all()
    
    # Check that 'all' is the same as combining 'lower' and 'upper'
    all_outer_halos = extract_halos(bricks, grid, 'outer', 'all').compute()
    lower_outer_halos = extract_halos(bricks, grid, 'outer', 'lower').compute()
    upper_outer_halos = extract_halos(bricks, grid, 'outer', 'upper').compute()

    all_inner_halos = extract_halos(bricks, grid, 'inner', 'all').compute()
    lower_inner_halos = extract_halos(bricks, grid, 'inner', 'lower').compute()
    upper_inner_halos = extract_halos(bricks, grid, 'inner', 'upper').compute()

    check(all_outer_halos, lower_outer_halos, upper_outer_halos)
    check(all_inner_halos, lower_inner_halos, upper_inner_halos)


if __name__ == "__main__":
    import dask.config
    dask.config.set(scheduler="synchronous")
    args = ['-s', '--tb=native', '--pyargs', 'tests.brick.test_brick']
    #args = ['-k', 'realign_bricks_to_same_grid'] + args
    pytest.main(args)
