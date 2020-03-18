import numpy as np

from neuclease.util import Grid, SparseBlockMask, round_box, extract_subvol

from ..util import persist_and_execute, downsample, DebugClient
from .brick import ( Brick, generate_bricks_from_volume_source, realign_bricks_to_new_grid, pad_brick_data_from_volume_source )

# Column names used below for logical box coordinates and physical box coordinates.
# FIXME: There might be a more elegant way to define thise via pandas multi-indexes...
LB_COLS = ['logical_z0', 'logical_y0', 'logical_x0', 'logical_z1', 'logical_y1', 'logical_x1']
PB_COLS = ['physical_z0', 'physical_y0', 'physical_x0', 'physical_z1', 'physical_y1', 'physical_x1']

LB_SHORT_COLS = ['lz0', 'ly0', 'lx0', 'lz1', 'ly1', 'lx1']
PB_SHORT_COLS = ['pz0', 'py0', 'px0', 'pz1', 'py1', 'px1']

class BrickWall:
    """
    Manages a (lazy) set of bricks within a Grid.
    Mostly just a convenience wrapper to simplify pipelines of transformations over RDDs of bricks.
    """
    ##
    ## Generic Constructor (See also: convenience constructors below) 
    ##

    def __init__(self, bounding_box, grid, bricks, num_bricks=None):
        """
        bounding_box: (start, stop) If you don't plan to use it later, you can set this to None.
        grid: Grid
        bricks: RDD of Bricks
        """
        self.bounding_box = np.array(bounding_box)
        self.grid = grid
        self.bricks = bricks

        # Valid after initialization, but could be None after various manipulations
        self.num_bricks = num_bricks
    
    ##
    ## Convenience Constructors
    ##

    @classmethod
    def from_accessor_func(cls, bounding_box, grid, volume_accessor_func=None, client=None, target_partition_size_voxels=None, sparse_boxes=None, lazy=False, compression=None):
        """
        Convenience constructor, taking an arbitrary volume_accessor_func.
        
        Args:
            bounding_box:
                (start, stop)
     
            grid:
                Grid (see brick.py)
     
            volume_accessor_func:
                Callable with signature: f(box) -> ndarray
                Note: The callable will be unpickled only once per partition, so initialization
                      costs after unpickling are only incurred once per partition.
     
            client:
                dask distributed.Client or suitable mock object.
     
            target_partition_size_voxels:
                Optional. If provided, the RDD partition lengths (i.e. the number of bricks per RDD partition)
                will be chosen to have (approximately) this many total voxels in each partition.
            
            sparse_boxes:
                A list of (physical) sparse boxes indicating which bricks should actually be present in the BrickWall.
                If not provided, all bricks within the bounding_box will be present. 

            lazy:
                If True, the bricks' data will not be created until their 'volume' member is first accessed.

            compression:
                If provided, the brick volume data will be serialized/stored in a compressed format.
                See ``flyemflows.util.compressed_volume.COMPRESSION_METHODS``
        """
        bounding_box = np.asarray(bounding_box)
        
        if client is None:
            client = DebugClient()
        
        if target_partition_size_voxels is None:
            if sparse_boxes is None:
                total_voxels = np.prod(bounding_box[1] - bounding_box[0])
            else:
                if not hasattr(sparse_boxes, '__len__'):
                    sparse_boxes = list(sparse_boxes)
                total_voxels = sum( map(lambda physbox: np.prod(physbox[1] - physbox[0]), sparse_boxes ) )
            
            ncores = sum(client.ncores().values())
            voxels_per_thread = total_voxels / ncores
            target_partition_size_voxels = (voxels_per_thread // 2) # Arbitrarily aim for 2 partitions per thread

        block_size_voxels = np.prod(grid.block_shape)
        rdd_partition_length = target_partition_size_voxels // block_size_voxels

        bricks, num_bricks = generate_bricks_from_volume_source(bounding_box, grid, volume_accessor_func, client, rdd_partition_length, sparse_boxes, lazy, compression=compression)
        return BrickWall( bounding_box, grid, bricks, num_bricks )


    @classmethod
    def from_volume_service(cls, volume_service, scale=0, bounding_box_zyx=None, client=None, target_partition_size_voxels=None, halo=0, sparse_block_mask=None, lazy=False, compression=None):
        """
        Convenience constructor, initialized from a VolumeService object.
        
        Args:
            volume_service:
                An instance of a VolumeService
        
            bounding_box_zyx:
                (start, stop) Optional.
                Bounding box to restrict the region of fetched blocks, always
                specified in FULL-RES coordinates, even if you are passing scale > 0
                If not provided, volume_service.bounding_box_zyx is used.
     
            scale:
                Brick data will be fetched at this scale.
                (Note: The bricks' sizes will still be the the full volume_service.preferred_message_shape,
                       but the overall bounding-box of the BrickWall be scaled down.) 
     
            client:
                dask distributed.Client
     
            target_partition_size_voxels:
                Optional. If provided, the RDD partition lengths (i.e. the number of bricks per RDD partition)
                will be chosen to have (approximately) this many total voxels in each partition.
            
            halo:
                If provided, add a halo to the brick grid that will be used to fetch the data.
                Depending on your use-case and/or input source, this can be faster than applying
                a halo after-the-fact, which involves shuffling data across the cluster.
            
            sparse_block_mask:
                Instance of SparseBlockMask
            
            lazy:
                If True, the bricks' data will not be created until their 'volume' member is first accessed.
            
            compression:
                If provided, the brick volume data will be serialized/stored in a compressed format.
                See ``flyemflows.util.compressed_volume.COMPRESSION_METHODS``
        """
        grid = Grid(volume_service.preferred_message_shape, (0,0,0), halo)
        
        if bounding_box_zyx is None:
            bounding_box_zyx = volume_service.bounding_box_zyx

        bounding_box_zyx = np.asarray(bounding_box_zyx)
                
        if scale == 0:
            downsampled_box = bounding_box_zyx
        else:
            full_box = bounding_box_zyx
            downsampled_box = np.zeros((2,3), dtype=int)
            downsampled_box[0] = full_box[0] // 2**scale # round down
            
            # Proper downsampled bounding-box would round up here...
            #downsampled_box[1] = (full_box[1] + 2**scale - 1) // 2**scale
            
            # ...but some some services probably don't do that, so we'll
            # round down to avoid out-of-bounds errors for higher scales. 
            downsampled_box[1] = full_box[1] // 2**scale

        sparse_boxes = None
        if sparse_block_mask is not None:
            # FIXME:
            # 
            #   It would save a lot of time in generate_bricks_from_volume_source() if we implemented
            #   a faster way to filter boxes in SparseBlockMask,
            #   and called it here.  Right now, workflows that process data in "slabs"
            #   end up passing the same SparseBlockMask for every slab, which gets processed from
            #   scratch in generate_bricks_from_volume_source() to filter boxes for each slab's bounding box.
            assert isinstance(sparse_block_mask, SparseBlockMask)
            assert scale == 0, "FIXME: I don't think the sparse feature works with scales other than 0."
            sparse_boxes = sparse_block_mask.sparse_boxes(grid)
            if len(sparse_boxes) == 0:
                # Some workflows check for this message; if you change it, change those checks!
                raise RuntimeError("SparseBlockMask selects no blocks at all!")

        return BrickWall.from_accessor_func( downsampled_box,
                                             grid,
                                             lambda box: volume_service.get_subvolume(box, scale),
                                             client,
                                             target_partition_size_voxels,
                                             sparse_boxes,
                                             lazy,
                                             compression=compression )


    ##
    ## Operations
    ##
    def drop_empty(self):
        """
        Remove all empty (completely zero) bricks from the BrickWall.
        """
        filtered_bricks = self.bricks.filter(lambda brick: brick.volume.any())
        def compress(brick):
            brick.compress()
            return brick
        filtered_bricks = filtered_bricks.map(compress)
        return BrickWall( self.bounding_box, self.grid, filtered_bricks, None ) # Don't know num_bricks any more


    def realign_to_new_grid(self, new_grid, output_accessor_fn=None):
        """
        Chop upand the Bricks in this BrickWall reassemble them into a new BrickWall,
        tiled according to the given new_grid.
        
        Note: Requires data shuffling.
        
        Returns: A a new BrickWall, with a new internal RDD for bricks.
        """
        new_bricks = realign_bricks_to_new_grid( new_grid, self.bricks, output_accessor_fn )
        new_wall = BrickWall( self.bounding_box, new_grid, new_bricks ) # Don't know num_bricks any more
        return new_wall


    def fill_missing(self, volume_accessor_func, padding_grid=None):
        """
        For each brick whose physical_box does not extend to all edges of its logical_box,
        fill the missing space with data from the given volume accessor.
        
        Args:
            volume_accessor_func:
                Callable with signature: f(box) -> ndarray
                Note: The callable will be unpickled only once per partition, so initialization
                      costs after unpickling are only incurred once per partition.
            
            padding_grid:
                (Optional.) Need not be identical to the BrickWall's native grid,
                but must divide evenly into it. If not provided, the native grid is used.
        """
        if padding_grid is None:
            padding_grid = self.grid
            
        def pad_brick(brick):
            return pad_brick_data_from_volume_source(padding_grid, volume_accessor_func, brick)
        
        padded_bricks = self.bricks.map( pad_brick )
        new_wall = BrickWall( self.bounding_box, self.grid, padded_bricks, self.num_bricks )
        return new_wall


    def translate(self, offset_zyx):
        """
        Translate all bricks by the given offset.
        Does not change the brick data, just the logical/physical boxes.
        
        Also, translates the bounding box and grid.
        """
        new_bounding_box = None
        if self.bounding_box is not None:
            new_bounding_box = self.bounding_box + offset_zyx

        new_grid = Grid( self.grid.block_shape, self.grid.offset + offset_zyx )

        def translate_brick(brick):
            # FIXME: This is needlessly inefficient for compressed bricks,
            #        since it uncompresses and recompresses the volume,
            #        but currently the Brick constructor doesn't allow me to
            #        provide the compressed form directly.
            return Brick( brick.logical_box + offset_zyx,
                          brick.physical_box + offset_zyx,
                          brick.volume,
                          location_id=tuple(brick.logical_box[0] // new_grid.block_shape),
                          compression=brick.compression )
        translated_bricks = self.bricks.map( translate_brick )
        
        return BrickWall( new_bounding_box, new_grid, translated_bricks, self.num_bricks )

    
    def persist_and_execute(self, description=None, logger=None, optimize_graph=True):
        self.bricks = persist_and_execute(self.bricks, description, logger, optimize_graph)
    

    def map_brick_volumes(self, f):
        """
        Given a function that accepts a Brick
        and returns an updated volume (ndarray),
        Return a new BrickWall with identical bricks, except for the new volume.
        The logical_box, physical_box, location_id, and compression are copied from the original bricks.
        """
        def _apply_to_brick(brick):
            assert isinstance(brick, Brick)
            newvol = f(brick)
            assert isinstance(newvol, np.ndarray)

            return Brick( brick.logical_box, brick.physical_box, newvol,
                          location_id=brick.location_id, compression=brick.compression )
            
        new_bricks = self.bricks.map(_apply_to_brick)
        return BrickWall( self.bounding_box, self.grid, new_bricks, self.num_bricks )
    

    def downsample(self, block_shape, method):
        """
        See util.downsample for available methods
        
        Note:
            If the downsampling block_shape does not
            perfectly divide into the brick's physical_box start or stop,
            voxels on the edge of the volume will be discarded before downsampling. 
        """
        assert block_shape[0] == block_shape[1] == block_shape[2], \
            "Currently, downsampling must be isotropic"

        factor = block_shape[0]
        def downsample_brick(brick):
            assert (brick.logical_box % factor == 0).all()

            # If the factor doesn't perfectly divide into
            # the brick's physical dimensions,
            # then chop off the edges until it does.
            if (brick.physical_box % factor != 0).any():
                clipped_box = round_box(brick.physical_box, factor, 'in')
                volume = extract_subvol(brick.volume, clipped_box - brick.physical_box[0])
            else:
                clipped_box = brick.physical_box
                volume = brick.volume
        
            downsampled_volume = downsample(volume, factor, method)
            downsampled_logical_box = brick.logical_box // factor
            downsampled_physical_box = clipped_box // factor
            
            return Brick(downsampled_logical_box, downsampled_physical_box, downsampled_volume, compression=brick.compression)

        new_bounding_box = None
        if self.bounding_box is not None:
            new_bounding_box = self.bounding_box // factor
        new_grid = Grid( self.grid.block_shape // factor, self.grid.offset // factor )
        new_bricks = self.bricks.map( downsample_brick )
        
        return BrickWall( new_bounding_box, new_grid, new_bricks, self.num_bricks )

    @classmethod
    def bricks_as_ddf(cls, bricks, logical=True, physical=False, names='short', set_index=False, wall_box=None, wall_grid=None):
        """
        Return given dask Bag of bricks as a dask DataFrame,
        with columns for the logical and physical boxes.
        
        Args:
            bricks:
                A dask.Bag of bricks
            logical:
                If True, include columns for the logical box coordinates
            physical:
                If True, include columns for the physical box coordinates
            names:
                How to name the columns. Must be 'short' or 'long'.
                See LB_COLS and LB_SHORT_COLS
            set_index:
                If True, add an index to the DataFrame, which corresponds to
                the scan-order index of each brick's logical box.
                You must provide wall_box and wall_grid.
            wall_box:
                The overall bounding box of the brick set
                Used to compute the index.
            wall_grid:
                The Grid on which these bricks are laid out.
                Used to compute the index.
        
        Returns:
            dask.dataframe.DataFrame
        """
        cols = []

        if logical:
            if names == 'long':
                cols += LB_COLS
            else:
                cols += LB_SHORT_COLS

        if physical:
            if names == 'long':
                cols += PB_COLS
            else:
                cols += PB_SHORT_COLS

        assert names in ('short', 'long')
        def boxes_and_brick(brick):
            bounds = []
            if logical:
                bounds += list(brick.logical_box.flat)
            
            if physical:
                bounds += list(brick.physical_box.flat)

            if set_index:
                brick_index = cls.compute_brick_index(brick)
                return (brick_index, *bounds, brick)
            else:
                return (*bounds, brick)
        
        brick_table = bricks.map(boxes_and_brick)
        dtypes = {}
        if set_index:
            brick_table = brick_table.persist()
            # Figure out if the index is already sorted.
            brick_indexes = brick_table.pluck(0).compute()
            is_sorted = (brick_indexes == sorted(brick_indexes))
            dtypes['brick_index'] = np.int32
        
        dtypes.update({col: np.int32 for col in cols})
        dtypes['brick'] = object
        bricks_ddf = brick_table.to_dataframe(dtypes)

        if set_index:
            bricks_ddf = bricks_ddf.set_index('brick_index', sorted=is_sorted)

        return bricks_ddf


    @classmethod
    def compute_brick_index(cls, brick, wall_box, wall_grid):
        """
        For a given brick in a BrickWall corresponding to wall_box and wall_grid,
        compute a unique index based on the brick's logical_box.
        (We just compute the scan-order index.)
        Note that in the case of a sparsely populated brick wall,
        brick indexes will not be consecutive.
        """
        wall_box = round_box( wall_box, wall_grid.block_shape, 'out' )
        wall_shape = wall_box[1] - wall_box[0]
        
        wall_index_box = np.array(([0,0,0], wall_shape)) // wall_grid.block_shape
        wall_index_shape = wall_index_box[1] - wall_index_box[0]

        location_id = (brick.logical_box[0] - wall_box[0]) // wall_grid.block_shape

        def _scan_order_index(coord, shape):
            # Example:
            #  coord = (z,y,x,c)
            #  shape = (Z,Y,X,C)
            #  strides = (Y*X*C, X*C, C, 1)
            #  index = z*Y*X*C + y*X*C + x*C + c*1
            
            # Accumulate products in reverse
            reverse_shape = tuple(shape[::-1]) # (C,X,Y,Z)
            reverse_strides = np.multiply.accumulate((1,) + reverse_shape[:-1]) # (1, C, C*X, C*X*Y)

            strides = reverse_strides[::-1]
            index = (coord * strides).sum()
            return np.int32(index)

        return _scan_order_index(location_id, wall_index_shape)


    @classmethod
    def compute_brick_indexes(cls, logical_corners, wall_box, wall_grid):
        """
        For an array of brick locations (logical_box[0]) in a BrickWall
        corresponding to wall_box and wall_grid,
        compute unique indexes for each brick based on the brick's logical_box.
        (We just compute the scan-order index.)
        Note that in the case of a sparsely populated brick wall,
        brick indexes will not be consecutive.
        """
        logical_corners = np.asarray(logical_corners)
        assert logical_corners.ndim == 2
        assert logical_corners.shape[1] == 3
        
        wall_box = round_box( wall_box, wall_grid.block_shape, 'out' )
        wall_shape = wall_box[1] - wall_box[0]
        
        wall_index_box = np.array(([0,0,0], wall_shape)) // wall_grid.block_shape
        wall_index_shape = wall_index_box[1] - wall_index_box[0]

        location_ids = (logical_corners - wall_box[0]) // wall_grid.block_shape

        def _scan_order_indexes(coords, shape):
            # Example:
            #  coords = [(z,y,x,c),
            #            (z,y,x,c),
            #            ...]
            #
            #  shape = (Z,Y,X,C)
            #  strides = (Y*X*C, X*C, C, 1)
            #  indexes = [z*Y*X*C + y*X*C + x*C + c*1,
            #             z*Y*X*C + y*X*C + x*C + c*1,
            #             ...]
            
            # Accumulate products in reverse
            reverse_shape = tuple(shape[::-1]) # (C,X,Y,Z)
            reverse_strides = np.multiply.accumulate((1,) + reverse_shape[:-1]) # (1, C, C*X, C*X*Y)

            strides = reverse_strides[::-1]
            indexes = (coords * strides).sum(axis=1)
            return indexes.astype(np.int32)

        return _scan_order_indexes(location_ids, wall_index_shape)


