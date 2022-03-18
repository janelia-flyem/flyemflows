import os
import copy
import time
import json
import pickle
import logging
from functools import partial

import h5py
import numpy as np
import pandas as pd
from confiddler import flow_style

from neuclease.util import (Timer, Grid, boxes_from_grid, block_stats_for_volume, BLOCK_STATS_DTYPES,
                            mask_for_labels, box_intersection, extract_subvol, SparseBlockMask)
from neuclease.dvid import fetch_repo_instances, fetch_instance_info
from neuclease.dvid.rle import runlength_encode_to_ranges

from dvid_resource_manager.client import ResourceManagerClient

from ..util import replace_default_entries, COMPRESSION_METHODS, DOWNSAMPLE_METHODS
from ..brick import BrickWall
from ..volumes import ( VolumeService, VolumeServiceWriter, SegmentationVolumeSchema, TransposedVolumeService, ScaledVolumeService, DvidVolumeService )

from . import Workflow
from .util.config_helpers import BodyListSchema, load_body_list

logger = logging.getLogger(__name__)

SEGMENT_STATS_COLUMNS = ['segment', 'voxel_count', 'bounding_box_start', 'bounding_box_stop'] #, 'block_list']

class CopySegmentation(Workflow):
    """
    Workflow to copy segmentation from one source (e.g. a DVID segmentation
    instance or a BrainMaps volume) into a DVID segmentation instance.

    Notes:

    - The data is written to DVID in block-aligned 'bricks'.
      If the source data is not block-aligned at the edges,
      pre-existing data (if any) is read from the DVID destination
      to fill out ('pad') the bricks until they are completely block aligned.

    - The data is also downsampled into a multi-scale pyramid and uploaded.

    - The volume is processed in Z-slabs. To avoid complications during downsampling,
      the Z-slabs must be aligned to a multiple of the DVID block shape, which may be
      rather large, depending on the highest scale of the pyramid.
      (It is recommended that you don't set this explicitly in the config, so a
      suitable default can be chosen for you.)

    - This workflow uses DvidVolumeService to write the segmentation blocks,
      which is able to send them to DVID in the pre-encoded 'labelarray' or 'labelmap' block format.
      This saves CPU resources on the DVID server.

    - As a convenience, size of each label 'body' in the copied volume is also
      calculated and exported in an HDF5 file, sorted by body size.
    """

    OptionsSchema = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "block-statistics-file": {
                "description": "Where to store block statistics for the INPUT segmentation\n"
                               "(but translated to output coordinates).\n"
                               "If the file already exists, it will be appended to (for restarting from a failed job).\n"
                               "Supported formats: .csv and .h5",
                "type": "string",
                "default": "block-statistics.h5"
            },
            "compute-block-statistics": {
                "description": "Whether or not to compute block statistics (from the scale 0 data).\n"
                               "Usually you'll need the statistics file to load labelindexes after copying the voxels,\n"
                               "but in some cases you might not need them (e.g. adding pyramids after ingesting only scale 0).\n"
                               "By default, the block shape will be chosen according to the output volume,\n"
                               "but you can provide a custom shape here.\n",
                "oneOf": [
                    {
                        "type": "boolean"
                    },
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 3,
                        "default": flow_style([-1,-1,-1])
                    }
                ],
                "default": True
            },
            "pyramid-depth": {
                "description": "Number of pyramid levels to generate \n"
                               "(-1 means choose automatically, 0 means no pyramid)",
                "type": "integer",
                "default": -1 # automatic by default
            },
            "permit-inconsistent-pyramid": {
                "description": "Normally overwriting a pre-existing data instance is\n"
                               "an error unless you rewrite ALL of its pyramid levels,\n"
                               "but this setting allows you to override that error.\n"
                               "(You had better know what you're doing...)\n",
                "type": "boolean",
                "default": False
            },
            "skip-scale-0-write": {
                "description": "Skip writing scale 0.  Useful if scale 0 is already downloaded and now\n"
                               "you just want to generate the rest of the pyramid to the same instance.\n",
                "type": "boolean",
                "default": False
            },
            "download-pre-downsampled": {
                "description": "Instead of downsampling the data, just download the pyramid from the server (if it's available).\n"
                               "Will not work unless you add the 'available-scales' setting to the input service's geometry config.",
                "type": "boolean",
                "default": False
            },
            "downsample-method": {
                "description": "Which downsampling method to use for label volume downsampling.\n",
                "type": "string",
                "enum": DOWNSAMPLE_METHODS,
                # FIXME: This not the fastest method, but the fastest method was
                #        observed to segfault in one conda environment.
                #        Need to investigate!
                "default": "labels-numba"
            },
            "brick-compression": {
                "description": "Internally, downloaded bricks will be stored in a compressed format.\n"
                               "This setting specifies the compression scheme to use.\n"
                               f"Options: {COMPRESSION_METHODS}"
                               "Note: This affects only in-memory storage while the workflow is running.\n"
                               "      It does NOT affect the compression used in DVID.\n",
                "type": "string",
                "enum": COMPRESSION_METHODS,
                "default": "lz4_2x"
            },
            "write-empty-blocks": {
                "description": "If a copied block would be completely empty, it can be skipped\n"
                               "if you're writing to a brand new volume.\n"
                               "By default, we don't bother writing such blocks.\n"
                               "Set this to True if you want to overwrite existing blocks with empty ones.",
                "type": "boolean",
                "default": False,
            },
            "dont-overwrite-identical-blocks": {
                "description": "Before writing each block, read the existing segmentation from DVID\n"
                               "and check to see if it already matches what will be written.\n"
                               "If our write would be a no-op, don't write it.\n",
                "type": "boolean",
                "default": False
            },
            "slab-shape": {
                "description": "The data is downloaded and processed in big slabs.\n"
                               "This setting determines how thick each Z-slab is.\n"
                               "If you make it a multiple of the message-block-shape, \n"
                               "then slabs will be completely independent, even after downsampling.\n"
                               "(But that's not a requirement.)\n",
                "type": "array",
                "items": {
                    "type": "integer",
                    "minItems": 3,
                    "maxItems": 3,
                },
                "default": flow_style([-1,-1,-1])  # Choose automatically: full XY plane, with Z = block_width * 2**pyramid_depth
            },
            "slab-depth": {
                "description": "Deprecated. Do not use.",
                "type": "integer",
                "default": -1
            },
            "delay-minutes-between-slabs": {
                "description": "Optionally introduce an artificial pause after finishing one slab before starting the next,\n"
                               "to give DVID time to index the blocks we've sent so far.\n"
                               "Should not be necessary for most use-cases.",
                "type": "integer",
                "default": 0,
            },
            "sparse-block-mask": {
                "description": "Optionally provide a mask which limits the set of bricks to be processed.\n"
                               "If you already have a map of where the valid data is, you can provide a\n"
                               "pickled SparseBlockMask here.\n",
                "type": "string",
                "default": ""
            },
            "input-mask-labels": {
                **BodyListSchema,
                "description": "If provided, only voxels under the given input labels in the output will be modified.\n"
                               "Others will remain untouched.\n",
            },
            "output-mask-labels": {
                **BodyListSchema,
                "description": "If provided, only voxels under the given labels in the output will be modified.\n"
                               "Others will remain untouched.\n"
                               "Note: At the time of this writing, the output mask is NOT used to enable sparse-fetching from DVID.\n"
                               "      Only the input mask is used for that, so if you're using an output mask without an input mask,\n"
                               "      you'll still fetch the entire input volume, even if most of it will be written unchanged!\n",
            },
            "skip-masking-step": {
                "description": "When using an input mask, normally the entire output block must be fetched so it can be combined with the input.\n"
                               "but if you know you're writing to an empty volume, or if the output happens to match the input\n"
                               "(e.g. if you are recomputing pyramids from an existing scale-0 segmentation),\n"
                               "then you may save time by skipping the fetch from the output.\n"
                               "In this case, input-mask-labels are used to determine which blocks to copy,\n"
                               "but not which voxels to copy -- all voxels in each block are directly written to the output.\n"
                               "Note: The input will still be PADDED from the output if necessary to achieve block alignment.\n",
                "type": "boolean",
                "default": False
            },
            "add-offset-to-ids": {
                "description": "If desired, add a constant offset to all input IDs before they are written to the output.",
                "type": "integer",
                "default": 0
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "output": SegmentationVolumeSchema,
        "copysegmentation" : OptionsSchema
    })

    @classmethod
    def schema(cls):
        return CopySegmentation.Schema


    def execute(self):
        self._init_services()
        self._init_masks()
        self._log_neuroglancer_links()
        self._sanitize_config()

        # Aim for 2 GB RDD partitions when loading segmentation
        GB = 2**30
        self.target_partition_size_voxels = 2 * GB // np.uint64().nbytes

        # (See note in _init_services() regarding output bounding boxes)
        input_bb_zyx = self.input_service.bounding_box_zyx
        output_bb_zyx = self.output_service.bounding_box_zyx
        self.translation_offset_zyx = output_bb_zyx[0] - input_bb_zyx[0]
        if self.translation_offset_zyx.any():
            logger.info(f"Translation offset is {self.translation_offset_zyx[:, ::-1].tolist()}")

        pyramid_depth = self.config["copysegmentation"]["pyramid-depth"]
        slab_shape = self.config["copysegmentation"]["slab-shape"][::-1]

        # Process data in Z-slabs
        # FIXME:
        #   If the SBM is smaller than the overall bounding box,
        #   then we should shrink the overall bounding box accordingly.
        #   Right now, the empty slabs are processed anyway, leading to strange
        #   errors when we attempt to extract non-existent parts of the SBM.

        output_slab_boxes = boxes_from_grid(output_bb_zyx, slab_shape, clipped=True)
        max_depth = max(map(lambda box: box[1][0] - box[0][0], output_slab_boxes))
        logger.info(f"Processing data in {len(output_slab_boxes)} slabs (mega-bricks) for {pyramid_depth} pyramid levels")

        if self.config["copysegmentation"]["compute-block-statistics"]:
            self._init_stats_file()

        # Read data and accumulate statistics, one slab at a time.
        for slab_index, output_slab_box in enumerate( output_slab_boxes ):
            with Timer() as timer:
                self._process_slab(slab_index, output_slab_box )
            logger.info(f"Slab {slab_index}: Total processing time: {timer.timedelta}")

            delay_minutes = self.config["copysegmentation"]["delay-minutes-between-slabs"]
            if delay_minutes > 0 and slab_index != len(output_slab_boxes)-1:
                logger.info(f"Delaying {delay_minutes} before continuing to next slab...")
                time.sleep(delay_minutes * 60)

        logger.info(f"DONE copying/downsampling all slabs")


    def _init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.

        Also check the service configurations for errors.
        """
        input_config = self.config["input"]
        output_config = self.config["output"]
        mgr_options = self.config["resource-manager"]

        options = self.config["copysegmentation"]
        slab_shape = options["slab-shape"][::-1]
        pyramid_depth = options["pyramid-depth"]
        permit_inconsistent_pyramids = options["permit-inconsistent-pyramid"]

        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.mgr_client )

        brick_shape = self.input_service.preferred_message_shape
        if (slab_shape % brick_shape).any():
            logger.warning(f"Your slab-shape {slab_shape} is not a multiple of the input's brick shape {brick_shape}")

        if isinstance(self.input_service.base_service, DvidVolumeService):
            assert input_config["dvid"]["supervoxels"], \
                'DVID input service config must use "supervoxels: true"'

        # Special handling for creation of multi-scale outputs:
        # auto-configure the pyramid depths
        multiscale_output_type = None
        for t in ["dvid", "n5", "zarr"]:
            if t in output_config and not hasattr(output_config[t], 'from_default'):
                multiscale_output_type = t
        if multiscale_output_type:
            out_fmt = multiscale_output_type
            if output_config[out_fmt]["create-if-necessary"]:
                if self.config["copysegmentation"]["skip-scale-0-write"] and pyramid_depth == 0:
                    # Nothing to write.  Maybe the user is just computing block statistics.
                    msg = ("Since your config specifies no pyramid levels to write, no output instance will be created. "
                           "Avoid this warning by removing 'create-if-necessary' from your config")
                    logger.warning(msg)
                    output_config[out_fmt]["create-if-necessary"] = False
                else:
                    max_scale = output_config[out_fmt]["creation-settings"]["max-scale"]
                    if max_scale not in (-1, pyramid_depth):
                        msg = (f"Inconsistent max-scale ({max_scale}) and pyramid-depth ({pyramid_depth}). "
                               "Omit max-scale from your creation-settings.")
                        raise RuntimeError(msg)
                    output_config[out_fmt]["creation-settings"]["max-scale"] = pyramid_depth

        # Replace 'auto' dimensions with input bounding box
        replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])
        self.output_service = VolumeService.create_from_config( output_config, self.mgr_client )
        output_service = self.output_service
        assert isinstance( output_service, VolumeServiceWriter )

        if "dvid" in output_config:
            assert output_config["dvid"]["supervoxels"], \
                'DVID output service config must use "supervoxels: true"'

            if output_service.instance_name in fetch_repo_instances(output_service.server, output_service.uuid):
                existing_depth = self._read_pyramid_depth()
                if pyramid_depth not in (-1, existing_depth) and not permit_inconsistent_pyramids:
                    raise Exception(f"Can't set pyramid-depth to {pyramid_depth}: "
                                    f"Data instance '{output_service.instance_name}' already existed, with depth {existing_depth}")

        # These services aren't supported because we copied some geometry (bounding-box)
        # directly from the input service.
        assert not isinstance( output_service, TransposedVolumeService )
        assert not isinstance( output_service, ScaledVolumeService ) or output_service.scale_delta == 0

        if isinstance(self.output_service.base_service, DvidVolumeService):
            assert output_service.base_service.disable_indexing, \
                "During ingestion, dvid labelmap indexing should be disabled.\n" \
                "Please add 'disable-indexing: true' to your output dvid config."

        logger.info(f"Output bounding box (xyz) is: {output_service.bounding_box_zyx[:,::-1].tolist()}")

        input_shape = -np.subtract(*self.input_service.bounding_box_zyx)
        output_shape = -np.subtract(*output_service.bounding_box_zyx)

        assert not any(np.array(output_service.preferred_message_shape) % output_service.block_width), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."
        assert (input_shape == output_shape).all(), \
            "Input bounding box and output bounding box do not have the same dimensions"

        if ("apply-labelmap" in output_config["adapters"]) and (output_config["adapters"]["apply-labelmap"]["file-type"] != "__invalid__"):
            assert output_config["adapters"]["apply-labelmap"]["apply-when"] == "reading-and-writing", \
                "Labelmap will be applied to voxels during pre-write and post-read (due to block padding).\n"\
                "You cannot use this workflow with non-idempotent labelmaps, unless your data is already perfectly block aligned."


    def _init_masks(self):
        options = self.config["copysegmentation"]
        self.sbm = None

        if options["sparse-block-mask"]:
            # In theory, we could just take the intersection of the masks involved.
            # But I'm too lazy to think about that right now.
            assert not options["input-mask-labels"] and not options["output-mask-labels"], \
                "Not Implemented: Can't use sparse-block-mask in conjunction with input-mask-labels or output-mask-labels"

            with open(options["sparse-block-mask"], 'rb') as f:
                self.sbm = pickle.load(f)

        is_supervoxels = False
        if isinstance(self.input_service.base_service, DvidVolumeService):
            is_supervoxels = self.input_service.base_service.supervoxels

        output_mask_labels = load_body_list(options["output-mask-labels"], is_supervoxels)
        self.output_mask_labels = set(output_mask_labels)

        output_sbm = None
        if len(output_mask_labels) > 0:
            if (self.output_service.preferred_message_shape != self.input_service.preferred_message_shape).any():
                logger.warn("Not using output mask to reduce data fetching: Your input service and output service don't have the same brick shape")
            elif (self.output_service.bounding_box_zyx != self.input_service.bounding_box_zyx).any():
                logger.warn("Not using output mask to reduce data fetching: Your input service and output service don't have the same bounding box")
            else:
                try:
                    output_sbm = self.output_service.sparse_block_mask_for_labels(output_mask_labels)
                except NotImplementedError:
                    output_sbm = None

        input_mask_labels = load_body_list(options["input-mask-labels"], is_supervoxels)

        input_sbm = None
        if len(input_mask_labels) > 0:
            try:
                input_sbm = self.input_service.sparse_block_mask_for_labels(input_mask_labels)
            except NotImplementedError:
                input_sbm = None

        if self.sbm is not None:
            pass
        elif input_sbm is None:
            self.sbm = output_sbm
        elif output_sbm is None:
            self.sbm = input_sbm
        else:
            assert (input_sbm.resolution == output_sbm.resolution).all(), \
                "FIXME: At the moment, you can't supply both an input mask and an output "\
                "mask unless the input and output sources use the same brick shape (message-block-shape)"

            final_box = box_intersection(input_sbm.box, output_sbm.box)

            input_box = (input_sbm.box - final_box) // input_sbm.resolution
            input_mask = extract_subvol(input_sbm.lowres_mask, input_box)

            output_box = (output_sbm - final_box) // output_sbm.resolution
            output_mask = extract_subvol(output_sbm.lowres_mask, output_box)

            assert input_mask.shape == output_mask.shape
            assert input_mask.dtype == output_mask.dtype == np.bool
            final_mask = (input_mask & output_mask)

            self.sbm = SparseBlockMask(final_mask, final_box, input_sbm.resolution)

        id_offset = options["add-offset-to-ids"]
        if id_offset != 0:
            id_offset = options["add-offset-to-ids"]
            input_mask_labels = np.asarray(input_mask_labels, np.uint64)
            input_mask_labels += id_offset
        self.input_mask_labels = set(input_mask_labels)


    def _read_pyramid_depth(self):
        """
        Read the MaxDownresLevel from the output instance we'll be writing to,
        and verify that it matches our config for pyramid-depth.
        """
        info = fetch_instance_info(*self.output_service.instance_triple)
        existing_depth = int(info["Extended"]["MaxDownresLevel"])
        return existing_depth


    def _log_neuroglancer_links(self):
        """
        Write a link to the log file for viewing the segmentation data after it is ingested.
        We assume that the output server is hosting neuroglancer at http://<server>:<port>/neuroglancer/
        """
        if not isinstance(self.output_service.base_service, DvidVolumeService):
            return

        output_service = self.output_service
        server = output_service.base_service.server
        uuid = output_service.base_service.uuid
        instance = output_service.base_service.instance_name

        output_box_xyz = np.array(output_service.bounding_box_zyx[:, :-1])
        output_center_xyz = (output_box_xyz[0] + output_box_xyz[1]) / 2

        link_prefix = f"{server}/neuroglancer/#!"
        link_json = \
        {
            "layers": {
                "segmentation": {
                    "type": "segmentation",
                    "source": f"dvid://{server}/{uuid}/{instance}"
                }
            },
            "navigation": {
                "pose": {
                    "position": {
                        "voxelSize": [8,8,8],
                        "voxelCoordinates": output_center_xyz.tolist()
                    }
                },
                "zoomFactor": 8
            }
        }
        logger.info(f"Neuroglancer link to output: {link_prefix}{json.dumps(link_json)}")


    def _sanitize_config(self):
        """
        Replace a few config values with reasonable defaults if necessary.
        (Note: Must be called AFTER services and output instances have been initialized.)
        """
        options = self.config["copysegmentation"]

        # Overwrite pyramid depth in our config (in case the user specified -1, i.e. automatic)
        if options["pyramid-depth"] == -1:
            options["pyramid-depth"] = self._read_pyramid_depth()
        pyramid_depth = options["pyramid-depth"]

        # Default slab shape is a full XY slice and a single block in Z
        block_width = self.output_service.block_width
        default_slab_shape = self.input_service.bounding_box_zyx[1] - self.input_service.bounding_box_zyx[0]
        default_slab_shape[0] = block_width * 2**pyramid_depth

        if options["slab-depth"] != -1:
            raise RuntimeError("Config problem: 'slab-depth' is no longer a valid setting.  Please use 'slab-shape' instead.")

        slab_shape = options["slab-shape"][::-1]
        replace_default_entries(slab_shape, default_slab_shape)
        options["slab-shape"] = slab_shape[::-1]

        if (options["download-pre-downsampled"] and (options["input-mask-labels"] or options["output-mask-labels"])):
            # TODO: This restriction could be lifted if we also used the mask when fetching
            #       the downscale pyramids, but that's not yet implemented.  Even if you're
            #       using 'skip-masking-step', the lowres pyramids are a problem.
            raise RuntimeError("You aren't allow to use download-pre-downsampled if you're using a mask.")

        if options["skip-scale-0-write"] and pyramid_depth == 0 and not options["compute-block-statistics"]:
            raise RuntimeError("According to your config, you aren't computing block stats, "
                               "you aren't writing scale 0, and you aren't writing pyramids.  "
                               "What exactly are you hoping will happen here?")

        if options["skip-masking-step"] and options["output-mask-labels"]:
            logger.warning("You specified output-mask-labels but also skip-masking-step. That's usually a mistake!")

    def _init_stats_file(self):
        stats_path = self.config["copysegmentation"]["block-statistics-file"]
        if os.path.exists(stats_path):
            logger.info(f"Block statistics already exists: {stats_path}")
            logger.info(f"Will APPEND to the pre-existing statistics file.")
            return

        if stats_path.endswith('.csv'):
            # Initialize (just the header)
            template_df = pd.DataFrame(columns=list(BLOCK_STATS_DTYPES.keys()))
            template_df.to_csv(stats_path, index=False, header=True)

        elif stats_path.endswith('.h5'):
            # Initialize a 0-entry 1D array with the correct (structured) dtype
            with h5py.File(stats_path, 'w') as f:
                f.create_dataset('stats', shape=(0,), maxshape=(None,), chunks=True, dtype=list(BLOCK_STATS_DTYPES.items()))
        else:
            raise RuntimeError(f"Unknown file format: {stats_path}")


    def _append_slab_statistics(self, slab_stats_df):
        """
        Append the rows of the given slab statistics DataFrame to the output statistics file.
        No attempt is made to drop duplicate rows
        (e.g. if you started from pre-existing statistics and the new
        bounding-box overlaps with the previous run's).

        Args:
            slab_stats_df: DataFrame to be appended to the stats file,
                           with columns and dtypes matching BLOCK_STATS_DTYPES
        """
        assert list(slab_stats_df.columns) == list(BLOCK_STATS_DTYPES.keys())
        stats_path = self.config["copysegmentation"]["block-statistics-file"]

        if stats_path.endswith('.csv'):
            slab_stats_df.to_csv(stats_path, header=False, index=False, mode='a')

        elif stats_path.endswith('.h5'):
            with h5py.File(stats_path, 'a') as f:
                orig_len = len(f['stats'])
                new_len = orig_len + len(slab_stats_df)
                f['stats'].resize((new_len,))
                f['stats'][orig_len:new_len] = slab_stats_df.to_records()
        else:
            raise RuntimeError(f"Unknown file format: {stats_path}")


    def _process_slab(self, slab_index, output_slab_box ):
        """
        (The main work of this file.)

        Process a large slab of voxels:

        1. Read a 'slab' of bricks from the input as a BrickWall
        2. Translate it to the output coordinates.
        3. Splice & group the bricks so that they are aligned to the optimal output grid
        4. 'Pad' the bricks on the edges of the wall by *reading* data from the output destination,
            so that all bricks are complete (i.e. they completely fill their grid block).
        5. Write all bricks to the output destination.
        6. Downsample the bricks and repeat steps 3-5 for the downsampled scale.
        """
        options = self.config["copysegmentation"]
        pyramid_depth = options["pyramid-depth"]

        input_slab_box = output_slab_box - self.translation_offset_zyx
        if self.sbm is None:
            slab_sbm = None
        else:
            slab_sbm = SparseBlockMask.create_from_sbm_box(self.sbm, input_slab_box)

        try:
            input_wall = BrickWall.from_volume_service( self.input_service,
                                                        0,
                                                        input_slab_box,
                                                        self.client,
                                                        self.target_partition_size_voxels,
                                                        sparse_block_mask=slab_sbm,
                                                        compression=options['brick-compression'] )

            if input_wall.num_bricks == 0:
                logger.info(f"Slab: {slab_index}: No bricks to process.  Skipping.")
                return

        except RuntimeError as ex:
            if "SparseBlockMask selects no blocks" in str(ex):
                return

        input_wall.persist_and_execute(f"Slab {slab_index}: Reading ({input_slab_box[:,::-1].tolist()})", logger)

        # Translate coordinates from input to output
        # (which will leave the bricks in a new, offset grid)
        # This has no effect on the brick volumes themselves.
        if any(self.translation_offset_zyx):
            input_wall = input_wall.translate(self.translation_offset_zyx)

        id_offset = options["add-offset-to-ids"]
        if id_offset != 0:
            def add_offset(brick):
                # Offset everything except for label 0, which remains 0
                vol = brick.volume.copy()
                brick.compress()
                vol[vol != 0] += id_offset
                return vol
            input_wall = input_wall.map_brick_volumes(add_offset)

        output_service = self.output_service

        # Pad internally to block-align to the OUTPUT alignment.
        # Here, we assume that any output labelmap (if any) is idempotent,
        # so it's okay to read pre-existing output data that will ultimately get remapped.
        padded_wall = self._consolidate_and_pad(slab_index, input_wall, 0, output_service)

        # Write scale 0 to DVID
        if not options["skip-scale-0-write"]:
            self._write_bricks( slab_index, padded_wall, 0, output_service )

        if options["compute-block-statistics"]:
            with Timer(f"Slab {slab_index}: Computing slab block statistics", logger):
                if options["compute-block-statistics"] is True:
                    block_shape = 3*[self.output_service.base_service.block_width]
                else:
                    block_shape = options["compute-block-statistics"]

                def block_stats_for_brick(brick):
                    vol = brick.volume
                    brick.compress()
                    return block_stats_for_volume(block_shape, vol, brick.physical_box)

                # We compute stats now (while the scale-0 data is available),
                # but don't append them to the file until after the slab successfully computes, below.
                slab_block_stats_per_brick = padded_wall.bricks.map(block_stats_for_brick).compute()
                slab_block_stats_df = pd.concat(slab_block_stats_per_brick, ignore_index=True)
                del slab_block_stats_per_brick

        for new_scale in range(1, 1+pyramid_depth):
            if options["download-pre-downsampled"] and new_scale in self.input_service.available_scales:
                del padded_wall
                downsampled_wall = BrickWall.from_volume_service(self.input_service,
                                                                 new_scale,
                                                                 input_slab_box,
                                                                 self.client,
                                                                 self.target_partition_size_voxels,
                                                                 compression=options["brick-compression"])
                downsampled_wall.persist_and_execute(f"Slab {slab_index}: Scale {new_scale}: Downloading pre-downsampled bricks", logger)
            else:
                # Compute downsampled (results in smaller bricks)
                downsampled_wall = padded_wall.downsample( (2,2,2), method=options["downsample-method"] )
                downsampled_wall.persist_and_execute(f"Slab {slab_index}: Scale {new_scale}: Downsampling", logger)
                del padded_wall

            # Consolidate to full-size bricks and pad internally to block-align
            consolidated_wall = self._consolidate_and_pad(slab_index, downsampled_wall, new_scale, output_service)
            del downsampled_wall

            # Write to DVID
            self._write_bricks( slab_index, consolidated_wall, new_scale, output_service )

            padded_wall = consolidated_wall
            del consolidated_wall
        del padded_wall

        # Now we append to the stats file, after successfully writing all scales.
        # This saves us from writing duplicate stats in the event that we have to kill the job and resume it.
        if options["compute-block-statistics"]:
            with Timer(f"Slab {slab_index}: Appending to stats file"):
                self._append_slab_statistics( slab_block_stats_df )


    def _consolidate_and_pad(self, slab_index, input_wall, scale, output_service):
        """
        Consolidate (align), and pad the given BrickWall

        Args:
            scale: The pyramid scale of the data.

            output_service: The output_service to align to and pad from

        Returns a pre-executed and persisted BrickWall.
        """
        options = self.config["copysegmentation"]

        # We'll pad from previously-existing pyramid data until
        # we have full storage blocks, e.g. (64,64,64),
        # but not necessarily full bricks, e.g. (64,64,6400)
        output_writing_grid = Grid(output_service.preferred_message_shape)
        storage_block_width = output_service.block_width
        output_padding_grid = Grid( (storage_block_width, storage_block_width, storage_block_width), output_writing_grid.offset )
        output_accessor_func = partial(output_service.get_subvolume, scale=scale)

        with Timer(f"Slab {slab_index}: Scale {scale}: Shuffling bricks into alignment", logger):
            # Consolidate bricks to full-size, aligned blocks (shuffles data)
            realigned_wall = input_wall.realign_to_new_grid(output_writing_grid,  output_accessor_func)
            del input_wall
            realigned_wall.persist_and_execute()

        input_mask_labels = self.input_mask_labels
        output_mask_labels = self.output_mask_labels

        # If no masks are involved, we merely need to pad the existing data on the edges.
        # (No need to fetch the entire output.)
        # Similarly, if scale > 0, then the masks were already applied and the input/output data was
        # already combined, we can simply write the (padded) downsampled data.
        if scale == 0 and (input_mask_labels or output_mask_labels) and not options["skip-masking-step"]:
            # If masks are involved, we must fetch the ALL the output
            # (unless skip-masking-step was given),
            # and select data from input or output according to the masks.
            output_service = self.output_service
            translation_offset_zyx = self.translation_offset_zyx
            def combine_with_output(input_brick):
                output_box = input_brick.physical_box + translation_offset_zyx
                output_vol = output_service.get_subvolume(output_box, scale=0)
                output_vol = np.asarray(output_vol, order='C')

                mask = None
                if input_mask_labels:
                    mask = mask_for_labels(input_brick.volume, input_mask_labels)

                if output_mask_labels:
                    output_mask = mask_for_labels(output_vol, output_mask_labels)

                    if mask is None:
                        mask = output_mask
                    else:
                        mask[:] &= output_mask

                # Start with the complete output, then
                # change voxels that fall within both masks.
                output_vol[mask] = input_brick.volume[mask]
                input_brick.compress()
                return output_vol

            combined_wall = realigned_wall.map_brick_volumes(combine_with_output)
            combined_wall.persist_and_execute(f"Slab {slab_index}: Scale {scale}: Combining masked bricks", logger)
            realigned_wall = combined_wall


        padded_wall = realigned_wall.fill_missing(output_accessor_func, output_padding_grid)
        del realigned_wall
        padded_wall.persist_and_execute(f"Slab {slab_index}: Scale {scale}: Padding", logger)
        return padded_wall


    def _write_bricks(self, slab_index, brick_wall, scale, output_service):
        """
        Writes partition to specified dvid.
        """
        block_width = output_service.block_width
        EMPTY_VOXEL = 0
        dont_overwrite_identical_blocks = self.config["copysegmentation"]["dont-overwrite-identical-blocks"]
        write_empty_blocks = self.config["copysegmentation"]["write-empty-blocks"]

        def write_brick(brick):
            logger = logging.getLogger(__name__)

            assert (brick.physical_box % block_width == 0).all(), \
                f"This function assumes each brick's physical data is already block-aligned: {brick}"

            if dont_overwrite_identical_blocks:
                try:
                    existing_stored_brick = output_service.get_subvolume(brick.physical_box, scale)
                except:
                    logger.error(f"Error reading brick: {brick.physical_box.tolist()}, scale={scale}")
                    raise

            x_size = brick.volume.shape[2]
            # Find all non-zero blocks (and record by block index)
            block_coords = []
            for block_index, block_x in enumerate(range(0, x_size, block_width)):
                new_block = brick.volume[:, :, block_x:block_x+block_width]

                # By default, write this block if it is non-empty
                write_block = write_empty_blocks or (new_block != EMPTY_VOXEL).any()

                # If dont-overwrite-identical-blocks is enabled,
                # write the block if it DIFFERS from the block that was already stored in DVID.
                # (Regardless of whether or not either block is empty.)
                if dont_overwrite_identical_blocks:
                    old_block = existing_stored_brick[:, :, block_x:block_x+block_width]
                    difference_map = (new_block != old_block)
                    write_block = difference_map.any()
                    if write_block:
                        block_coord_zyx = brick.physical_box[0] + [0, 0, block_x]
                        block_coord_xyz = block_coord_zyx[::-1].tolist()
                        changed_voxel_list_new = np.unique(new_block[difference_map]).tolist()
                        changed_voxel_list_old = np.unique(old_block[difference_map]).tolist()
                        msg = (f"Slab {slab_index}: Scale {scale}: Overwriting block: "
                               '{ '
                                    f'"block-coord-xyz": {block_coord_xyz}, '
                                    f'"difference-voxel-count": {difference_map.sum()}, '
                                    f'"new-ids": {changed_voxel_list_new}, '
                                    f'"old-ids": {changed_voxel_list_old} '
                               ' }')
                        logger.info(msg)

                if write_block:
                    block_coords.append( (0, 0, block_index) ) # (Don't care about Z,Y indexes, just X-index)

            # Find *runs* of non-zero blocks
            block_coords = np.asarray(block_coords, dtype=np.int32)
            block_runs = runlength_encode_to_ranges(block_coords, True) # returns [[Z,Y,X1,X2], [Z,Y,X1,X2], ...]

            # Convert stop indexes from inclusive to exclusive
            block_runs[:,-1] += 1

            # Discard Z,Y indexes and convert from indexes to pixels
            ranges = block_width * block_runs[:, 2:4]

            # iterate through contiguous blocks and write to DVID
            for (data_x_start, data_x_end) in ranges:
                datacrop = brick.volume[:,:,data_x_start:data_x_end].copy()
                data_offset_zyx = brick.physical_box[0] + (0,0,data_x_start)

                with Timer() as _put_timer:
                    try:
                        output_service.write_subvolume(datacrop, data_offset_zyx, scale)
                    except:
                        logger.error(f"Error writing brick at {brick.physical_box.tolist()}, scale={scale}, offset={data_offset_zyx}")
                        raise

                # Note: This timing data doesn't reflect ideal throughput, since throttle
                #       and/or the resource manager muddy the numbers a bit...
                #megavoxels_per_second = datacrop.size / 1e6 / put_timer.seconds
                #logger.info(f"Put block {data_offset_zyx} in {put_timer.seconds:.3f} seconds ({megavoxels_per_second:.1f} Megavoxels/second)")

            brick.compress()

        msg = f"Slab {slab_index}: Scale {scale}: Writing bricks"
        if isinstance(output_service.base_service, DvidVolumeService):
            instance_name = output_service.base_service.instance_name
            msg += f" to {instance_name}"

        with Timer(msg, logger):
            brick_wall.bricks.map(write_brick).compute()
