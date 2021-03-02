import copy
import logging
from itertools import chain

import numpy as np
import pandas as pd
import dask.bag as db

from neuclease.util import Timer, Grid, clipped_boxes_from_grid, round_box
from neuclease.dvid import fetch_instance_info, fetch_repo_instances, fetch_labelmap_voxels, post_labelmap_blocks, parse_labelarray_data

from dvid_resource_manager.client import ResourceManagerClient

from ..util import replace_default_entries
from ..volumes import ( VolumeService, VolumeServiceWriter, DvidSegmentationVolumeSchema,
                        TransposedVolumeService, LabelmappedVolumeService, ScaledVolumeService )
from . import Workflow

logger = logging.getLogger(__name__)

class LabelmapCopy(Workflow):
    """
    Workflow to copy compressed labelmap blocks directly from one repo to another.
    
    - No non-block-aligned copies
    - No coordinate changes
    - No block statistics computation
    - No downsampling computation
    - No sparse block support
    - No masks
    - No offsetting/remapping of IDs
    """

    OptionsSchema = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "min-scale": {
                "description": "Minimum pyramid scale to copy from input to output.",
                "type": "integer",
                "minValue": 0,
                "maxValue": 10,
                "default": 0
                },
            "max-scale": {
                "description": "Maximum pyramid scale to copy from input to output.  -1 means 'choose automatically.'",
                "type": "integer",
                "minValue": -1,
                "maxValue": 10,
                "default": -1
            },
            "slab-shape": {
                "description": "The data will be processed in 'slabs' of this shape. By default, the slabs comprise an entire Z-layer of bricks.",
                "type": "array",
                "items": {"type": "integer"},
                "default": [-1,-1,-1]
            },
            "record-label-sets": {
                "description": "Whether or not to record the list of all new labels contained in the copied blocks.\n"
                               "(For scale 0 only.)  Exported to CSV.\n",
                "type": "boolean",
                "default": True
            },
            "record-only": {
                "description": "If True, skip the copy operation entirely, and just read all the input data to record the set of labels in the dataset.\n",
                "type": "boolean",
                "default": False
            },
            "dont-overwrite-identical-blocks": {
                "description": "Before writing each block, read the existing segmentation from DVID\n"
                               "and check to see if it already matches what will be written.\n"
                               "If our write would be a no-op, don't write it.\n",
                "type": "boolean",
                "default": False
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidSegmentationVolumeSchema,
        "output": DvidSegmentationVolumeSchema,
        "labelmapcopy" : OptionsSchema
    })


    @classmethod
    def schema(cls):
        return LabelmapCopy.Schema


    def execute(self):
        self._init_services()
        options = self.config["labelmapcopy"]
        
        input_service = self.input_service
        output_service = self.output_service
        mgr_client = self.mgr_client

        record_labels = options["record-label-sets"]
        record_only = options["record-only"]
        check_existing = options["dont-overwrite-identical-blocks"]
        
        if record_only:
            assert options["min-scale"] == 0 and options["max-scale"] == 0, \
                ("In record-only mode, there is no reason to process any scales other than 0. "
                "Set min-scale and max-scale to 0.")
        
        def copy_box(box, scale):
            assert not record_only or scale == 0
            box = round_box(box, 64, 'out')
            box_shape = (box[1] - box[0])

            # Read input blocks
            with mgr_client.access_context(input_service.server, True, 1, np.prod(box_shape)):
                input_raw_blocks = fetch_labelmap_voxels(*input_service.instance_triple, box, scale,
                                                   False, input_service.supervoxels, format='raw-response')

            # If we're just recording, parse and return
            if scale == 0 and record_only:
                _input_spans, input_labels = parse_labelarray_data(input_raw_blocks, extract_labels=True)
                return list(set(chain(*input_labels.values())))

            # If not checking the output, just copy input to output
            if not check_existing:
                with mgr_client.access_context(output_service.server, False, 1, np.prod(box_shape)):
                    post_labelmap_blocks(*output_service.instance_triple, None, input_raw_blocks, scale,
                                         output_service.enable_downres, output_service.disable_indexing, False, is_raw=True)

                if scale == 0 and record_labels:
                    _input_spans, input_labels = parse_labelarray_data(input_raw_blocks, extract_labels=True)
                    return list(set(chain(*input_labels.values())))
                return []

            # Read from output
            with mgr_client.access_context(output_service.server, True, 1, np.prod(box_shape)):
                output_raw_blocks = fetch_labelmap_voxels(*output_service.instance_triple, box, scale,
                                                          False, output_service.supervoxels, format='raw-response')

            # If no differences, no need to parse
            if (input_raw_blocks == output_raw_blocks):
                return []

            input_spans = parse_labelarray_data(input_raw_blocks, extract_labels=False)
            output_spans = parse_labelarray_data(output_raw_blocks, extract_labels=False)
                
            # Compare block IDs
            input_ids = set(input_spans.keys())
            output_ids = set(output_spans.keys())
            
            missing_from_output = input_ids - output_ids
            missing_from_input = output_ids - input_ids
            common_ids = input_ids & output_ids
            
            for block_id in missing_from_input:
                # FIXME: We should pass this in the result so it can be logged in the client, not the worker.
                logger.error(f"Not overwriting block-id: {block_id}.  It doesn't exist in the input.")
            
            # Filter the input blocks so only the new/different ones remain
            filtered_input_list = []
            for block_id in missing_from_output:
                start, stop = input_spans[block_id]
                filtered_input_list.append( (block_id, input_raw_blocks[start:stop]) )

            filtered_output_list = []
            for block_id in common_ids:
                in_start, in_stop = input_spans[block_id]
                out_start, out_stop = output_spans[block_id]
                
                in_buf = input_raw_blocks[in_start:in_stop]
                out_buf = output_raw_blocks[out_start:out_stop]
                
                if in_buf != out_buf:
                    filtered_input_list.append( (block_id, in_buf) )
                    filtered_output_list.append( (block_id, out_buf) )
            
            # Sort filtered blocks so they appear in the same order in which we received them.
            filtered_input_list = sorted(filtered_input_list, key=lambda k_v: input_spans[k_v[0]][0])
            
            # Post them
            filtered_input_buf = b''.join([buf for (_, buf) in filtered_input_list])
            with mgr_client.access_context(output_service.server, False, 1, np.prod(box_shape)):
                post_labelmap_blocks(*output_service.instance_triple, None, filtered_input_buf, scale,
                                     output_service.enable_downres, output_service.disable_indexing, False, is_raw=True)
        
            if scale == 0 and record_labels:
                filtered_output_buf = b''.join([buf for (_, buf) in filtered_output_list])
                
                _, filtered_input_labels = parse_labelarray_data(filtered_input_buf, extract_labels=True)
                _, filtered_output_labels = parse_labelarray_data(filtered_output_buf, extract_labels=True)
                
                input_set = set(chain(*filtered_input_labels.values()))
                output_set = set(chain(*filtered_output_labels.values()))
                return list(input_set - output_set)

            return []

        all_labels = set()
        try:
            for scale in range(options["min-scale"], 1+options["max-scale"]):
                scaled_bounding_box = input_service.bounding_box_zyx // (2**scale)
                slab_boxes = clipped_boxes_from_grid(scaled_bounding_box, options["slab-shape"][::-1])
                logger.info(f"Scale {scale}: Copying {len(slab_boxes)} slabs")
                for slab_index, slab_box in enumerate(slab_boxes):
                    brick_boxes = clipped_boxes_from_grid(slab_box, Grid(self.input_service.preferred_message_shape) )
                    with Timer(f"Scale {scale} slab {slab_index}: Copying {slab_box[:,::-1].tolist()} ({len(brick_boxes)} bricks)", logger):
                        brick_labels = db.from_sequence(brick_boxes).map(lambda box: copy_box(box, scale)).compute()
                        slab_labels = chain(*brick_labels)
                        all_labels |= set(slab_labels)
        finally:
            if record_labels:
                name = 'sv' if input_service.supervoxels else 'body'
                pd.Series(sorted(all_labels), name=name).to_csv('recorded-labels.csv', index=False, header=True)


    def _init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        
        Also check the service configurations for errors.
        """
        input_config = self.config["input"]
        output_config = self.config["output"]
        mgr_options = self.config["resource-manager"]
        
        options = self.config["labelmapcopy"]
        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.mgr_client )
        assert input_config["dvid"]["supervoxels"], \
            'DVID input service config must use "supervoxels: true"'
        assert output_config["dvid"]["supervoxels"], \
            'DVID output service config must use "supervoxels: true"'

        input_service = self.input_service

        max_scale = options["max-scale"]
        if max_scale == -1:
            info = fetch_instance_info(*input_service.instance_triple)
            max_scale = int(info["Extended"]["MaxDownresLevel"])
            options["max-scale"] = max_scale

        assert not (set(range(1+max_scale)) - set(input_service.available_scales)), \
            "Your input config's 'available-scales' must include all levels you wish to copy."

        assert len(options["slab-shape"]) == 3
        slab_shape_zyx = np.array(options["slab-shape"][::-1])

        # FIXME: Should be a whole slab (per the docs above), not just the brick shape!
        replace_default_entries(slab_shape_zyx, input_service.preferred_message_shape)
        options["slab-shape"] = slab_shape_zyx[::-1].tolist()
        
        assert (slab_shape_zyx % input_service.preferred_message_shape[0] == 0).all(), \
            "slab-shape must be divisible by the brick shape"

        # Transposed/remapped services aren't supported because we're not going to inflate the downloaded blocks.
        assert all(not isinstance( svc, TransposedVolumeService ) for svc in input_service.service_chain)
        assert all(not isinstance( svc, LabelmappedVolumeService ) for svc in input_service.service_chain)

        assert not (input_service.bounding_box_zyx % input_service.block_width).any(), \
            "Input bounding-box should be a multiple of the block size in all dimensions."
        assert not (input_service.preferred_message_shape % input_service.block_width).any(), \
            "Input message-block-shape should be a multiple of the block size in all dimensions."

        assert all(not isinstance( svc, ScaledVolumeService ) or svc.scale_delta == 0 for svc in input_service.service_chain), \
            "For now, we don't support rescaled input, though it would be possible in theory."

        if options["record-only"]:
            # Don't need to check output setting if we're not writing
            self.output_service = None
            assert options["record-label-sets"], "If using 'record-only', you must set 'record-label-sets', too."
            assert not options["dont-overwrite-identical-blocks"], \
                "In record only mode, the output service can't be accessed, and you can't use dont-overwrite-identical-blocks"
            return

        if output_config["dvid"]["create-if-necessary"]:
            creation_depth = output_config["dvid"]["creation-settings"]["max-scale"]
            if creation_depth not in (-1, max_scale):
                msg = (f"Inconsistent max-scale options in the labelmapcopy config options ({max_scale}) and creation-settings options ({creation_depth}). "
                       "Omit max-scale from your creation-settings.")
                raise RuntimeError(msg)
            output_config["dvid"]["creation-settings"]["max-scale"] = max_scale

        # Replace 'auto' dimensions with input bounding box
        replace_default_entries(output_config["geometry"]["bounding-box"], input_service.bounding_box_zyx[:, ::-1])
        self.output_service = VolumeService.create_from_config( output_config, self.mgr_client )

        output_service = self.output_service
        assert isinstance( output_service, VolumeServiceWriter )

        if output_service.instance_name in fetch_repo_instances(output_service.server, output_service.uuid):
            info = fetch_instance_info(*output_service.instance_triple)
            existing_depth = int(info["Extended"]["MaxDownresLevel"])
            if max_scale not in (-1, existing_depth):
                raise Exception(f"Can't set pyramid-depth to {max_scale}: \n"
                                f"Data instance '{output_service.instance_name}' already existed, with depth {existing_depth}.\n"
                                f"For now, you are required to populate ALL scales of the output, or create a new output instance from scratch.")


        assert all(not isinstance( svc, TransposedVolumeService ) for svc in output_service.service_chain)
        assert all(not isinstance( svc, LabelmappedVolumeService ) for svc in output_service.service_chain)
        assert all(not isinstance( svc, ScaledVolumeService ) or svc.scale_delta == 0 for svc in output_service.service_chain)

        # Output can't be a scaled service because we copied some geometry (bounding-box)
        # directly from the input service.
        assert not isinstance( output_service, ScaledVolumeService ) or output_service.scale_delta == 0

        assert output_service.base_service.disable_indexing, \
            "During ingestion, indexing should be disabled.\n" \
            "Please add 'disable-indexing':true to your output dvid config."

        logger.info(f"Output bounding box (xyz) is: {output_service.bounding_box_zyx[:,::-1].tolist()}")

        assert (input_service.bounding_box_zyx == output_service.bounding_box_zyx).all(), \
            "Input and output service bounding boxes must match exactly."
        assert input_service.block_width == output_service.block_width, \
            "Input and output must use the same block-width"
        assert not (output_service.bounding_box_zyx % output_service.block_width).any(), \
            "Output bounding-box should be a multiple of the block size in all dimensions."
        assert not (output_service.preferred_message_shape % output_service.block_width).any(), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."
        
