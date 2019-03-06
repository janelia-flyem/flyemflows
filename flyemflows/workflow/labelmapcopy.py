import copy
import logging
from itertools import chain

import numpy as np
import pandas as pd
import dask.bag as db

from neuclease.util import Timer, Grid, clipped_boxes_from_grid
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
                "description": "Whether or not to record the list of all labels contained in the copied blocks.\n"
                               "(For scale 0 only.)  Exported to CSV.\n",
                "type": "boolean",
                "default": True
            },
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
        
        def copy_box(box, scale):
            box_shape = (box[1] - box[0])
            with mgr_client.access_context(input_service.server, True, 1, np.prod(box_shape)):
                raw_blocks = fetch_labelmap_voxels(*input_service.instance_triple, box, scale,
                                                   False, input_service.supervoxels, format='raw-response')

            labels = []
            if scale == 0 and record_labels:
                block_fields = parse_labelarray_data(raw_blocks)
                block_ids, labels, spans = zip(*block_fields)
                
            with mgr_client.access_context(output_service.server, False, 1, np.prod(box_shape)):
                post_labelmap_blocks(*output_service.instance_triple, None, raw_blocks, scale,
                                     output_service.enable_downres, output_service.disable_indexing, False, is_raw=True)
            
            return list(set(chain(*labels)))

        all_labels = set()
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

        max_scale = options["max-scale"]
        if max_scale == -1:
            info = fetch_instance_info(*self.input_service.instance_triple)
            max_scale = int(info["Extended"]["MaxDownresLevel"])
            options["max-scale"] = max_scale

        assert self.input_service.available_scales == list(range(1+max_scale)), \
            "Your input's available-scales must include all levels you wish to copy."

        if output_config["dvid"]["create-if-necessary"]:
            creation_depth = output_config["dvid"]["creation-settings"]["max-scale"]
            if creation_depth not in (-1, max_scale):
                msg = (f"Inconsistent max-scale options in the labelmapcopy config options ({max_scale}) and creation-settings options ({creation_depth}). "
                       "Omit max-scale from your creation-settings.")
                raise RuntimeError(msg)
            output_config["dvid"]["creation-settings"]["max-scale"] = max_scale

        # Replace 'auto' dimensions with input bounding box
        replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])
        self.output_service = VolumeService.create_from_config( output_config, self.mgr_client )

        input_service = self.input_service
        output_service = self.output_service
        assert isinstance( output_service, VolumeServiceWriter )

        if output_service.instance_name in fetch_repo_instances(output_service.server, output_service.uuid):
            info = fetch_instance_info(*output_service.instance_triple)
            existing_depth = int(info["Extended"]["MaxDownresLevel"])
            if max_scale not in (-1, existing_depth):
                raise Exception(f"Can't set pyramid-depth to {max_scale}: \n"
                                f"Data instance '{output_service.instance_name}' already existed, with depth {existing_depth}.\n"
                                f"For now, you are required to populate ALL scales of the output, or create a new output instance from scratch.")


        # Transposed/remapped services aren't supported because we're not going to inflate the downloaded blocks.
        assert all(not isinstance( svc, TransposedVolumeService ) for svc in input_service.service_chain)
        assert all(not isinstance( svc, TransposedVolumeService ) for svc in output_service.service_chain)
        assert all(not isinstance( svc, LabelmappedVolumeService ) for svc in input_service.service_chain)
        assert all(not isinstance( svc, LabelmappedVolumeService ) for svc in output_service.service_chain)
        assert all(not isinstance( svc, ScaledVolumeService ) or svc.scale_delta == 0 for svc in output_service.service_chain)

        assert all(not isinstance( svc, ScaledVolumeService ) or svc.scale_delta == 0 for svc in input_service.service_chain), \
            "For now, we don't support rescaled input, though it would be possible in theory."

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

        assert not (input_service.bounding_box_zyx % input_service.block_width).any(), \
            "Input bounding-box should be a multiple of the block size in all dimensions."
        assert not (output_service.bounding_box_zyx % output_service.block_width).any(), \
            "Output bounding-box should be a multiple of the block size in all dimensions."

        assert not (input_service.preferred_message_shape % input_service.block_width).any(), \
            "Input message-block-shape should be a multiple of the block size in all dimensions."
        assert not (output_service.preferred_message_shape % output_service.block_width).any(), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."
        
        assert len(options["slab-shape"]) == 3
        slab_shape_zyx = np.array(options["slab-shape"][::-1])
        replace_default_entries(slab_shape_zyx, input_service.preferred_message_shape)
        options["slab-shape"] = slab_shape_zyx[::-1].tolist()
        
        assert (slab_shape_zyx % input_service.preferred_message_shape[0] == 0).all(), \
            "slab-shape must be divisible by the brick shape"
