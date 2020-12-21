import os
import copy
import logging

import pandas as pd
import dask.bag as db

from dvid_resource_manager.client import ResourceManagerClient

from neuclease.dvid import fetch_annotation_label, fetch_supervoxels
from neuclease.misc.measure_tbar_mito_distances import measure_tbar_mito_distances

from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from ..util import stdout_redirected, auto_retry
from .util.config_helpers import BodyListSchema, load_body_list
from .base.contexts import LocalResourceManager
from .base.base_schema import ResourceManagerSchema
from . import Workflow

logger = logging.getLogger(__name__)


class MitoDistances(Workflow):
    """
    For each point in an arbitrary set of points within a neuron,
    calculate the shortest-path distance to the nearest mitochondrion
    within that neuron.

    Parallelized across neurons, not points, so the neuron with the most
    points becomes the theoretical bottleneck for the job.
    """

    MitoDistancesOptionsSchema = {
        "type": "object",
        "description": "Settings specific to the MitoDistances workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            # TODO: Option for arbitrary points, provided via CSV, rather than using dvid to fetch all synapses
            # TODO: Allow non-default SearchConfig lists
            "bodies": {
                **BodyListSchema,
                "description": "List of bodies to process.\n"
                               "Note: The computation will skip any bodies which seem\n"
                               "      to already have results in the output directory!\n"
            },
            "mito-labelmap": {
                "description": "Specify a mito labelmap instance which can be used to fetch the mito IDs for each body.",
                "type": "object",
                "required": ["server", "uuid", "instance"],
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "server": { "type": "string", "default": ""},
                    "uuid": { "type": "string", "default": ""},
                    "instance": { "type": "string", "default": ""}
                }
            },
            "synapse-criteria": {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "server": {
                        "type": "string",
                        "default": ""
                    },
                    "uuid": {
                        "type": "string",
                        "default": ""
                    },
                    "instance": {
                        "type": "string",
                        "default": ""
                    },
                    "type": {
                        "oneOf": [
                            {"type": "null"},
                            {"type": "string", "enum": ["pre", "post"]}
                        ],
                        "default": None
                    },
                    "confidence": {
                        "type": "number",
                        "default": 0.0
                    }
                }
            },
            "dilation-radius": {
                "description": "The body mask can be dilated before the shortest path search begins.\n"
                               "Specify the dilation radius here, in scale-0 voxel units.\n",
                "type": "number",
                "default": 0
            },
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "dvid-access-manager": {
            **ResourceManagerSchema,
            "description": "Resource manager settings to use for throttling "
                           "access to the dvid server(s) used for fetching "
                           "synapses and mito supervoxel IDs.\n"
        },
        "body-seg": SegmentationVolumeSchema,
        "mito-seg": SegmentationVolumeSchema,

        "output-directory": {
            "description": "Where to write the output CSV files.\n"
                           "The computation will skip any bodies which seem to already have results in this directory!\n",
            "type": "string",
            "default": "point-distances"
        },
        "mitodistances": MitoDistancesOptionsSchema
    })

    @classmethod
    def schema(cls):
        return MitoDistances.Schema

    def execute(self):
        options = self.config["mitodistances"]
        output_dir = self.config["output-directory"]
        body_svc, mito_svc = self.init_services()

        # Resource manager context must be initialized before resource manager client
        # (to overwrite config values as needed)
        dvid_mgr_config = self.config["dvid-access-manager"]
        dvid_mgr_context = LocalResourceManager(dvid_mgr_config)
        dvid_mgr_client = ResourceManagerClient( dvid_mgr_config["server"], dvid_mgr_config["port"] )

        syn_server, syn_uuid, syn_instance = (options['synapse-criteria'][k] for k in ('server', 'uuid', 'instance'))
        syn_conf = float(options['synapse-criteria']['confidence'])
        syn_types = ['PreSyn', 'PostSyn']
        if options['synapse-criteria']['type'] == 'pre':
            syn_types = ['PreSyn']
        elif options['synapse-criteria']['type'] == 'post':
            syn_types = ['PostSyn']

        @auto_retry(3)
        def _fetch_synapses(body):
            with dvid_mgr_client.access_context(syn_server, True, 1, 1):
                syn_df = fetch_annotation_label(syn_server, syn_uuid, syn_instance, body, format='pandas')
                syn_types, syn_conf
                syn_df = syn_df.query('kind in @syn_types and conf >= @syn_conf').copy()
                return syn_df

        bodies = load_body_list(options["bodies"], False)
        skip_flags = [os.path.exists(f'{output_dir}/{body}.csv') for body in bodies]
        bodies_df = pd.DataFrame({'body': bodies, 'should_skip': skip_flags})
        bodies = bodies_df.query('not should_skip')['body'].values

        dilation = options["dilation-radius"]
        os.makedirs('body-logs')
        os.makedirs(output_dir, exist_ok=True)

        mito_server, mito_uuid, mito_instance = (options['mito-labelmap'][k] for k in ('server', 'uuid', 'instance'))

        def process_and_save(body):
            with dvid_mgr_client.access_context(mito_server, True, 1, 1):
                valid_mitos = fetch_supervoxels(mito_server, mito_uuid, mito_instance, body)

            with open(f"body-logs/{body}.log", "w") as f:
                with stdout_redirected(f):
                    tbars = _fetch_synapses(body)
                    processed_tbars = measure_tbar_mito_distances(body_svc, mito_svc, body, tbars=tbars, valid_mitos=valid_mitos, dilation_radius_s0=dilation)
                    processed_tbars.to_csv(f'{output_dir}/{body}.csv', header=True, index=False)

        psize = min(10, len(bodies) // (5*self.num_workers))
        psize = max(1, psize)

        with dvid_mgr_context:
            logger.info(f"Processing {len(bodies)}, skipping {bodies_df['should_skip'].sum()}")
            db.from_sequence(bodies, partition_size=psize).map(process_and_save).compute()

    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        mgr_config = self.config["resource-manager"]
        resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )

        body_seg_config = self.config["body-seg"]
        mito_seg_config = self.config["mito-seg"]

        body_svc = VolumeService.create_from_config( body_seg_config, resource_mgr_client )
        mito_svc = VolumeService.create_from_config( mito_seg_config, resource_mgr_client )

        if isinstance(body_svc.base_service, DvidVolumeService):
            assert not body_svc.base_service.supervoxels, \
                "body segmentation source shouldn't be a supervoxel source."

        if isinstance(mito_svc.base_service, DvidVolumeService):
            assert body_svc.base_service.supervoxels, \
                "mito segmentation source MUST be a supervoxel souce. 'Grouped' mitos are not appropriate for this computation."

        return body_svc, mito_svc
