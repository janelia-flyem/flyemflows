import os
import copy
import logging

import pandas as pd
import dask.bag as db


from dvid_resource_manager.client import ResourceManagerClient
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
            # TODO: Option for arbitrary points, provided via CSV, rather than using neuprint.
            # TODO: Allow non-default SearchConfig lists
            "bodies": {
                **BodyListSchema,
                "description": "List of bodies to process.\n"
                               "Note: The computation will skip any bodies which seem\n"
                               "      to already have results in the output directory!\n"
            },
            "neuprint": {
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "server": {
                        "type": "string",
                        # no default
                    },
                    "dataset": {
                        "type": "string",
                        # no default
                    }
                },
            },
            "synapse-criteria": {
                "description": "Keyword arguments to use when fetching the list of synapse points from neuprint.\n"
                               "See neuprint.SynapseCriteria for argument options. Provide then here as json fields.\n"
                               "Note that the default values chosen here are not necessarily the same defaults that neuprint uses.\n",
                "type": "object",
                "additionalProperties": False,
                "default": {},
                "properties": {
                    "rois": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": []
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
                    },
                    "primary_only": {
                        "type": "boolean",
                        "default": True,
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
        "neuprint-resource-manager": {
            **ResourceManagerSchema,
            "description": "Resource manager settings to use for throttling neuprint access.\n"
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
        # Late import (neuprint is not a default dependency in flyemflows)
        from neuprint import Client, fetch_synapses, SynapseCriteria as SC, NeuronCriteria as NC

        options = self.config["mitodistances"]
        output_dir = self.config["output-directory"]
        body_svc, mito_svc = self.init_services()

        logger.info(f"Using {options['neuprint']['server']} / {options['neuprint']['dataset']}")

        @auto_retry(3)
        def create_client():
            return Client(options['neuprint']['server'], options['neuprint']['dataset'])
        c = create_client()

        neuprint_mgr_config = self.config["neuprint-resource-manager"]
        neuprint_mgr_client = ResourceManagerClient( neuprint_mgr_config["server"], neuprint_mgr_config["port"] )

        sc = SC(**options['synapse-criteria'], client=c)

        @auto_retry(3)
        def _fetch_synapses(body):
            nc = NC(bodyId=body, label='Segment', client=c)
            with neuprint_mgr_client.access_context(options['neuprint']['server'], True, 1, 1):
                return fetch_synapses(nc, sc, client=c)

        bodies = load_body_list(options["bodies"], False)
        skip_flags = [os.path.exists(f'{output_dir}/{body}.csv') for body in bodies]
        bodies_df = pd.DataFrame({'body': bodies, 'should_skip': skip_flags})
        bodies = bodies_df.query('not should_skip')['body'].values
        logger.info(f"Processing {len(bodies)}, skipping {bodies_df['should_skip'].sum()}")

        dilation = options["dilation-radius"]
        os.makedirs('body-logs')
        os.makedirs(output_dir, exist_ok=True)

        def process_and_save(body):
            with open(f"body-logs/{body}.log", "w") as f:
                with stdout_redirected(f):
                    tbars = _fetch_synapses(body)
                    processed_tbars = measure_tbar_mito_distances(body_svc, mito_svc, body, tbars=tbars, dilation_radius_s0=dilation)
                    processed_tbars.to_csv(f'{output_dir}/{body}.csv', header=True, index=False)

        psize = min(10, len(bodies) // (5*self.num_workers))
        psize = max(1, psize)

        with LocalResourceManager(neuprint_mgr_config):
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
                "body segmentation source must NOT be a supervoxel souce. (Neuprint isn't aware of supervoxels...)"

        if isinstance(mito_svc.base_service, DvidVolumeService):
            assert body_svc.base_service.supervoxels, \
                "mito segmentation source MUST be a supervoxel souce. 'Grouped' mitos are not appropriate for this computation."

        return body_svc, mito_svc
