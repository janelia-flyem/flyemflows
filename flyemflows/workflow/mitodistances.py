import os
import copy
import pickle
import logging

import pandas as pd
import distributed
from requests.exceptions import HTTPError

from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Timer, tqdm_proxy
from neuclease.dvid import fetch_annotation_label, fetch_supervoxels
from neuclease.misc.measure_tbar_mito_distances import initialize_results, measure_tbar_mito_distances

from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from ..util import stdout_redirected, as_completed_synchronous
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
            }
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
            "description": "Where to write the output files. Each output is written twice: In CSV format and pickle format.\n"
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

        bodies = load_body_list(options["bodies"], False)
        skip_flags = [os.path.exists(f'{output_dir}/{body}.csv') for body in bodies]
        bodies_df = pd.DataFrame({'body': bodies, 'should_skip': skip_flags})
        bodies = bodies_df.query('not should_skip')['body']

        # Shuffle for better load balance
        # TODO: Would be better to sort by synapse count, and put large bodies first,
        #       assigned to partitions in round-robin style.
        #       Then work stealing will be more effective at knocking out the smaller jobs at the end.
        #       This requires knowing all the body sizes, though.
        #       Perhaps mito count would be a decent proxy for synapse count, and it's readily available.
        bodies = bodies.sample(frac=1.0).values

        os.makedirs('body-logs')
        os.makedirs(output_dir, exist_ok=True)

        mito_server, mito_uuid, mito_instance = (options['mito-labelmap'][k] for k in ('server', 'uuid', 'instance'))

        def _fetch_synapses(body):
            with dvid_mgr_client.access_context(syn_server, True, 1, 1):
                syn_df = fetch_annotation_label(syn_server, syn_uuid, syn_instance, body, format='pandas')
                if len(syn_df) == 0:
                    return syn_df
                syn_types, syn_conf
                syn_df = syn_df.query('kind in @syn_types and conf >= @syn_conf').copy()
                return syn_df[[*'zyx', 'kind', 'conf']]

        def _fetch_mito_ids(body):
            with dvid_mgr_client.access_context(mito_server, True, 1, 1):
                try:
                    return fetch_supervoxels(mito_server, mito_uuid, mito_instance, body)
                except HTTPError:
                    return []

        def process_and_save(body):
            tbars = _fetch_synapses(body)
            valid_mitos = _fetch_mito_ids(body)

            # TODO:
            #   Does the stdout_redirected() mechanism work correctly in the context of multiprocessing?
            #   If not, I should probably just use a custom logging handler instead.
            with open(f"body-logs/{body}.log", "w") as f, stdout_redirected(f), Timer() as timer:
                processed_tbars = []
                if len(tbars) == 0:
                    logging.getLogger(__name__).warning(f"Body {body}: No synapses found")

                if len(valid_mitos) == 0:
                    logging.getLogger(__name__).warning(f"Body {body}: Failed to fetch mito supervoxels")
                    processed_tbars = initialize_results(body, tbars)

                if len(valid_mitos) and len(tbars):
                    processed_tbars = measure_tbar_mito_distances(
                        body_svc, mito_svc, body, tbars=tbars, valid_mitos=valid_mitos)

            if len(processed_tbars) > 0:
                processed_tbars.to_csv(f'{output_dir}/{body}.csv', header=True, index=False)
                with open(f'{output_dir}/{body}.pkl', 'wb') as f:
                    pickle.dump(processed_tbars, f)

            if len(tbars) == 0:
                return (body, 0, 'no-synapses', timer.seconds)

            if len(valid_mitos) == 0:
                return (body, len(processed_tbars), 'no-mitos', timer.seconds)

            return (body, len(tbars), 'success', timer.seconds)

        logger.info(f"Processing {len(bodies)}, skipping {bodies_df['should_skip'].sum()}")
        with dvid_mgr_context:
            batch_size = max(1, len(bodies) // 30_000)
            futures = self.client.map(process_and_save, bodies, batch_size=batch_size)

            # Support synchronous testing with a fake 'as_completed' object
            if hasattr(self.client, 'DEBUG'):
                ac = as_completed_synchronous(futures, with_results=True)
            else:
                ac = distributed.as_completed(futures, with_results=True)

            try:
                results = []
                for f, r in tqdm_proxy(ac, total=len(futures)):
                    results.append(r)
            finally:
                results = pd.DataFrame(results, columns=['body', 'synapses', 'status', 'processing_time'])
                results.to_csv('results-summary.csv', header=True, index=False)
                num_errors = len(results.query('status == "error"'))
                if num_errors:
                    logger.warning(f"Encountered {num_errors} errors. See results-summary.csv")

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
