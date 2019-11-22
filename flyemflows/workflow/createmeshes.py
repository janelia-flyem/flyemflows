import os
import copy
import logging
from itertools import chain

import numpy as np
import pandas as pd
import lz4.frame
from requests import HTTPError

import dask
import dask.bag as db
import dask.dataframe as ddf
from dask.delayed import delayed
from distributed import worker_client

from neuclease.util import Timer, SparseBlockMask, compute_nonzero_box, box_intersection, extract_subvol, parse_timestamp, switch_cwd, iter_batches, box_shape
from neuclease.dvid import (fetch_mappings, fetch_repo_instances, create_tarsupervoxel_instance,
                            create_instance, is_locked, post_load, post_keyvalues, fetch_exists, fetch_keys,
                            fetch_supervoxels, fetch_server_info, fetch_mapping, compute_affected_bodies,
                            read_kafka_messages, filter_kafka_msgs_by_timerange, resolve_ref)

from dvid_resource_manager.client import ResourceManagerClient
from dvidutils import LabelMapper

from vol2mesh import Mesh

from ..util.dask_util import drop_empty_partitions
from .util.config_helpers import BodyListSchema, load_body_list
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from ..brick import BrickWall
from . import Workflow

logger = logging.getLogger(__name__)

class CreateMeshes(Workflow):
    """
    Generate meshes for many (or all) segments in a volume.
    """
    GenericDvidInstanceSchema = \
    {
        "description": "Parameters to specify a generic dvid instance (server/uuid/instance).\n"
                       "Omitted values will be copied from the input, or given default values.",
        "type": "object",
        "required": ["server", "uuid"],
    
        #"default": {}, # Must not have default. (Appears below in a 'oneOf' context.)
        "additionalProperties": False,
        "properties": {
            "server": {
                "description": "location of DVID server to READ.",
                "type": "string",
                "default": ""
            },
            "uuid": {
                "description": "version node from dvid",
                "type": "string",
                "default": ""
            },
            "instance": {
                "description": "Name of the instance to create",
                "type": "string"
            },
            "sync-to": {
                "description": "When creating a tarsupervoxels instance, it should be sync'd to a labelmap instance.\n"
                               "Give the instance name here.",
                "type": "string",
                "default": ""
            },
            "create-if-necessary": {
                "description": "Whether or not to create the instance if it doesn't already exist.\n"
                               "If you expect the instance to exist on the server already, leave this\n"
                               "set to False to avoid confusion in the case of typos, UUID mismatches, etc.\n",
                "type": "boolean",
                "default": False
            },
        }
    }

    TarsupervoxelsOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "tarsupervoxels": GenericDvidInstanceSchema
        }
    }

    KeyvalueOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "keyvalue": GenericDvidInstanceSchema
        }
    }

    DirectoryOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "directory": {
                "description": "Directory to write supervoxel meshes into.",
                "type": "string",
                #"default": "" # Must not have default. (Appears below in a 'oneOf' context.)
            }
        }
    }

    MeshParametersSchema = \
    {
        # TODO: skip-decimation-body-size
        # TODO: downsample-before-marching-cubes?
        "default": {},
        "properties": {
            "smoothing": {
                "description": "How many iterations of smoothing to apply to each mesh before decimation.",
                "type": "integer",
                "default": 0
            },
            "decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 1.0
            },
            "max-vertices": {
                "description": "If necessary, decimate the mesh even further to avoid exceeding this maximum vertex count.\n",
                "type": "number",
                "minValue": 0,
                "default": 0 # no max
            },
            "compute-normals": {
                "description": "Compute vertex normals and include them in the uploaded results.",
                "type": "boolean",
                "default": False
            }
        }
    }

    SizeFiltersSchema = \
    {
        "default": {},
        "properties": {
            "minimum-supervoxel-size": {
                "type": "number",
                "default": 0
            },
            "maximum-supervoxel-size": {
                "type": "number",
                "default": 1e12
            },
            "minimum-body-size": {
                "type": "number",
                "default": 0
            },
            "maximum-body-size": {
                "type": "number",
                "default": 1e12
            }
        }
    }

    SupervoxelListSchema = copy.copy(BodyListSchema)
    SupervoxelListSchema["description"] = \
        ("List of supervoxel IDs to process, or a path to a CSV file with the list.\n"
         "NOTE: If you're using a non-labelmap source (e.g. HDF5, etc.),\n"
         "      it is considered supervoxel data.\n")

    BodyListSchema = copy.copy(BodyListSchema)
    BodyListSchema["description"] = \
        ("List of body IDs to process, a path to a CSV file with the list,\n"
         "or a timestamp (e.g. '2018-11-22 17:34:00') which will be used with \n"
         "the kafka log to determine bodies that have changed (since the given time).\n"
         "NOTE: If you're using a non-labelmap source (e.g. HDF5, etc.), \n"
         "      it is considered supervoxel data.\n"
         "      This config setting can only be used when using a labelmap source.\n")

    CreateMeshesOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the CreateMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "subset-supervoxels": SupervoxelListSchema,
            "subset-bodies": BodyListSchema,

            "halo": {
                "description": "How much overlapping context between bricks in the grid (in voxels)\n",
                "type": "integer",
                "minValue": 1,
                "default": 0
            },

            "pre-stitch-parameters": MeshParametersSchema,
            "post-stitch-parameters": MeshParametersSchema,

            "stitch-method": {
                "description": "How to combine each segment's blockwise meshes into a single file.\n"
                               "Choices are 'simple-concatenate' and 'stitch'.\n"
                               "The 'stitch' method should only be used when halo >= 1 and there is no pre-stitch smoothing or decimation.\n",
                "type": "string",
                "enum": ["simple-concatenate", # Just dump the vertices and faces into the same file
                                               # (renumber the faces to match the vertices, but don't unify identical vertices.)
                                               # If using this setting it is important to use a task-block-halo of > 2 to hide
                                               # the seams, even if smoothing is used.

                         "stitch",             # Search for duplicate vertices and remap the corresponding face corners,
                                               # so that the duplicate entries are not used. Topologically stitches adjacent faces.
                                               # Will be ineffective unless you used a task-block-halo of at least 1, and no
                                               # pre-stitch smoothing or decimation.
                        ],
                
                "default": "simple-concatenate",
            },
            "size-filters": SizeFiltersSchema,
            
            # TODO
            "max-body-vertices": {
                "description": "NOT YET IMPLEMENTED.\n"
                               "If necessary, dynamically increase decimation on a per-body, per-brick basis so that\n"
                               "the total vertex count for each mesh (across all bricks) final mesh will not exceed\n"
                               "this total vertex count.\n"
                               "If omitted, no maximum is used.\n",
                "oneOf": [{"type": "number"}, {"type": "null"}],
                "default": None
            },

            "rescale-before-write": {
                "description": "How much to rescale the meshes before writing to DVID.\n"
                               "Specified as a multiplier, not power-of-2 'scale'.\n"
                               "For anisotropic scaling, provide a list of [X,Y,Z] scaling multipliers.\n",
                "oneOf": [{"type": "number"},
                          {"type": "array", "items": {"type": "number"}}],
                "default": 1.0
            },
            "format": {
                "description": "Format to save the meshes in. Either obj, drc, or ngmesh.\n"
                               "Note: Neuroglancer meshes are expected to be saved using nm units,\n"
                               "      but the meshes from this workflow will use voxel units by default.\n"
                               "      When using the ngmesh format, use rescale-before-write to multiply all mesh coordinates accordingly.\n"
                               "      For example, use rescale-before-write: [16,16,16] if your volume is stored at is at 8nm resolution,\n"
                               "      and you are fetching from scale 1 (2x).\n",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  See note above about scaling.
                "default": "obj"
            },
            "skip-existing": {
                "description": "Do not generate meshes for meshes that already exist in the output location.\n",
                "type": "boolean",
                "default": False
            },
            "include-empty": {
                "description": "Objects too small to generate proper meshes for may be 'serialized' as an empty buffer (0 bytes long).\n"
                               "This setting specifies whether 0-byte files are uploaded to the destination server in such cases,\n"
                               "or if they are omitted entirely.  May only be used in conjunction with 'skip-existing,'\n",
                "type": "boolean",
                "default": False
            },
            "export-labelcounts": {
                "description": "Debugging feature.  Export labelcount DataFrames.",
                "type": "boolean",
                "default": False
            },
            "subset-batch-size": {
                "description": "Instead of computing all meshes in one big batch, break them into smaller batches of the given size.\n"
                               "Only allowed when using 'subset-supervoxels' or 'subset-bodies'.\n",
                "type": "integer",
                "default": 0
            },
            "parallelize-within-bricks": {
                "description": "If multiple labels-of-interest exist within a single brick, \n"
                               "their brick meshes can be computed in parallel, at the expense of\n"
                               "duplicating the brick data to more workers.  It's probably faster\n"
                               "in most cases, but this parallelism this can be disabled if it seems\n"
                               "to be causing issues.\n",
                "type": "boolean",
                "default": True
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,

        "output": {
            "oneOf": [
                DirectoryOutputSchema,
                TarsupervoxelsOutputSchema,
                KeyvalueOutputSchema,
            ],
            "default": {"directory": "meshes"}
        },

        "createmeshes": CreateMeshesOptionsSchema,
        
    })

    @classmethod
    def schema(cls):
        return CreateMeshes.Schema

    def _sanitize_config(self):
        options = self.config["createmeshes"]
        if options['stitch-method'] == 'stitch' and options['halo'] != 1:
            logger.warn("Your config uses 'stitch' aggregation, but your halo != 1.\n"
                        "This will waste CPU and/or lead to unintuitive results.")

        if options['include-empty'] and not options['skip-existing']:
            raise RuntimeError("It's not permitted to use 'include-empty' unless 'skip-existing' was also used, "
                               "to avoid overwriting potentially valid files with empty files.")

        if options['subset-batch-size'] > 0 and not (options['subset-supervoxels'] or options['subset-bodies']):
            raise RuntimeError("The batch feature is not supported unless you explicitly specify subset-supervoxels or subset-bodies.")

    def _init_input_service(self): 
        input_config = self.config["input"]
        resource_config = self.config["resource-manager"]
        self.resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        input_service = VolumeService.create_from_config(input_config, self.resource_mgr_client)
        self.input_service = input_service


    def _prepare_output(self):
        """
        If necessary, create the output directory or
        DVID instance so that meshes can be written to it.
        """
        output_cfg = self.config["output"]
        output_fmt = self.config["createmeshes"]["format"]

        ## directory output
        if 'directory' in output_cfg:
            # Convert to absolute so we can chdir with impunity later.
            output_cfg['directory'] = os.path.abspath(output_cfg['directory'])
            os.makedirs(output_cfg['directory'], exist_ok=True)
            return

        ##
        ## DVID output (either keyvalue or tarsupervoxels)
        ##
        (instance_type,) = output_cfg.keys()

        server = output_cfg[instance_type]['server']
        uuid = output_cfg[instance_type]['uuid']
        instance = output_cfg[instance_type]['instance']

        # If the output server or uuid is left blank,
        # we assume it should be auto-filled from the input settings.
        if server == "" or uuid == "":
            base_input = self.input_service.base_service
            if not isinstance(base_input, DvidVolumeService):
                # Can't copy from the input if the input ain't a dvid source
                raise RuntimeError("Output destination server/uuid was left blank.")

            if server == "":
                server = base_input.server
                output_cfg[instance_type]['server'] = server

            if uuid == "":
                uuid = base_input.uuid
                output_cfg[instance_type]['uuid'] = uuid

        # Resolve in case a branch was given instead of a specific uuid
        uuid = resolve_ref(server, uuid)

        if is_locked(server, uuid):
            info = fetch_server_info(server)
            if "Mode" in info and info["Mode"] == "allow writes on committed nodes":
                logger.warn(f"Output is a locked node ({uuid}), but server is in full-write mode. Proceeding.")
            else:
                raise RuntimeError(f"Can't write to node {uuid} because it is locked.")

        if instance_type == 'tarsupervoxels' and not self.input_is_labelmap_supervoxels():
            msg = ("You shouldn't write to a tarsupervoxels instance unless "
                   "you're reading supervoxels from a labelmap input.\n"
                   "Use a labelmap input source, and set supervoxels: true")
            raise RuntimeError(msg)

        existing_instances = fetch_repo_instances(server, uuid)
        if instance in existing_instances:
            # Instance exists -- nothing to do.
            return

        if not output_cfg[instance_type]['create-if-necessary']:
            msg = (f"Output instance '{instance}' does not exist, "
                   "and your config did not specify create-if-necessary")
            raise RuntimeError(msg)

        assert instance_type in ('tarsupervoxels', 'keyvalue')
        
        ## keyvalue output
        if instance_type == "keyvalue":
            create_instance(server, uuid, instance, "keyvalue", tags=["type=meshes"])
            return

        ## tarsupervoxels output
        sync_instance = output_cfg["tarsupervoxels"]["sync-to"]

        
        if not sync_instance:
            # Auto-fill a default 'sync-to' instance using the input segmentation, if possible.
            base_input = self.input_service.base_service
            if isinstance(base_input, DvidVolumeService):
                if base_input.instance_name in existing_instances:
                    sync_instance = base_input.instance_name

        if not sync_instance:
            msg = ("Can't create a tarsupervoxels instance unless "
                   "you specify a 'sync-to' labelmap instance name.")
            raise RuntimeError(msg)

        if sync_instance not in existing_instances:
            msg = ("Can't sync to labelmap instance '{sync_instance}': "
                   "it doesn't exist on the output server.")
            raise RuntimeError(msg)

        create_tarsupervoxel_instance(server, uuid, instance, sync_instance, output_fmt, tags=["type=meshes"])


    def input_is_labelmap(self):
        return isinstance(self.input_service.base_service, DvidVolumeService)

    def input_is_labelmap_supervoxels(self):
        if isinstance(self.input_service.base_service, DvidVolumeService):
            return self.input_service.base_service.supervoxels
        return False


    def input_is_labelmap_bodies(self):
        if isinstance(self.input_service.base_service, DvidVolumeService):
            return not self.input_service.base_service.supervoxels
        return False

    def execute(self):
        """
        This workflow is designed for generating supervoxel meshes from a labelmap input (with supervoxels: true).
        But other sources. In those cases, each object is treated as a 'supervoxel', and each 'body' has only one supervoxel.
        
        NOTE:
            If your input is a labelmap, but you have configured it to read body labels (i.e. supervoxels: false),
            then it is treated like any other non-supervoxel-aware datasource, such as HDF5.
            That is, in the code below, 'supervoxels' and 'bodies' refer to the same thing.
            The real underlying supervoxel IDs in the labelmap instance are not used.
        
        TODO:
            - max-body-vertices option
            - Rethink terminolgy (supervoxel vs label vs body)
            - Rethink brick_counts_df (label counts) - how can I reduce the size of that data?
              -- drop unnecessary 'label' column?
              -- Require/encourage bigger brick sizes?
              -- Reduce lz0/ly0/lx0 columns to a single index?
              -- Write it to disk (database) instead of collecting it back to the driver?
              -- Distinguish between supervoxels on the surface vs. completely internal supervoxels?
              -- Make brick_counts_df a distributed DF...
                -- probably the most easily scalable option
                -- requires sending subset_supervoxels to all workers,
                   possibly more efficient to filter as an inner merge,
                   depending on the efficiency of that operation when the
                   merge col isn't the index.
              -- Process bodies in batches
            - 
        """
        self._sanitize_config()
        self._init_input_service()
        self._prepare_output()

        subset_supervoxels, existing_svs = self._load_subset_supervoxels()
        batch_size = self.config["createmeshes"]["subset-batch-size"]

        # Prefetch all bricks for all batches because in most cases
        # there will be a lot of common bricks between batches,
        # and the segmentation is relatively cheap to store (it's compressed)
        # TODO:
        #   In some cases it might be nice to fetch the bricks separately for each batch.
        #   This could be made configurable.
        bricks_ddf, subset_supervoxels = self.init_bricks_ddf(self.input_service, subset_supervoxels)
        bricks_ddf = bricks_ddf.persist()
        
        if batch_size == 0:
            self.execute_batch(0, bricks_ddf, subset_supervoxels, existing_svs)
        else:
            assert len(subset_supervoxels) > 0

            batches = iter_batches(subset_supervoxels, batch_size)
            logger.info(f"Creating meshes in {len(batches)} batches")

            for batch_index, batch_subset_svs in enumerate(batches):
                batch_existing_svs = None
                if existing_svs is not None:
                    batch_existing_svs = existing_svs & set(batch_subset_svs)
                
                with Timer(f"Batch {batch_index:02}: Running batch", logger):
                    with switch_cwd(f'batch-{batch_index:02d}', create=True):
                        self.execute_batch(batch_index, bricks_ddf, batch_subset_svs, batch_existing_svs)


    def execute_batch(self, batch_index, bricks_ddf, subset_supervoxels, existing_svs):
        brick_counts_df = self._compute_brick_labelcounts(batch_index, bricks_ddf, subset_supervoxels)
        brick_counts_df = self._compute_body_stats(batch_index, brick_counts_df)
        brick_counts_df, existing_svs = self._filter_svs(batch_index, brick_counts_df, subset_supervoxels, existing_svs)
        num_brick_meshes = len(brick_counts_df)

        bricks_ddf, num_bricks = self._distribute_counts(batch_index, bricks_ddf, brick_counts_df)
        del brick_counts_df
        
        brick_meshes_ddf = self._compute_brickwise_meshes(batch_index, bricks_ddf, num_brick_meshes, num_bricks)
        
        # TODO: max-body-vertices (before assembly...)

        sv_meshes_ddf = self._combine_brick_meshes(batch_index, brick_meshes_ddf)
        del brick_meshes_ddf

        # TODO: Repartition?

        self._write_meshes(batch_index, sv_meshes_ddf, subset_supervoxels, existing_svs)


    def init_bricks_ddf(self, volume_service, subset_labels):
        """
        Initialize a BrickWall from the given volume service and (optionally) a subset of labels,
        and convert it to a dask.DataFrame of bricks before returning it.
        """
        sbm = None
        msg = f"Initializing BrickWall"
        if len(subset_labels) > 0:
            msg = f"Initializing BrickWall for {len(subset_labels)} labels"
            try:
                brick_coords_df = volume_service.sparse_brick_coords_for_labels(subset_labels)
                
                # If any labels couldn't be found, we don't want those included in downstream decisions (e.g. for overwriting).
                subset_labels = np.sort(brick_coords_df['label'].unique())
                np.save('brick-coords.npy', brick_coords_df.to_records(index=False))
    
                brick_shape = volume_service.preferred_message_shape
                brick_indexes = brick_coords_df[['z', 'y', 'x']].values // brick_shape
                sbm = SparseBlockMask.create_from_lowres_coords(brick_indexes, brick_shape)
            except NotImplementedError:
                logger.warning("The volume service does not support sparse fetching.  All bricks will be analyzed.")
                sbm = None

        with Timer(msg, logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            
            # Apply halo WHILE downloading the data.
            # TODO: Allow the user to configure whether or not the halo should
            #       be fetched from the outset, or added after the blocks are loaded.
            halo = self.config["createmeshes"]["halo"]
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, halo, sbm, compression='lz4_2x')

        # Convert to dask.DataFrame
        bricks_ddf = BrickWall.bricks_as_ddf(brickwall.bricks, logical=True)
        bricks_ddf = bricks_ddf[['lz0', 'ly0', 'lx0', 'brick']]
        return bricks_ddf, subset_labels


    def _load_subset_supervoxels(self):
        """
        If the user's config specifies either subset-supervoxels,
        load them and return them.
        
        If the user's config specifies subset-bodies (for a DVID source),
        return the set of all supervoxels contained by those bodies.
        
        If neither subset-supervoxels nor subset-bodies is specified,
        an empty list is returned.
        
        TODO:
            For DVID sources, it would also be possible to pre-filter the subset
            according to the user's size constraints, too, since fetching the
            full labelindex isn't much more expensive than fetching the supervoxel
            list for each body.
        
        Returns:
            (subset_supervoxels, existing_svs)
            list of supervoxel IDs, and the list of pre-existing svs

            If subset-supervoxels (or subset-bodies) was specified and skip-existing is specified,
            then the returned supervoxel list is pre-filtered to omit pre-existing supervoxels.
            
            If 'subset-supervoxels' was not specified or skip-existing was not specified,
            existing_svs will be None.
        """
        options = self.config["createmeshes"]
        subset_supervoxels = load_body_list(options["subset-supervoxels"], True)
        subset_bodies = options["subset-bodies"]

        if len(subset_supervoxels) and len(subset_bodies):
            raise RuntimeError("Can't use both subset-supervoxels and subset-bodies.  Choose one.")

        if len(subset_bodies) and not self.input_is_labelmap():
            raise RuntimeError("Can't use 'subset-bodies' unless your input is a DVID labelmap. Try subset-supervoxels.")

        # Load subset_bodies from a list, CSV, JSON, or from the kafka log
        assert isinstance(subset_bodies, (list, str))
        
        if isinstance(subset_bodies, list) or subset_bodies.endswith(".json") or subset_bodies.endswith(".csv"):
            subset_bodies = load_body_list(options["subset-bodies"], False)
        else:
            # subset-bodies must be a timestamp
            subset_bodies = self._determine_changed_labelmap_bodies(subset_bodies)

        # If user supplied bodies, convert to supervoxels.
        if self.input_is_labelmap_supervoxels() and len(subset_bodies) > 0:
            with Timer(f"Fetching supervoxel set for {len(subset_bodies)} labelmap bodies", logger):
                seg_instance = self.input_service.base_service.instance_triple
                def fetch_svs(body):
                    try:
                        svs = fetch_supervoxels(*seg_instance, body)
                        return (body, svs)
                    except HTTPError as ex:
                        if (ex.response is not None) and (ex.response.status_code == 404):
                            # Body doesn't exist (any more)
                            return (body, np.array([], np.uint64))
                        raise

                bodies_and_svs = db.from_sequence(subset_bodies, npartitions=512).map(fetch_svs).compute()
                bad_bodies = [body for body, svs in bodies_and_svs if len(svs) == 0]
                if bad_bodies:
                    pd.Series(bad_bodies, name='body').to_csv('missing-bodies.csv', index=False, header=True)
                    if len(bad_bodies) < 100:
                        logger.warning(f"Could not fetch supervoxel list for {len(bad_bodies)} bodies: {bad_bodies}")
                    else:
                        logger.warning(f"Could not fetch supervoxel list for {len(bad_bodies)} bodies.  See missing-bodies.csv")

                subset_supervoxels = list(chain(*(svs for body, svs in bodies_and_svs)))
                if len(subset_supervoxels) == 0:
                    raise RuntimeError("None of the listed bodies could be found.  No supervoxels to process.")

                logger.info(f"Selected bodies contain {len(subset_supervoxels)} supervoxels")

        if self.input_is_labelmap_bodies():
            assert len(subset_supervoxels) == 0, \
                "Can't use subset-supervoxels when reading from a labelmap in body mode.  Please use subset-bodies."

            # In the rest of this workflow, voxels read from the input source are referred to as 'supervoxels'.
            # Since the user is reading pre-mapped bodies from a labelmap,
            # We won't be computing group-stats like we would with supervoxel meshes.
            # We now refer to the body IDs as if they were supervoxel IDs.
            #
            # FIXME: I need to just rename 'supervoxel' to 'label' and make this less confusing.
            subset_supervoxels = subset_bodies

        # Pre-exclude existing supervoxels from the subset if necessary.
        existing_svs = None
        if len(subset_supervoxels) and options["skip-existing"]:
            existing_svs = self._determine_existing(subset_supervoxels)
            subset_supervoxels = set(subset_supervoxels) - set(existing_svs)
            subset_supervoxels = list(subset_supervoxels)

            if not subset_supervoxels:
                raise RuntimeError("Based on your current settings, no meshes will be generated at all.\n"
                                   "See subset-supervoxels and skip-existing")
                
        return subset_supervoxels, existing_svs


    def _determine_changed_labelmap_bodies(self, kafka_timestamp_string):
        """
        Read the entire labelmap kafka log, and determine
        which bodies have changed since the given timestamp (a string).
        
        Example timestamps:
            - "2018-11-22"
            - "2018-11-22 17:34:00"
            
        Returns:
            list of body IDs
        """
        logger.info(f"Determining which bodies have changed since {kafka_timestamp_string}")
        
        try:
            kafka_timestamp = parse_timestamp(kafka_timestamp_string)
        except:
            raise RuntimeError(f"Could not parse your subset-bodies config setting ({kafka_timestamp_string}) "
                               "as either a body list or a kafka timestamp")

        if not self.input_is_labelmap():
            raise RuntimeError("Can't specify subset-bodies as a kafka timestamp for sources other than DVID labelmap")

        seg_instance = self.input_service.base_service.instance_triple

        kafka_msgs = read_kafka_messages(*seg_instance)
        filtered_kafka_msgs = filter_kafka_msgs_by_timerange(kafka_msgs, min_timestamp=kafka_timestamp)
        
        new_bodies, changed_bodies, _removed_bodies, new_supervoxels = compute_affected_bodies(filtered_kafka_msgs)
        sv_split_bodies = set(fetch_mapping(*seg_instance, new_supervoxels)) - set([0])
        
        subset_bodies = set(chain(new_bodies, changed_bodies, sv_split_bodies))
        subset_bodies = np.fromiter(subset_bodies, np.uint64)
        subset_bodies = np.sort(subset_bodies).tolist()

        if not subset_bodies:
            raise RuntimeError("Based on your current settings, no meshes will be generated at all.\n"
                               f"No bodies have changed since your specified timestamp {kafka_timestamp_string}")


        logger.info(f"The kafka log shows that {len(subset_bodies)} bodies have changed since ({kafka_timestamp_string})")
        return subset_bodies


    def _determine_existing(self, all_svs):
        """
        Determine which of the given supervoxels already have
        meshes stored in the configured destination.
        """
        with Timer("Determining which meshes are already stored (skip-existing)", logger):
            fmt = self.config["createmeshes"]["format"]
            destination = self.config["output"]
            (destination_type,) = destination.keys()
            assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')

            if destination_type == 'directory':
                d = self.config["output"]["directory"]
                existing_svs = set()
                for sv in all_svs:
                    if os.path.exists(f"{d}/{sv}.{fmt}"):
                        existing_svs.add(sv)

            elif destination_type == 'tarsupervoxels':
                tsv_instance = [destination['tarsupervoxels'][k] for k in ('server', 'uuid', 'instance')]
                exists = fetch_exists(*tsv_instance, all_svs, batch_size=10_000, processes=32, show_progress=False)
                existing_svs = set(exists[exists].index)

            elif destination_type == 'keyvalue':
                logger.warning("Using skip-exists with a keyvalue output.  This might take a LONG time if there are many meshes already stored.")
                kv_instance = [destination['keyvalue'][k] for k in ('server', 'uuid', 'instance')]
                keys = fetch_keys(*kv_instance)
                existing_svs = set(int(k[:-4]) for k in keys)

        return existing_svs


    def _compute_brick_labelcounts(self, batch_index, bricks_ddf, subset_supervoxels):
        """
        Compute the brickwise labelcounts for 
        """
        export_labelcounts = self.config["createmeshes"]["export-labelcounts"]
        if export_labelcounts:
            os.makedirs('brick_ddf_partitions')
            os.makedirs('brick_labelcounts')

        with Timer(f"Batch {batch_index:02}: Computing brickwise labelcounts", logger):
            if export_labelcounts:
                logger.info(f"Batch {batch_index:02}:  *** Also exporting labelcounts (slow) ***")

            dtypes = {'label': np.uint64, 'count': np.int64,
                      'lz0': np.int32, 'ly0': np.int32, 'lx0': np.int32}

            # subset_supervoxels could be large, so it's more efficient to pass
            # it via dask.delayed rather than a normally captured variable.
            brick_counts_df = (bricks_ddf
                                .map_partitions(compute_brick_labelcounts,
                                                    subset_labels=delayed(subset_supervoxels),
                                                    export_labelcounts=export_labelcounts,
                                                    meta=dtypes)
                                .clear_divisions()
                                .compute())

        if len(brick_counts_df) == 0:
            raise RuntimeError("All bricks are empty (no non-zero voxels)!")

        return brick_counts_df

    def _compute_body_stats(self, batch_index, brick_counts_df):
        with Timer(f"Batch {batch_index:02}: Aggregating brick stats into body stats", logger):
            if self.input_is_labelmap_supervoxels():
                seg_instance = self.input_service.base_service.instance_triple
    
                brick_counts_df['sv'] = brick_counts_df['label'].values
    
                # Arbitrary heuristic for whether to do the body-lookups on DVID or on the client.
                if len(brick_counts_df['sv']) < 100_000:
                    # If we're only dealing with a few supervoxels,
                    # ask dvid to map them to bodies for us.
                    brick_counts_df['body'] = fetch_mapping(*seg_instance, brick_counts_df['sv'])
                else:
                    # If we're dealing with a lot of supervoxels, ask for
                    # the entire mapping, and look up the bodies ourselves.
                    mapping = fetch_mappings(*seg_instance)
                    mapper = LabelMapper(mapping.index.values, mapping.values)
                    brick_counts_df['body'] = mapper.apply(brick_counts_df['sv'].values, True)
                
                total_sv_counts = brick_counts_df.groupby('sv')['count'].sum().rename('sv_size').reset_index()
                total_body_counts = brick_counts_df.groupby('body')['count'].sum().rename('body_size').reset_index()
            else:
                # No distinction between 'body' and 'supervoxels'.
                # Every label is treated as a supervoxel for our purposes.
                brick_counts_df['sv'] = brick_counts_df['label']
                brick_counts_df['body'] = brick_counts_df['label']
                total_sv_counts = brick_counts_df.groupby('sv')['count'].sum().rename('sv_size').reset_index()
                total_body_counts = total_sv_counts.rename(columns={'sv': 'body', 'sv_size': 'body_size'})

            brick_counts_df = brick_counts_df.merge(total_sv_counts, 'left', 'sv')
            brick_counts_df = brick_counts_df.merge(total_body_counts, 'left', 'body')

        with Timer(f"Batch {batch_index:02}: Exporting brick sv/body sizes", logger):
            np.save('brick-counts.npy', brick_counts_df.to_records(index=False))
            np.save('sv-sizes.npy', total_sv_counts.to_records(index=False))
            np.save('body-sizes.npy', total_body_counts.to_records(index=False))

        return brick_counts_df


    def _filter_svs(self, batch_index, brick_counts_df, subset_supervoxels, existing_svs):
        """
        Filter the brickwise label counts based on the user's config for:
            - subset-supervoxels
            - supervoxel size
            - body size (if applicable)
            - pre-existing meshes
        """
        options = self.config["createmeshes"]
        with Timer(f"Batch {batch_index:02}: Filtering", logger):
            # Filter for subset
            # (For DVID labelmap sources, this has already been done,
            #  but for other sources, we can only filter after reading the data.)
            if len(subset_supervoxels) > 0:
                sv_set = set(subset_supervoxels) #@UnusedVariable
                brick_counts_df = brick_counts_df.query('sv in @sv_set')
    
            # Filter for sv/body size constraints
            size_filters = options["size-filters"]
            min_sv_size = size_filters['minimum-supervoxel-size'] #@UnusedVariable
            max_sv_size = size_filters['maximum-supervoxel-size'] #@UnusedVariable
            min_body_size = size_filters['minimum-body-size']     #@UnusedVariable
            max_body_size = size_filters['maximum-body-size']     #@UnusedVariable
            q = ('sv_size >= @min_sv_size and sv_size <= @max_sv_size and '
                 'body_size >= @min_body_size and body_size <= @max_body_size')
            brick_counts_df = brick_counts_df.query(q)
    
            if len(brick_counts_df) == 0:
                raise RuntimeError("Based on your current settings, no meshes will be generated at all.\n"
                                   "See sv-sizes.npy and body-sizes.npy")

            
            if options["skip-existing"]:
                all_svs = pd.unique(brick_counts_df['sv'])
                if existing_svs is None:
                    existing_svs = self._determine_existing(all_svs)
            
                brick_counts_df = brick_counts_df.query('sv not in @existing_svs')
                if len(brick_counts_df) == 0:
                    raise RuntimeError("Based on your current settings, no meshes will be generated at all.\n"
                                       "All possible meshes already exist in the destination location.\n"
                                       "To regenerate them anyway, use 'skip-existing: false'")

        with Timer(f"Batch {batch_index:02}: Saving filtered brick counts", logger):
            np.save('filtered-brick-counts.npy', brick_counts_df.to_records(index=False))

        num_bodies = len(pd.unique(brick_counts_df['body']))
        num_svs = len(pd.unique(brick_counts_df['sv']))
        logger.info(f"Batch {batch_index:02}: After filtering, {num_svs} supervoxels remain, from {num_bodies} bodies.")

        return brick_counts_df, existing_svs


    def _distribute_counts(self, batch_index, bricks_ddf, brick_counts_df):
        with Timer(f"Batch {batch_index:02}: Distributing counts", logger):
            brick_counts_grouped_df = (brick_counts_df
                                        .groupby(['lz0', 'ly0', 'lx0'])[['sv', 'sv_size', 'body', 'body_size']]
                                        .agg(list)
                                        .reset_index())

            np.save('grouped-brick-counts.npy', brick_counts_grouped_df.to_records(index=False))

            # Send count lists to their respective bricks
            # Use an inner merge to discard bricks that had no objects of interest.
            brick_counts_grouped_ddf = ddf.from_delayed([delayed(brick_counts_grouped_df)],
                                                        meta=brick_counts_grouped_df.iloc[0:0])
            bricks_ddf = bricks_ddf.merge(brick_counts_grouped_ddf, 'inner', ['lz0', 'ly0', 'lx0'])
            bricks_ddf = drop_empty_partitions(bricks_ddf)

        return bricks_ddf, len(brick_counts_grouped_df)


    def _compute_brickwise_meshes(self, batch_index, bricks_ddf, num_brick_meshes, num_bricks):
        options = self.config["createmeshes"]
        def compute_meshes_for_bricks(bricks_partition_df):
            assert len(bricks_partition_df) > 0, "partition is empty" # drop_empty_partitions() should have eliminated these.
            result_dfs = []
            for row in bricks_partition_df.itertuples():
                assert len(row.sv) > 0
                stats_df = pd.DataFrame({'sv': row.sv, 'sv_size': row.sv_size,
                                         'body': row.body, 'body_size': row.body_size})

                dtypes = {'sv': np.uint64, 'sv_size': np.uint64,
                          'body': np.uint64, 'body_size': np.uint64}
                stats_df = stats_df.astype(dtypes)

                brick_meshes_df = compute_meshes_for_brick(row.brick, stats_df, options)
                brick_meshes_df['lz0'] = row.lz0
                brick_meshes_df['ly0'] = row.ly0
                brick_meshes_df['lx0'] = row.lx0

                # Reorder columns
                cols = ['lz0', 'ly0', 'lx0', 'sv', 'body', 'mesh', 'vertex_count', 'compressed_size']
                brick_meshes_df = brick_meshes_df[cols]
                
                result_dfs.append(brick_meshes_df)

            result_df = pd.concat(result_dfs, ignore_index=True)
            assert result_df.columns.tolist() == cols
            return result_df
                
        dtypes = {'lz0': np.int32, 'ly0': np.int32, 'lx0': np.int32,
                  'sv': np.uint64, 'body': np.uint64,
                  'mesh': object,
                  'vertex_count': int, 'compressed_size': int}

        brick_meshes_ddf = bricks_ddf.map_partitions(compute_meshes_for_bricks, meta=dtypes).clear_divisions()
        brick_meshes_ddf = brick_meshes_ddf.persist()

        msg = f"Batch {batch_index:02}: Computing {num_brick_meshes} brickwise meshes from {num_bricks} bricks"
        with Timer(msg, logger):
            # Export brick mesh statistics
            os.makedirs('brick-mesh-stats')
            brick_stats_ddf = brick_meshes_ddf.drop(['mesh'], axis=1)

            # to_csv() blocks, so this triggers the computation.
            brick_stats_ddf.to_csv('brick-mesh-stats/partition-*.csv', index=False, header=True)
            del brick_stats_ddf

        return brick_meshes_ddf


    def _combine_brick_meshes(self, batch_index, brick_meshes_ddf):
        options = self.config["createmeshes"]
        final_smoothing = options["post-stitch-parameters"]["smoothing"]
        post_decimation = options["post-stitch-parameters"]["decimation"]
        max_vertices = options["post-stitch-parameters"]["max-vertices"]
        compute_normals = options["post-stitch-parameters"]["compute-normals"]

        stitch_method = options["stitch-method"]
        def assemble_sv_meshes(sv_brick_meshes_df):
            assert len(sv_brick_meshes_df) > 0
            
            try:
                sv = sv_brick_meshes_df['sv'].iloc[0]
            except Exception as ex:
                # Re-raise with the whole input
                # (Can't use exception chaining, sadly.)
                # NOTE:
                #   If any of the code below raises an exception,
                #   it can have strage consequences for subsequent calls to this function.
                #   If you're seeing weird errors here, like KeyError: 'sv',
                #   that's probably a sign that something BELOW is failing.
                #   Step through the code below in a debugger.
                np.save('failed-sv_brick_meshes_df.npy', sv_brick_meshes_df.to_records(index=True))
                raise Exception('WrappedError:', type(ex), ex,
                                sv_brick_meshes_df.index,
                                sv_brick_meshes_df.columns.tolist(),
                                str(sv_brick_meshes_df.iloc[0]),
                                'See failed-sv_brick_meshes_df.npy')

            assert (sv_brick_meshes_df['sv'] == sv).all()

            mesh = Mesh.concatenate_meshes(sv_brick_meshes_df['mesh'])
            
            if stitch_method == 'stitch':
                mesh.stitch_adjacent_faces(drop_unused_vertices=True, drop_duplicate_faces=True)
            
            if final_smoothing != 0:
                mesh.laplacian_smooth(final_smoothing)
            
            final_decimation = post_decimation
            if max_vertices != 0 and len(mesh.vertices_zyx) > max_vertices:
                final_decimation = min( post_decimation, max_vertices / len(mesh.vertices_zyx) )
                
            if final_decimation != 1.0:
                mesh.simplify(final_decimation, in_memory=True)
            
            if not compute_normals:
                mesh.drop_normals()
            elif len(mesh.normals_zyx) == 0:
                mesh.recompute_normals()

            vertex_count = len(mesh.vertices_zyx)
            compressed_size = mesh.compress()
            
            return pd.DataFrame({'sv': sv,
                                 'mesh': mesh,
                                 'vertex_count': vertex_count,
                                 'compressed_size': compressed_size},
                                index=[sv])

        sv_brick_meshes_dgb = brick_meshes_ddf.groupby('sv')
        #del brick_meshes_ddf
        
        dtypes = {'sv': np.uint64, 'mesh': object, 'vertex_count': np.int64, 'compressed_size': int}
        sv_meshes_ddf = sv_brick_meshes_dgb.apply(assemble_sv_meshes, meta=dtypes)
        del sv_brick_meshes_dgb
        sv_meshes_ddf = drop_empty_partitions(sv_meshes_ddf)

        with Timer(f"Batch {batch_index:02}: Combining brick meshes", logger):
            # Export stitched mesh statistics
            os.makedirs('stitched-mesh-stats')
            
            # to_csv() blocks, so this triggers the computation.
            sv_meshes_ddf[['sv', 'vertex_count', 'compressed_size']].to_csv('stitched-mesh-stats/partition-*.csv', index=False, header=True)

        return sv_meshes_ddf


    def _write_meshes(self, batch_index, sv_meshes_ddf, subset_supervoxels, existing_svs):
        options = self.config["createmeshes"]
        fmt = options["format"]
        include_empty = options["include-empty"]
        skip_existing = options["skip-existing"]
        destination = self.config["output"]
        resource_mgr = self.resource_mgr_client
        
        def write_sv_meshes(sv_meshes_df, log=True):
            (destination_type,) = destination.keys()
            assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')

            names = [f"{sv}.{fmt}" for sv in sv_meshes_df['sv']]
            binary_meshes = [serialize_mesh(sv, mesh, None, fmt=fmt, log=log)
                             for (sv, mesh) in sv_meshes_df[['sv', 'mesh']].itertuples(index=False)]
            keyvalues = dict(zip(names, binary_meshes))
            filesizes = [len(mesh_bytes) for mesh_bytes in keyvalues.values()]
            
            if not include_empty:
                keyvalues = {k:v for (k,v) in keyvalues.items() if len(v) > 0}

            if destination_type == 'directory':
                for name, mesh_bytes in keyvalues.items():
                    path = destination['directory'] + "/" + name
                    with open(path, 'wb') as f:
                        f.write(mesh_bytes)
            else:
                instance = [destination[destination_type][k] for k in ('server', 'uuid', 'instance')]
                with resource_mgr.access_context(instance[0], False, 1, sum(filesizes)):
                    if destination_type == 'tarsupervoxels':
                        post_load(*instance, keyvalues)
                    elif 'keyvalue' in destination:
                        post_keyvalues(*instance, keyvalues)

            result_df = sv_meshes_df[['sv', 'vertex_count', 'compressed_size']].copy()
            result_df['file_size'] = filesizes
            return result_df

        with Timer(f"Batch {batch_index:02}: Writing meshes", logger):
            dtypes = {'sv': np.uint64, 'vertex_count': np.int64, 'compressed_size': int, 'file_size': int}
            written_stats_df = sv_meshes_ddf.map_partitions(write_sv_meshes, meta=dtypes).clear_divisions().compute()

        missing_svs = None
        if include_empty and len(subset_supervoxels):
            # This assertion guarantees that we won't overwrite any existing files with empty files
            assert skip_existing, "Can't use include-empty without skip-existing"

            # Write empty files for any supervoxels in the user's subset
            # that we didn't even see in the data (e.g. because they don't exist at the scale we used).
            missing_svs = set(subset_supervoxels) - set(written_stats_df['sv']) - existing_svs
            
        if not missing_svs:
            final_stats_df = written_stats_df
        else:
            logger.warning(f"Batch {batch_index:02}: Writing empty files for {len(missing_svs)} labels from your subset which could not be found in the segmentation")
            missing_sv_meshes_df = pd.DataFrame({'sv': list(missing_svs)}, dtype=np.uint64)
            missing_sv_meshes_df['mesh'] = Mesh(np.zeros((0,3)), np.zeros((0,3))) # Empty mesh
            missing_sv_meshes_df['vertex_count'] = np.int64(0)
            missing_sv_meshes_df['compressed_size'] = 0
            missing_sv_meshes_df['file_size'] = 0
            missing_sv_meshes_df = write_sv_meshes(missing_sv_meshes_df, log=False)
            final_stats_df = pd.concat((written_stats_df, missing_sv_meshes_df))

        np.save('final-mesh-stats.npy', final_stats_df.to_records(index=False))


def compute_brick_labelcounts(brick_df, subset_labels, export_labelcounts):
    """
    For the given pandas DataFrame of Bricks,
    i.e. one partition of the dask.DataFrame returned by BrickWall.bricks_as_ddf(),
    perform a voxel count of every label contained in the brick and return the resulting dataframe.
    
    Note:
        Obviously, this does NOT return a DataFrame of the same
        size or columns as the input dataframe.
        Thus, when used with dask.DataFrame.map_partitions, you must use clear_divisions().
    
    Args:
        brick_df:
            One partition of a brick dataframe, as returned by BrickWall.bricks_as_ddf().
     
        subset_labels:
            The labels IDs of interest.
            If empty, counts for all labels will be returned.

        export_labelcounts:
            Debugging feature.
            See export-labelcounts config option.
        
     Returns:
        pandas DataFrame with columns ['lz0', 'ly0', 'lx0', 'label', 'count'],
        where 'lz0', 'ly0', 'lx0' are the coordinates of the top corner of the
        brick's logical_box.
    """
    if export_labelcounts and len(brick_df) > 0:
        debug_file_name = 'z{:05d}-y{:05d}-x{:05d}'.format(*brick_df[['lz0', 'ly0', 'lx0']].iloc[0].values.tolist())
        debug_file_name += '-{:04d}'.format(np.random.randint(10000))
        np.save(f'brick_ddf_partitions/{debug_file_name}.npy', brick_df[['lz0', 'ly0', 'lx0']].to_records(index=True))

    brick_counts_dfs = []
    for row in brick_df.itertuples():
        brick = row.brick
        inner_box = box_intersection(brick.logical_box, brick.physical_box)
        inner_box -= brick.physical_box[0]
        inner_vol = extract_subvol(brick.volume, inner_box)
        brick.compress() # Discard uncompressed
        
        if len(subset_labels) > 0:
            # Relabel everything to 0 except for our subset of interest,
            # to reduce the size of the dataframe that we send back to the driver.
            subset_labels = np.asarray(subset_labels, dtype=np.uint64)
            inner_vol = np.asarray(inner_vol, dtype=np.uint64, order='C')

            mapper = LabelMapper(subset_labels, subset_labels)
            inner_vol = mapper.apply_with_default(inner_vol, 0)
        
        label_counts = pd.Series(inner_vol.reshape(-1)).value_counts().sort_index()
        label_counts.index.name = 'label'
        label_counts.name = 'count'
        if label_counts.index[0] == 0:
            label_counts = label_counts.iloc[1:]
        
        brick_counts_df = label_counts.reset_index()
        brick_counts_df['lz0'] = brick.logical_box[0,0]
        brick_counts_df['ly0'] = brick.logical_box[0,1]
        brick_counts_df['lx0'] = brick.logical_box[0,2]
        
        brick_counts_dfs.append(brick_counts_df)

        if export_labelcounts:
            debug_file_name = 'z{:05d}-y{:05d}-x{:05d}'.format(*brick.logical_box[0].tolist())
            debug_file_name += '-r{:04d}'.format(np.random.randint(10000))
            np.save(f'brick_labelcounts/{debug_file_name}.npy', brick_counts_df.to_records(index=False))

    if len(brick_counts_dfs) > 0:
        return pd.concat(brick_counts_dfs, ignore_index=True)
    else:
        # Return empty DataFrame, but with correct columns
        s = pd.Series(np.zeros((0,), np.int32), index=np.zeros((0,), np.uint64))
        s.name = 'count'
        s.index.name = 'label'
        df = s.reset_index()
        df['lz0'] = np.zeros((0,), np.int32)
        df['ly0'] = np.zeros((0,), np.int32)
        df['lx0'] = np.zeros((0,), np.int32)
        return df


def compute_meshes_for_brick(brick, stats_df, options):
    logging.getLogger(__name__).info(f"Computing meshes for brick: {brick} ({len(stats_df)} meshes)")
    
    smoothing = options["pre-stitch-parameters"]["smoothing"]
    decimation = options["pre-stitch-parameters"]["decimation"]
    max_vertices = options["pre-stitch-parameters"]["max-vertices"]
    rescale_factor = options["rescale-before-write"]
    
    if isinstance(rescale_factor, list):
        rescale_factor = np.array( rescale_factor[::-1] ) # zyx
    else:
        rescale_factor = np.array(3*[rescale_factor])

    # TODO: max-body-vertices

    cols = ['sv', 'body', 'mesh', 'vertex_count', 'compressed_size']
    if len(stats_df) == 0:
        empty64 = np.zeros((0,1), dtype=np.uint64)
        emptyObject = np.zeros((0,1), dtype=object)
        return pd.DataFrame([empty64, empty64, emptyObject, empty64, empty64], columns=cols)
    
    volume = brick.volume
    brick.compress() # Is this necessary? Or will the brick be discarded?

    def crop(mask, orig_mask_box):
        # Pre-crop the volume, leaving a 1-px halo where possible
        mask_box = compute_nonzero_box(mask)
        mask_box[:] += [(-1, -1, -1), (1, 1, 1)]
        mask_box = box_intersection(mask_box, [(0,0,0), mask.shape])

        mask = extract_subvol(mask, mask_box)
        mask_box += orig_mask_box[0]
        return mask, mask_box

    if len(stats_df) == 1 or not options["parallelize-within-bricks"]:
        mesh_results = []
        for row in stats_df.itertuples():
            mask = (volume == row.sv)
            mask, mask_box = crop(mask, brick.physical_box)
            mesh_results.append( generate_mesh(row.sv, row.body, mask, mask_box, smoothing, decimation, max_vertices, rescale_factor, False) )
    else:
        # Parallelize using dask's ability to launch tasks-within-tasks
        # https://distributed.dask.org/en/latest/task-launch.html
        with worker_client() as client:
            tasks = []
            for row in stats_df.itertuples():
                mask = (volume == row.sv)
                mask, mask_box = crop(mask, brick.physical_box)
    
                # We compress the mask before passing it to delayed(),
                # mostly to save RAM if there are lots of tasks that end up on one node.
                compressed_mask = lz4.frame.compress(np.asarray(mask, order='C'))
    
                args = (row.sv, row.body, compressed_mask, mask_box, smoothing, decimation, max_vertices, rescale_factor, True)
                
                # Apparently the delayed() approach doesn't work
                # (I get errors "Workers don't have promised key")
                #task = delayed(generate_mesh)(*args)

                # Use client/future method instead
                future = client.submit(generate_mesh, *args)
                
                tasks.append( future )

            # See note above.  Use client/future method
            #mesh_results = dask.compute(*tasks)
            mesh_results = client.gather(tasks)

    dtypes = {'sv': np.uint64, 'body': np.uint64,
              'mesh': object,
              'vertex_count': int, 'compressed_size': int}
    
    return pd.DataFrame(mesh_results, columns=cols).astype(dtypes)


def generate_mesh(sv, body, mask, mask_box, smoothing, decimation, max_vertices, rescale_factor, compressed=False):
    if compressed:
        mask_shape = box_shape(mask_box)
        mask = np.frombuffer(lz4.frame.decompress(mask), np.bool).reshape(mask_shape)
    
    mesh = Mesh.from_binary_vol(mask, mask_box)
    
    if smoothing != 0:
        mesh.laplacian_smooth(smoothing)
    
    if max_vertices != 0 and len(mesh.vertices_zyx) > max_vertices:
        decimation = min( decimation, max_vertices / len(mesh.vertices_zyx) )
    
    # Don't bother decimating really tiny meshes -- something usually goes wrong anyway.
    if decimation != 1.0 and len(mesh.vertices_zyx) > 10:
        # TODO: Implement a timeout here for the in-memory case (use multiprocessing)?
        mesh.simplify(decimation, in_memory=True)
    
    if (rescale_factor != 1.0).any():
        mesh.vertices_zyx[:] *= rescale_factor
    
    vertex_count = len(mesh.vertices_zyx)
    compressed_size = mesh.compress()

    return sv, body, mesh, vertex_count, compressed_size


def serialize_mesh(sv, mesh, path=None, fmt=None, log=True):
    """
    Call mesh.serialize(), but if an error occurs,
    log it and save an .obj to 'bad-meshes'
    """
    if log:
        logging.getLogger(__name__).info(f"Serializing mesh for {sv}")

    try:
        return mesh.serialize(path, fmt)
    except:
        logger = logging.getLogger(__name__)
        
        msg = f"Failed to serialize mesh as {fmt}."
        if path:
            msg += f" Attempted to write to {path}"
        try:
            if not os.path.exists('bad-meshes'):
                os.makedirs('bad-meshes', exist_ok=True)
            output_path = f'bad-meshes/failed-serialization-{sv}.obj'
            mesh.serialize(output_path, 'obj')
            msg += f" Wrote to {output_path}"
            logger.error(msg)
        except Exception:
            msg += "Couldn't write as OBJ, either."
            logger.error(msg)

        return b''

