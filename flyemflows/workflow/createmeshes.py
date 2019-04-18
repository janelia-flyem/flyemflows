import os
import copy
import logging

import numpy as np
import pandas as pd

import dask.dataframe as ddf

from neuclease.util import Timer, SparseBlockMask, box_intersection, extract_subvol
from neuclease.dvid import fetch_mappings
from dvid_resource_manager.client import ResourceManagerClient
from dvidutils import LabelMapper

from vol2mesh import Mesh

from ..util.dask_util import drop_empty_partitions
from .util.config_helpers import BodyListSchema, load_body_list, LabelGroupSchema, load_label_groups
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
    
        #"default": {},
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
                "type": "string",
                "default": ""
            },
            "create-if-necessary": {
                "description": "Whether or not to create the instance if it doesn't already exist.\n"
                               "If you expect the instance to exist on the server already, leave this\n"
                               "set to False to avoid confusion in the case of typos, UUID mismatches, etc.\n",
                "type": "boolean",
                "default": False
            }
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
                "description": "FIXME",
                "type": "string",
                "default": "meshes"
            }
        }
    }

    MeshParametersSchema = \
    {
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
            "compute-normals": {
                "description": "Compute vertex normals and include them in the uploaded results.",
                "type": "boolean",
                "default": False
            }
        }
    }

    SizeFiltersSchema = \
    {
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

    CreateMeshesOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the CreateMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "subset-labels": BodyListSchema,

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
            
            "max-body-vertices": {
                "description": "If necessary, dynamically increase decimation on a per-body, per-brick basis so that\n"
                               "the total vertex count for each mesh (across all bricks) final mesh will not exceed\n"
                               "this total vertex count.\n"
                               "If omitted, no maximum is used.\n",
                "oneOf": [{"type": "number"}, {"type": "null"}],
                "default": None
            },

            "rescale-before-write": {
                "description": "How much to rescale the meshes before writing to DVID.\n"
                               "Specified as a multiplier, not power-of-2 'scale'.\n",
                "type": "number",
                "default": 1.0
            },
            "format": {
                "description": "Format to save the meshes in. ",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  Note: Data is presumed to be 8nm resolution
                "default": "obj"
            },
            "include-empty": {
                "description": "Objects too small to generate proper meshes for may be 'serialized' as an empty buffer (0 bytes long).\n"
                               "This setting specifies whether 0-byte files are uploaded to the destination server in such cases,\n"
                               "or if they are omitted entirely.\n",
                "type": "boolean",
                "default": False
            },
            "skip-existing": {
                "description": "Do not generate meshes for meshes that already exist in the output location.\n",
                "type": "boolean",
                "default": False
            },
            
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,

        "output": {
            "oneOf": [
                TarsupervoxelsOutputSchema,
                KeyvalueOutputSchema,
                DirectoryOutputSchema,
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

    def _prepare_output(self):
        destination = self.config["output"]
        if 'directory' in destination:
            os.makedirs(destination['directory'])
        else:
            # TODO: other output destinations (respecting create-if-necessary, etc.)
            raise NotImplementedError("Output type not yet supported.")

    def execute(self):
        self._sanitize_config()

        input_config = self.config["input"]
        options = self.config["createmeshes"]
        resource_config = self.config["resource-manager"]

        self.resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        input_service = VolumeService.create_from_config(input_config, self.resource_mgr_client)

        self._prepare_output()        

        # Load body list and eliminate duplicates
        is_supervoxels = False
        if isinstance(input_service.base_service, DvidVolumeService):
            is_supervoxels = input_service.base_service.supervoxels
        subset_labels = load_body_list(options["subset-labels"], is_supervoxels)

        brickwall = self.init_brickwall(input_service, subset_labels)
        bricks_ddf = BrickWall.bricks_as_ddf(brickwall.bricks, logical=True)
        bricks_ddf = bricks_ddf[['lz0', 'ly0', 'lx0', 'brick']]
        
        def compute_brick_labelcounts(brick_df):
            brick_counts_dfs = []
            for row in brick_df.itertuples():
                brick = row.brick
                inner_box = box_intersection(brick.logical_box, brick.physical_box)
                inner_box -= brick.physical_box[0]
                inner_vol = extract_subvol(brick.volume, inner_box)
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

            return pd.concat(brick_counts_dfs, ignore_index=True)

        dtypes = {'label': np.uint64, 'count': np.int64,
                  'lz0': np.int32, 'ly0': np.int32, 'lx0': np.int32}
        brick_counts_df = bricks_ddf.map_partitions(compute_brick_labelcounts, meta=dtypes).clear_divisions().compute()
        
        if is_supervoxels:
            seg_instance = input_service.base_service.instance_triple
            mapping = fetch_mappings(*seg_instance)
            mapper = LabelMapper(mapping.index.values, mapping.values)

            brick_counts_df['sv'] = brick_counts_df['label'].values
            brick_counts_df['body'] = mapper.apply(brick_counts_df['sv'].values)
            total_sv_counts = brick_counts_df.groupby('sv')['count'].sum().rename('sv_size').reset_index()
            total_body_counts = brick_counts_df.groupby('body')['count'].sum().rename('body_size').reset_index()
        else:
            # Every label is treated as a supervoxel for our purposes.
            brick_counts_df['sv'] = brick_counts_df['label']
            brick_counts_df['body'] = brick_counts_df['label']
            total_sv_counts = brick_counts_df.groupby('sv')['count'].sum().rename('sv_size').reset_index()
            total_body_counts = total_sv_counts.rename(columns={'sv': 'body', 'sv_size': 'body_size'})

        brick_counts_df = brick_counts_df.merge(total_sv_counts, 'left', 'sv')
        brick_counts_df = brick_counts_df.merge(total_body_counts, 'left', 'body')

        logger.info("Saving brick-counts.npy")
        np.save('brick-counts.npy', brick_counts_df.to_records(index=False))

        brick_counts_grouped_df = brick_counts_df.groupby(['lz0', 'ly0', 'lx0'])[['sv', 'sv_size', 'body', 'body_size']].agg(list).reset_index()
        with Timer("Distributing counts to bricks", logger):
            # Send count lists to their respective bricks
            # Use an inner merge to discard bricks that had no objects of interest.
            brick_counts_grouped_ddf = ddf.from_pandas(brick_counts_grouped_df, npartitions=1) # FIXME: What's good here?
            bricks_ddf = bricks_ddf.merge(brick_counts_grouped_ddf, 'inner', ['lz0', 'ly0', 'lx0'])
            bricks_ddf = drop_empty_partitions(bricks_ddf)
        
        def compute_meshes_for_bricks(bricks_partition_df):
            assert len(bricks_partition_df) > 0, "partition is empty" # drop_empty_partitions() should have eliminated these.
            mesh_dfs = []
            for row in bricks_partition_df.itertuples():
                stats_df = pd.DataFrame({'sv': row.sv, 'sv_size': row.sv_size,
                                         'body': row.body, 'body_size': row.body_size})

                brick_meshes_df = compute_meshes_for_brick(row.brick, stats_df, options)
                mesh_dfs.append(brick_meshes_df)

            return pd.concat(mesh_dfs, ignore_index=True)
                
        dtypes = {'sv': np.int64, 'body': np.int64, 'mesh': object, 'vertex_count': int, 'compressed_size': int}
        brick_meshes_ddf = bricks_ddf.map_partitions(compute_meshes_for_bricks, meta=dtypes).clear_divisions()

        final_smoothing = options["post-stitch-parameters"]["smoothing"]
        final_decimation = options["post-stitch-parameters"]["decimation"]

        def assemble_sv_meshes(sv_brick_meshes_df):
            sv = sv_brick_meshes_df['sv'].iloc[0]

            # TODO: stitch-method
            mesh = Mesh.concatenate_meshes(sv_brick_meshes_df['mesh'])
            
            if final_smoothing != 0:
                mesh.laplacian_smooth(final_smoothing)
            
            if final_decimation != 1.0:
                mesh.simplify(final_decimation, in_memory=True)
            
            # TODO: respect compute-normals (discard if neccesary)
            
            vertex_count = len(mesh.vertices_zyx)
            return pd.DataFrame({'sv': [sv],
                                 'mesh': [mesh],
                                 'vertex_count': [vertex_count]})

        sv_brick_meshes_ddf = brick_meshes_ddf.groupby('sv')
        
        dtypes = {'sv': np.uint64, 'mesh': object, 'vertex_count': np.int64}
        sv_meshes_ddf = sv_brick_meshes_ddf.apply(assemble_sv_meshes, meta=dtypes)
        
        # TODO: Repartition?
        
        destination = self.config["output"]
        fmt = options["format"]
        def write_sv_meshes(sv_meshes_df):
            for row in sv_meshes_df.itertuples():
                if 'directory' in destination:
                    path = destination['directory'] + f"/{row.sv}.{fmt}"
                    row.mesh.serialize(path)
                else:
                    msg = f"Output destination type not yet supported: {destination.keys()[0]}"
                    raise NotImplementedError(msg)
            
            # TODO: Return the final size in bytes
            #       (Requires a tweak to the serialization call.)
            return None
        
        # TODO: Use map_paritions to bundle writes
        #dtypes = {'sv': np.uint64, 'mesh': object}
        dtypes = {'none': object}
        sv_meshes_ddf[['sv', 'mesh']].map_partitions(write_sv_meshes, meta=dtypes).compute()
        

    def init_brickwall(self, volume_service, subset_labels):
        sbm = None
        if subset_labels:
            try:
                brick_coords_df = volume_service.sparse_block_mask_for_labels(subset_labels)
                np.save('brick-coords.npy', brick_coords_df.to_records(index=False))
    
                brick_shape = volume_service.preferred_message_shape
                brick_indexes = brick_coords_df[['z', 'y', 'x']].values // brick_shape
                sbm = SparseBlockMask.create_from_lowres_coords(brick_indexes, brick_shape)
            except NotImplementedError:
                logger.warning("The volume service does not support sparse fetching.  All bricks will be analyzed.")
                sbm = None
            
        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            
            # Apply halo WHILE downloading the data.
            # TODO: Allow the user to configure whether or not the halo should
            #       be fetched from the outset, or added after the blocks are loaded.
            halo = self.config["createmeshes"]["halo"]
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, halo, sbm, compression='lz4_2x')

        return brickwall


def compute_meshes_for_brick(brick, stats_df, options):
    size_filters = options["size-filters"]

    min_sv_size = size_filters['minimum-supervoxel-size'] #@UnusedVariable
    max_sv_size = size_filters['maximum-supervoxel-size'] #@UnusedVariable
    min_body_size = size_filters['minimum-body-size']     #@UnusedVariable
    max_body_size = size_filters['maximum-body-size']     #@UnusedVariable
    
    smoothing = options["pre-stitch-parameters"]["smoothing"]
    decimation = options["pre-stitch-parameters"]["decimation"]

    # TODO: skip-existing
    # TODO: max-body-vertices

    q = ('sv_size >= @min_sv_size and sv_size <= @max_sv_size and '
         'body_size >= @min_body_size and body_size <= @max_body_size')
    filtered_stats_df = stats_df.query(q)
    
    cols = ['sv', 'body', 'mesh', 'vertex_count', 'compressed_size']
    if len(filtered_stats_df) == 0:
        empty64 = np.zeros((0,1), dtype=np.uint64)
        emptyObject = np.zeros((0,1), dtype=object)
        return pd.DataFrame([empty64, empty64, emptyObject, empty64, empty64])
    
    volume = brick.volume
    brick.compress()
    
    meshes = []
    for row in filtered_stats_df.itertuples():
        mesh, vertex_count, compressed_size = generate_mesh(volume, brick.physical_box, row.sv, smoothing, decimation)
        meshes.append( (row.sv, row.body, mesh, vertex_count, compressed_size ) )
    
    return pd.DataFrame(meshes, columns=cols)


def generate_mesh(volume, box, label, smoothing, decimation):
    # TODO: rescale-before-write
    # TODO: include-empty (or better, always produce something)

    mask = (volume == label)
    mesh = Mesh.from_binary_vol(mask, box)
    
    if smoothing != 0:
        mesh.laplacian_smooth(smoothing)
    
    if decimation != 1.0:
        mesh.simplify(decimation, in_memory=True)
    
    vertex_count = len(mesh.vertices_zyx)
    compressed_size = mesh.compress()

    return mesh, vertex_count, compressed_size

