import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from collections.abc import Mapping

import numpy as np
import pandas as pd

from neuclease.util import Timer, lexsort_columns, groupby_presorted
from dvid_resource_manager.client import ResourceManagerClient
logger = logging.getLogger(__name__)

class VolumeService(metaclass=ABCMeta):

    SUPPORTED_SERVICES = ['hdf5', 'dvid', 'boss', 'tensorstore', 'brainmaps', 'n5', 'slice-files', 'zarr']

    @abstractproperty
    def dtype(self):
        raise NotImplementedError

    @abstractproperty
    def preferred_message_shape(self):
        raise NotImplementedError

    @abstractproperty
    def block_width(self):
        raise NotImplementedError

    @property
    def base_service(self):
        """
        If this service wraps another one (e.g. ScaledVolumeService, etc.),
        return the wrapped service.
        Default for 'base' services (e.g. DvidVolumeService) is to just return self.
        """
        return self

    @property
    def service_chain(self):
        """
        If this service wraps another service(s) (e.g. ScaledVolumeService, etc.),
        return the chain wrapped services, including the base service.
        If this service is a base service, self will be the only item in the list.
        """
        if hasattr(self, 'original_volume_service'):
            return [self] + self.original_volume_service.service_chain
        return [self]

    @property
    def resource_manager_client(self):
        """
        Return the base_service's resource manager client.
        If the base service doesn't override this property,
        the default is to return a dummy client.
        (See dvid_resource_manager/client.py)
        """
        if self.base_service is self:
            # Dummy client
            return ResourceManagerClient("", 0)
        return self.base_service.resource_manager_client

    @classmethod
    def create_from_config( cls, volume_config, resource_manager_client=None ):
        from .hdf5_volume_service import Hdf5VolumeService
        from .dvid_volume_service import DvidVolumeService
        from .boss_volume_service import BossVolumeServiceReader
        from .tensorstore_volume_service import TensorStoreVolumeServiceReader
        from .brainmaps_volume_service import BrainMapsVolumeServiceReader
        from .n5_volume_service import N5VolumeService
        from .zarr_volume_service import ZarrVolumeService
        from .slice_files_volume_service import SliceFilesVolumeService

        VolumeService.remove_default_service_configs(volume_config)

        service_keys = set(volume_config.keys()).intersection( set(VolumeService.SUPPORTED_SERVICES) )
        if len(service_keys) != 1:
            raise RuntimeError(f"Unsupported service (or too many specified): {service_keys}")

        # Choose base service
        if "hdf5" in volume_config:
            service = Hdf5VolumeService( volume_config )
        elif "dvid" in volume_config:
            service = DvidVolumeService( volume_config, resource_manager_client )
        elif "boss" in volume_config:
            service = BossVolumeServiceReader( volume_config, resource_manager_client )
        elif "tensorstore" in volume_config:
            service = TensorStoreVolumeServiceReader( volume_config, resource_manager_client )
        elif "brainmaps" in volume_config:
            service = BrainMapsVolumeServiceReader( volume_config, resource_manager_client )
        elif "n5" in volume_config:
            service = N5VolumeService( volume_config )
        elif "zarr" in volume_config:
            service = ZarrVolumeService( volume_config )
        elif "slice-files" in volume_config:
            service = SliceFilesVolumeService( volume_config )
        else:
            raise RuntimeError( "Unknown service type." )

        for k in ('apply-labelmap', 'transpose-axes', 'rescale-level', 'labelmap'):
            if k in volume_config:
                msg = ("Sorry, the expected config schema has changed, and your config appears out-of-date.\n"
                       "Adapter config settings should now be listed in the 'adapters' section, not at the service top-level.\n"
                       f"Please create an 'adapters' section, and move '{k}', etc. under it.\n")
                raise RuntimeError(msg)

        if 'adapters' not in volume_config:
            return service

        adapter_config = volume_config["adapters"]

        if 'labelmap' in adapter_config:
            raise RuntimeError("Bad key for volume service: 'labelmap' -- did you mean 'apply-labelmap'?")

        # Wrap with labelmap service
        from . import LabelmappedVolumeService
        if ("apply-labelmap" in adapter_config) and (adapter_config["apply-labelmap"]["file-type"] != "__invalid__"):
            service = LabelmappedVolumeService(service, adapter_config["apply-labelmap"])

        # Wrap with transpose service
        from . import TransposedVolumeService
        if ("transpose-axes" in adapter_config) and (adapter_config["transpose-axes"] != TransposedVolumeService.NO_TRANSPOSE):
            service = TransposedVolumeService(service, adapter_config["transpose-axes"])

        # Even if rescale-level == 0, we still wrap in a scaled volumeservice because
        # it enables more 'available-scales'.
        # We only avoid the ScaledVolumeService adapter if rescale-level is None.
        from . import ScaledVolumeService
        if "rescale-level" in adapter_config and adapter_config["rescale-level"] is not None:
            rescale_cfg = adapter_config["rescale-level"]
            if isinstance(rescale_cfg, Mapping):
                level = rescale_cfg["level"]
                method = rescale_cfg["method"]
                available_scales = rescale_cfg["available-scales"]
            else:
                level = rescale_cfg
                method = None
                available_scales = None

            service = ScaledVolumeService(service, level, method, available_scales)

        return service

    @classmethod
    def remove_default_service_configs(cls, volume_config):
        """
        The validate_and_inject_defaults() function will insert default
        settings for all possible service configs, but we are only interested
        in the one that the user actually wrote.
        Fortunately, that function places a special hint 'from_default' on the config
        dict to make it easy to figure out which configs were completely default-generated.
        """
        for key in VolumeService.SUPPORTED_SERVICES:
            if key in volume_config and hasattr(volume_config[key], 'from_default') and volume_config[key].from_default:
                del volume_config[key]


class VolumeServiceReader(VolumeService):

    @abstractproperty
    def bounding_box_zyx(self):
        raise NotImplementedError

    @abstractproperty
    def available_scales(self):
        raise NotImplementedError

    @abstractmethod
    def get_subvolume(self, box_zyx, scale=0):
        raise NotImplementedError

    def sample_labels(self, points_zyx, scale=0, npartitions=1024):
        """
        Read the label under each of the given points.
        """
        if isinstance(points_zyx, pd.DataFrame):
            assert not ({*'zyx'} - {*points_zyx.columns}), \
                "points must have columns 'z', 'y', 'x', your dataframe had: {points_zyx.columns.tolist()}"
            points_zyx = points_zyx[[*'zyx']].values
        else:
            points_zyx = np.asarray(points_zyx)

        assert points_zyx.shape[1] == 3

        brick_shape = self.preferred_message_shape // (2**scale)
        idx = np.arange(len(points_zyx))[:, None]

        # columns: [bz, by, bx, z, y, x, i]
        brick_ids_and_points = np.concatenate((points_zyx // brick_shape, points_zyx, idx), axis=1)
        brick_ids_and_points = lexsort_columns(brick_ids_and_points)

        # extract columns brick_ids, zyxi
        brick_ids = brick_ids_and_points[:, :3]
        sorted_points = brick_ids_and_points[:, 3:]

        # This is faster than pandas.DataFrame.groupby() for large data
        point_groups = [*groupby_presorted(sorted_points, brick_ids)]
        num_groups = len(point_groups)
        logger.info(f"Sampling labels for {len(points_zyx)} points from {num_groups} bricks")

        def sample_labels_from_brick(points_zyxi):
            points_zyx = points_zyxi[:, :3]
            box = (points_zyx.min(axis=0), 1+points_zyx.max(axis=0))
            vol = self.get_subvolume(box, scale)
            localpoints = points_zyx - box[0]
            labels = vol[(*localpoints.transpose(),)]
            df = pd.DataFrame(points_zyxi, columns=[*'zyxi'])
            df['label'] = labels
            return df

        import dask.bag as db

        point_groups = db.from_sequence(point_groups, npartitions=npartitions)
        label_dfs = point_groups.map(sample_labels_from_brick).compute()
        label_df = pd.concat(label_dfs, ignore_index=True)

        # Return in the same order the user passed in
        label_df = label_df.sort_values('i')
        return label_df["label"].values


    def sparse_brick_coords_for_labels(self, labels, clip=True):
        """
        Return a DataFrame indicating the brick
        coordinates (starting corner) that encompass the given labels.
        
        Args:
            labels:
                A list of label IDs.
            
            clip:
                If True, filter the results to exclude any coordinates
                that fall outside this service's bounding-box.
                Otherwise, all brick coordinates that encompass the given labels
                will be returned, whether or not they fall within the bounding box.
                
        Returns:
            DataFrame with columns [z,y,x,label],
            where z,y,x represents the starting corner (in full-res coordinates)
            of a brick that contains the label.
        """
        raise NotImplementedError

    ##
    ## Other sparse functions
    ## (adapters that call sparse_brick_coords_for_labels() internally)
    ##
    def sparse_block_mask_for_labels(self, labels, clip=True):
        """
        Determine which bricks (each with our ``preferred_message_shape``)
        would need to be accessed download all data for the given labels,
        and return the result as a ``SparseBlockMask`` object.

        This function uses a dask to fetch the coarse sparsevols in parallel.
        The sparsevols are extracted directly from the labelindex.
        If the ``self.supervoxels`` is True, the labels are grouped
        by body before fetching the labelindexes,
        to avoid fetching the same labelindexes more than once.

        Args:
            labels:
                A list of body IDs (if ``self.supervoxels`` is False),
                or supervoxel IDs (if ``self.supervoxels`` is True).

            clip:
                If True, filter the results to exclude any coordinates
                that fall outside this service's bounding-box.
                Otherwise, all brick coordinates that encompass the given label groups
                will be returned, whether or not they fall within the bounding box.

        Returns:
            ``SparseBlockMask``
        """
        from neuclease.util import SparseBlockMask
        coords_df = self.sparse_brick_coords_for_labels(labels, clip)
        coords_df.drop_duplicates(['z', 'y', 'x'], inplace=True)

        brick_shape = self.preferred_message_shape
        coords_df[['z', 'y', 'x']] //= brick_shape

        coords = coords_df[['z', 'y', 'x']].values
        return SparseBlockMask.create_from_lowres_coords(coords, brick_shape)


    def sparse_brick_coords_for_label_pairs(self, label_pairs, clip=True):
        """
        Given a list of label pairs, determine which bricks in
        the volume contain both labels from at least one of the given pairs.

        If you're interested in examining adjacencies between
        pre-determined pairs of bodies (or supervoxels),
        this function tells you which bricks contain both labels in the pair,
        and thus might encompass the coordinates at which the labels are adjacent.

        Args:
            label_pairs:
                array of shape (N,2) listing the pairs of labels you're interested in.

            clip:
                If True, filter the results to exclude any coordinates
                that fall outside this service's bounding-box.
                Otherwise, all brick coordinates that encompass the given label pairs
                will be returned, whether or not they fall within the bounding box.

        Returns:
            DataFrame with columns ``[z, y, x, group, label]``,
            where ``[z, y, x]`` are brick coordinates (starting corners).
            Note that duplicate brick coordinates will be listed
            (one for each label+pair combination present in the brick).
        """
        label_pairs = np.asarray(label_pairs)
        label_groups_df = pd.DataFrame({'label': label_pairs.reshape(-1)})
        label_groups_df['group'] = np.arange(label_pairs.size, dtype=np.int32) // 2
        return self.sparse_brick_coords_for_label_groups(label_groups_df, 2, clip)


    def sparse_brick_coords_for_label_groups(self, label_groups_df, min_subset_size=2, clip=True):
        """
        Given a set of label groups, determine which bricks in
        the volume contain at least N labels from any group (or groups).

        For instance, suppose you have two groups of labels [1,2,3], [4,5,6] and you're only
        interested in those bricks which contain at least two labels from one (or both)
        of those groups.  A brick that contains only labels [1,4] is not of interest,
        but a brick containing [1,3] is of interest, and so is a brick containing [5,6],
        or [1,2,4,5] etc.

        Args:
            label_groups_df:
                DataFrame with columns ['label', 'group'],
                where the labels are either body IDs or supervoxel IDs
                (depending on ``self.supervoxels``), and the group IDs
                are arbitrary integers.

            min_subset_size:
                The minimum number of labels (N) which must be present from any
                single group to qualify the brick for inclusion in the results.

            clip:
                If True, filter the results to exclude any coordinates
                that fall outside this service's bounding-box.
                Otherwise, all brick coordinates that encompass the given label groups
                will be returned, whether or not they fall within the bounding box.

        Returns:
            DataFrame with columns ``[z, y, x, group, label]``,
            where ``[z, y, x]`` are brick indexes, not full-scale coordinates.
            Note that duplicate brick coordinates will be listed
            (one for each label+group combination present in the brick).
        """
        assert isinstance(label_groups_df, pd.DataFrame)
        label_groups_df = label_groups_df[['label', 'group']]
        assert min_subset_size >= 1
        all_labels = label_groups_df['label'].unique()
        coords_df = self.sparse_brick_coords_for_labels(all_labels, clip)

        with Timer(f"Associating brick coords with group IDs", logger):
            combined_df = coords_df.merge(label_groups_df, 'inner', 'label', copy=False)
            combined_df = combined_df[['z', 'y', 'x', 'group', 'label']]

        if min_subset_size == 1:
            logger.info(f"Keeping {len(combined_df)} label+group combinations")
            return combined_df

        with Timer(f"Determining which bricks to keep for groups of size >= {min_subset_size}", logger):
            # Count the number of labels per group in each block
            labelcounts = combined_df.groupby(['z', 'y', 'x', 'group'], as_index=False, sort=False).agg('count')
            labelcounts = labelcounts.rename(columns={'label': 'labelcount'})

            # Keep brick/group combinations that have enough labels.
            brick_groups_to_keep = labelcounts.loc[(labelcounts['labelcount'] >= min_subset_size)]
            brick_groups_to_keep = brick_groups_to_keep[['z', 'y', 'x', 'group']]
            
        with Timer("Filtering for kept bricks", logger):
            filtered_df = combined_df.merge(brick_groups_to_keep, 'inner', ['z', 'y', 'x', 'group'], copy=False)
            assert filtered_df.columns.tolist() == ['z', 'y', 'x', 'group', 'label']

        logger.info(f"Keeping {len(filtered_df)} label+group combinations")
        return filtered_df

class VolumeServiceWriter(VolumeServiceReader):

    @abstractmethod
    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        raise NotImplementedError

