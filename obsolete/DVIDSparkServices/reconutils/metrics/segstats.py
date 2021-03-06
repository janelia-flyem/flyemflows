from __future__ import division
from math import log

"""Contains all overlap information and stat types for a given subvolume.

It provides functionality to hold stats for a given volume and top-level
functionality to combine volume stats together.
"""
class SubvolumeStats(object):
    def __init__(self, subvolume, voxelfilter=1000, ptfilter=10, num_displaybodies=100, nogt=False, selfcompare=False, enable_sparse=False, important_bodies=None, subvolume_threshold=0):
        # TODO: support for skeletons
        self.subvolumes = [subvolume]
        self.disable_subvolumes = False 
        self.ignore_subvolume = self.disable_subvolumes

        # contains "rand", "vi", etc for substack
        self.subvolume_stats = []

        # contains overlaps over the various sets
        # of overlap computed
        self.gt_overlaps = []
        self.seg_overlaps = []

        # bodies that touch edge of ROI
        self.boundarybodies = set()
        self.boundarybodies2 = set()

        self.voxelfilter = voxelfilter
        self.ptfilter = ptfilter
        self.num_displaybodies = num_displaybodies
        self.nogt = nogt
        self.selfcompare = selfcompare
        self.enable_sparse = enable_sparse
        self.subvolume_threshold = subvolume_threshold
        self.important_bodies = set(important_bodies)
        self.subvolsize = 0

    def compute_subvolume(self):
        """Performs metric computation for each stat and saves relevant state.

        Note: not every stat has state that needs to be
        computed at the substack level.  This should only be run
        when the number of subvolume ids is 1 as reduce should
        handle adjustments to the state.
        """
        
        assert len(self.subvolumes) == 1

        for stat in self.subvolume_stats:
            stat.compute_subvolume_before_remapping()

    def write_subvolume_stats(self):
        """For each stat, returns subvolume stats as an array.
        """

        # should actually be a subvolume
        assert len(self.subvolumes) == 1
        assert not self.disable_subvolumes

        # not all stats will support subvolume stats
        subvolumestats = []
        for stat in self.subvolume_stats:
            subvolumestats.extend(stat.write_subvolume_stats())

        # set flag to ignore by default in viewer
        for res in subvolumestats:
            res["ignore"] = self.ignore_subvolume

        return subvolumestats

    def write_summary_stats(self): 
        """For each stat, returns summary stats as an array.
        """
        summarystats = []
        for stat in self.subvolume_stats:
            summarystats.extend(stat.write_summary_stats())

        return summarystats

    def write_body_stats(self):
        """For each stat, returns body stats as an array.
        """
        bodystats = []
        for stat in self.subvolume_stats:
            bodystats.extend(stat.write_body_stats())

        return bodystats

    def write_bodydebug(self):
        """For each stat, returns various debug information.
        """
        debuginfo = []
        for stat in self.subvolume_stats:
            print(stat)
            debuginfo.extend(stat.write_bodydebug())
        return debuginfo


    def remap_stats(self):
        for iter1 in range(0, len(self.gt_overlaps)):
            self.gt_overlaps[iter1]._partial_remap()           
        for iter1 in range(0, len(self.seg_overlaps)):
            self.seg_overlaps[iter1]._partial_remap()           

    # drops subvolume stats and subvolume
    def merge_stats(self, subvolume, enablemetrics=True):
        assert(len(self.seg_overlaps) == len(subvolume.seg_overlaps))
        assert(len(self.gt_overlaps) == len(subvolume.gt_overlaps))
        for iter1 in range(0, len(self.gt_overlaps)):
            self.gt_overlaps[iter1].combine_tables(subvolume.gt_overlaps[iter1])           
        for iter1 in range(0, len(self.seg_overlaps)):
            self.seg_overlaps[iter1].combine_tables(subvolume.seg_overlaps[iter1])           

        assert(len(self.subvolume_stats) == len(subvolume.subvolume_stats))

        if enablemetrics:
            for iter1, val in enumerate(self.subvolume_stats):
                self.subvolume_stats[iter1].reduce_subvolume(subvolume.subvolume_stats[iter1])
            
        self.subvolumes.extend(subvolume.subvolumes)
       
        self.boundarybodies = self.boundarybodies.union(subvolume.boundarybodies)
        self.boundarybodies2 = self.boundarybodies2.union(subvolume.boundarybodies2)

        # enable subvolume computation if one of the internal subvolumes is enabled
        if self.ignore_subvolume:
            self.ignore_subvolume = subvolume.ignore_subvolume

    def add_gt_overlap(self, table):
        self.gt_overlaps.append(table)
    
    def add_seg_overlap(self, table):
        self.seg_overlaps.append(table)

    def add_stat(self, value):
        value.set_segstats(self)
        self.subvolume_stats.append(value)


