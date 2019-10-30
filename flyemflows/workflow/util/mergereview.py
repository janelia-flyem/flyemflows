import os
import zlib
import json
import logging
from functools import partial
from itertools import chain, starmap

import numpy as np
import pandas as pd
import networkx as nx

from dvidutils import LabelMapper

from neuclease.util import Timer, tqdm_proxy, swap_df_cols, compute_parallel
from neuclease.dvid import (fetch_instance_info, fetch_mapping, fetch_labels_batched, determine_bodies_of_interest,
                            fetch_combined_roi_volume, determine_point_rois)

from neuclease.mergereview.mergereview import generate_mergereview_assignments

from flyemflows.workflow.findadjacencies import select_closest_edges

logger = logging.getLogger(__name__)


def extract_assignment_fragments( server, uuid, syn_instance,
                                  edge_table,
                                  boi_rois=None,
                                  min_tbars_in_roi=2,
                                  min_psds_in_roi=10,
                                  fragment_rois=None,
                                  processes=16,
                                  *,
                                  synapse_table=None,
                                  boi_table=None,
                                  seg_instance=None,
                                  update_edges=False ):
    """
    Using the edge table emitted from the FindAdjacencies workflow,
    emit a table of "fragments" (sets of bodies) which connect two
    "bodies of interest" (BOIs, described below).
    
    The emitted fragments be used to generate
    focused assignments and/or merge review assignments.
    
    Essentially, we construct an adjacency graph from the edge table,
    and then search for any paths that can connect two BOIs:
    
        BOI - b - b - b - ... - b - BOI
    
    The path from one BOI to another is called a "fragment".
    
    If the path contains only the two BOIs and no other bodies, then 
    the two BOIs are directly adjacent, with no intervening bodies:
    
        BOI - BOI

    In those cases, it is possible to create a "focused proofreading" task
    from the body pair.  In all other cases, you can create a "merge review"
    task for the fragment.  See the following functions:
    
        generate_mergereview_assignments_from_df()
        neuclease.focused.asssignments.generate_focused_assignments()
    
    Exactly which bodies are considered "bodies of interest" is determined
    by the presence of T-bars and PSDs within the specified ROIs (boi_rois,
     if provided), thresholded by the given criteria.  If no boi_rois are
    specified, then all T-bars and PSDs in the given bodies are counted.
    
    Additionally, the final fragment set can be filtered to exclude
    fragments that travel outside of a given list of ROIs.

    See the explanation of the edge_table parameter for an explanation of
    the FindAdjacencies output.

    Tip:
        To visualize the adjacency graph for a subset of rows in either
        the input edge table or the output tables, see display_graph(), below.

    Args:
        server, uuid, syn_instance:
            DVID synapse (annotation) instance
    
        edge_table:
            A DataFrame as explained below, or a filepath to a
            .npy file that can be loaded into one.
        
            The FindAdjacencies workflow finds the sites at which
            preselected bodies are adjacent to one another.
            
            The user provides a list of body "groups" which are analyzed independently.
            In addition to "direct" adjacencies between touching bodies (distance=1.0),
            the workflow can be configured to also search for near-adjacencies,
            in which bodies come close to each other without physically touching (distance > 1.0).
            Each adjacency is referred to as an edge, and the results are emitted as
            an "edge table" with the following columns:
        
                [label_a, label_b, za, ya, xa, zb, yb, xb, distance, group, group_cc]
        
            with the following definitions:
            
                label_a, label_b:
                    Body IDs (assuming the FindAdjacencies workflow was executed on a body input source)
    
                 za, ya, xa, zb, yb, xb:
                    Coordinates that fall within the body on each side of the edge.
    
                distance:
                    The euclidean distance between the two coordinates.
                    For "direct" adjacencies, distance is always 1.0.
                    For "nearby" adjacencies, distance is always > 1.0.
    
                group:
                    The original body groups the user selected for adjacency analysis.
                    The exact group ID values are arbitrary (not necessarily consecutive),
                    and were provided by the user that ran the FindAdjacencies workflow.
                    Note that one body may exist in more than one group.
                
                group_cc:
                    An independent subgraph is constructed for each group (from the group's 'edges').
                    A connected components analysis is then performed on each subgraph,
                    and a unique ID is assigned to each CC.
    
                    Although the connected components are computed on each group in isolation,
                    the assigned group_cc values are unique across all of the groups in the table.
    
                    The group_cc values are otherwise arbitrary. (That is, they aren't necessarily
                    consecutive, or related to the other CC IDs in their group.)
                    For example, group 123 might be found to contain two connected components,
                    labeled group_cc=53412 and group_cc=82344

        boi_rois:
            Optional.  List of ROI instance names.
            If provided, only T-bars and PSDs that fall within the given list of ROIs will be
            counted when determining which bodies are considered BOIs.  Otherwise, all synapses
            in the volume are considered.
        
        min_tbars_in_roi, min_psds_in_roi:
            The criteria for determining what counts as a BOI.
            As indicated in the argument names, only synapse points WITHIN the ROI(s)
            will be counted towards these requirements. 
        
        fragment_rois:
            Optional.  Any fragments that extend outside of the given list of ROIs
            will be discarded from the result, even though they contained BOIs
            that matched the BOI criteria.
        
        processes:
            Various steps in this function can be parallelized.
            This specifies how much parallelism to use.
        
        synapse_table:
            Optional.  If you already fetched the synapses from DVID
            (via fetch_synapses_in_batches() or fetch_roi_synapses()),
            you can provide it here (or a file path to a stored .npy file),
            in which case this function will not need to fetch the synapses from DVID.
            (Not needed at all if you're providing your own boi_table.)
        
        boi_table:
            Optional.
            Normally this function computes the boi_table directly from the synapse points,
            but if you already have it handy, you can pass it in here.
            It will still be filtered according to min_tbars_in_roi and min_psds_in_roi,
            so the BOIs used will be accurate as long as the table contains all of the BOIs
            you might be interested in, or more.
        
        seg_instance:
            By default, this BOIs in this table will be extracted from the segmentation
            instance that is associated with the given synapse annotation instance.
            But if you would like to use a different segmentation instance, provide it here.
        
        update_edges:
            If True, re-fetch the body label under each coordinate in the table,
            and re-select the "best" (most central) edge for body pairs with multiple edges.
            This takes a while to run. It's only necessary if your edge table is likely to
            be out-of-date with respect to the given UUID.
    
    Returns:
        (focused_fragments_df, mr_fragments_df, bois), where:
        
        focused_fragments_df:
            A DataFrame consisting of rows suitable for "focused proofreading",
            i.e. every row (edge) is a single-edge fragment.
        
        mr_fragments_df:
            A DataFrame consisting of edges that belong to fragments with more
            than one edge, meaning they are not suitable for "focused proofreading"
            and are instead suitable for "merge review".
            The fragment IDs are the (group_cc, cc_task) columns.
            Edges with the same fragment ID should be grouped together into the
            same merge review task.
        
        mr_endpoint_df:
            A DataFrame containing only the 'endpoint' bodies of the MR fragments,
            one pair per row.
            The columns in which the bodies are found (a vs b) will not be the same
            as they appear in mr_fragments_df, but the group_cc and cc_task columns
            will correspond to the appropriate rows in the full DataFrame.
            This 'endpoint' dataframe does not contain enough information to create
            merge review tasks (it lacks information about the intermediate bodies
            that connect the two endpoints) but it is more convenient to analyze
            when computing certain statistics to describe the types of merge review
            tasks that were found.
        
        boi_table:
            A DataFrame containing the BOIs (based on the criteria given above)
            that were used to selecting fragments, indexed by body, with
            columns ['PreSyn', 'PostSyn'].
            (See ``neuclease.dvid.annotation.determine_bodies_of_interest()``.)
            Note that the returned fragments do not necessarily cover
            every BOI in this list.
    """
    if isinstance(boi_rois, str):
        boi_rois = [boi_rois]

    if isinstance(fragment_rois, str):
        fragment_rois = [fragment_rois]
    
    if seg_instance is None:
        syn_info = fetch_instance_info(server, uuid, syn_instance)
        seg_instance = syn_info["Base"]["Syncs"][0]

    ref_seg = (server, uuid, seg_instance)
    
    # Load edges (if necessary), pre-filter, normalize
    edges_df = load_edges(edge_table)
    
    if update_edges:
        # Update the table for consistency with the given UUID,
        # and re-post-process it to find the correct "central" and "closest" edges,
        # (in case some groups were merged).
        edges_df = update_localized_edges(*ref_seg, edges_df, processes)

    # Technically, you could provide 0 for either of these,
    # but that's probably a mistake on your part.
    # (Unless you specifically appended some 0-synapse bodies to your
    # synapse table, and expect those to be considered BOIs.)
    assert min_tbars_in_roi >= 1 and min_psds_in_roi >= 1

    if boi_table is not None:
        boi_table = boi_table.query('PreSyn >= @min_tbars_in_roi or PostSyn >= @min_psds_in_roi')
    else:
        assert not boi_rois, \
            "You can't specify boi_rois if you're providing your own boi_table"
        
        # Fetch synapse labels and determine the set of BOIs
        boi_table = determine_bodies_of_interest( server, uuid, syn_instance,
                                                  boi_rois,
                                                  min_tbars_in_roi, min_psds_in_roi,
                                                  processes,
                                                  synapse_table=synapse_table )
    
    assert boi_table.index.name == 'body'
    assert set(boi_table.columns) == {'PreSyn', 'PostSyn'}

    bois = set(boi_table.index)

    # We're trying to connect BOIs to each other.
    # Therefore, we're not interested in groups of bodies
    # that don't contain at least 2 BOIs. 
    edges_df = filter_groups_for_min_boi_count(edges_df, bois, ['group_cc'], 2)

    # Find the paths ('fragments', a.k.a. 'tasks') that connect BOIs within each group.
    fragment_edges_df = compute_fragment_edges(edges_df, bois, processes)

    if fragment_rois is not None:
        # Drop fragments that extend outside of the specified ROIs.
        fragment_edges_df = filter_fragments_for_roi(server, uuid, fragment_rois, fragment_edges_df)

    # If a group itself contained multiple CCs, it's possible that the BOIs were separated
    # into separate tasks, meaning that each individual task no longer satisfies the 2-BOI requirement.
    # Refilter.
    fragment_edges_df = filter_groups_for_min_boi_count(fragment_edges_df, bois, ['group_cc', 'cc_task'], 2)

    # Fetch the supervoxel IDs for each edge.
    with Timer("Sampling supervoxel IDs", logger):
        points_a = fragment_edges_df[['za', 'ya', 'xa']].values
        points_b = fragment_edges_df[['zb', 'yb', 'xb']].values
        fragment_edges_df['sv_a'] = fetch_labels_batched(*ref_seg, points_a, True, processes=processes)
        fragment_edges_df['sv_b'] = fetch_labels_batched(*ref_seg, points_b, True, processes=processes)
    
    # Divide into 'focused' and 'merge review' fragments,
    # i.e. single-edge fragments and multi-edge fragments
    focused_fragments_df = (fragment_edges_df
                               .groupby(['group_cc', 'cc_task'])
                               .filter(lambda task_df: len(task_df) == 1) # exactly one edge
                               .copy())

    mr_fragments_df = (fragment_edges_df
                          .groupby(['group_cc', 'cc_task'])
                          .filter(lambda task_df: len(task_df) > 1) # multiple edges
                          .copy())

    num_focused_fragments = len(focused_fragments_df)
    num_mr_fragments = len(mr_fragments_df.drop_duplicates(['group_cc', 'cc_task']))
    fragment_bodies = pd.unique(fragment_edges_df[['label_a', 'label_b']].values.reshape(-1))
    num_fragment_bois = len(set(fragment_bodies).intersection(set(boi_table.index)))
    
    logger.info(f"Emitting {num_focused_fragments} focused fragments and "
                f"{num_mr_fragments} merge-review fragments, "
                f"covering {num_fragment_bois} BOIs out of {len(boi_table)}.")

    with Timer("Merging synapse counts onto results", logger):
        focused_fragments_df = focused_fragments_df.merge( boi_table, 'left', left_on='label_a', right_index=True )
        focused_fragments_df = focused_fragments_df.merge( boi_table, 'left', left_on='label_b', right_index=True,
                                                           suffixes=('_a', '_b') )

        mr_fragments_df = mr_fragments_df.merge( boi_table, 'left', left_on='label_a', right_index=True )
        mr_fragments_df = mr_fragments_df.merge( boi_table, 'left', left_on='label_b', right_index=True,
                                                 suffixes=('_a', '_b') )
    
    with Timer("Constructing merge-review 'endpoint' dataframe", logger):
        try:
            mr_endpoint_df = construct_mr_endpoint_df(mr_fragments_df, bois)
        except BaseException as ex:
            logger.error(str(ex))
            logger.error("Failed to construct the merge-review 'endpoint' dataframe.  Returning None.")
            mr_endpoint_df = None
    
    return focused_fragments_df, mr_fragments_df, mr_endpoint_df, boi_table


def generate_mergereview_assignments_from_df_OLD(server, uuid, instance, mr_fragments_df, bois, assignment_size, output_dir):
    # Sort table by task size (edge count)
    group_sizes = mr_fragments_df.groupby(['group_cc', 'cc_task']).size().rename('group_size')
    mr_fragments_df = mr_fragments_df.merge(group_sizes, 'left', left_on=['group_cc', 'cc_task'], right_index=True)
    mr_fragments_df = mr_fragments_df.sort_values(['group_size', 'group_cc', 'cc_task'])

    # Group assignments by task size and emit an assignment for each group
    assignments = {}
    for group_size, same_size_tasks_df in mr_fragments_df.groupby('group_size'):
        sv_groups = {}
        for task_index, ((_cc, _cc_task), task_df) in enumerate(same_size_tasks_df.groupby(['group_cc', 'cc_task']), start=1):
            sv_groups[task_index] = pd.unique(task_df[['sv_a', 'sv_b']].values.reshape(-1))

        num_bodies = group_size+1
        logger.info(f"{len(sv_groups)} tasks of size {num_bodies}")
        output_subdir = f'{output_dir}/{num_bodies:02}-bodies'
        assignments[num_bodies] = generate_mergereview_assignments(server, uuid, instance, sv_groups, bois, assignment_size, output_subdir)

    return assignments


def generate_mergereview_assignments_from_df(server, uuid, instance, mr_fragments_df, bois, assignment_size, output_dir):
    """
    Generate a set of assignments for the given mergereview fragments.
    The assignments are written to a nested hierarchy:
    Grouped first by task size (number of bodies in each task),
    and then grouped in batches of N tasks (assignment_size).
    
    The body IDs emitted in the assignments and their classification as "BOI"
    or not is determined by fetching the mappings for each supervoxel in the dataframe.
    """
    # Sort table by task size (edge count)
    group_sizes = mr_fragments_df.groupby(['group_cc', 'cc_task']).size().rename('group_size')
    mr_fragments_df = mr_fragments_df.merge(group_sizes, 'left', left_on=['group_cc', 'cc_task'], right_index=True)
    mr_fragments_df = mr_fragments_df.sort_values(['group_size', 'group_cc', 'cc_task'])

    mr_fragments_df['body_a'] = fetch_mapping(server, uuid, instance, mr_fragments_df['sv_a'])
    mr_fragments_df['body_b'] = fetch_mapping(server, uuid, instance, mr_fragments_df['sv_b'])
    
    mr_fragments_df['is_boi_a'] = mr_fragments_df.eval('body_a in @bois')
    mr_fragments_df['is_boi_b'] = mr_fragments_df.eval('body_b in @bois')
    
    # Group assignments by task size and emit an assignment for each group
    all_tasks = {}
    for group_size, same_size_tasks_df in mr_fragments_df.groupby('group_size'):
        group_tasks = []
        for (group_cc, cc_task), task_df in same_size_tasks_df.groupby(['group_cc', 'cc_task']):
            svs = pd.unique(task_df[['sv_a', 'sv_b']].values.reshape(-1))
            svs = np.sort(svs)
            
            boi_svs  = set(task_df[task_df['is_boi_a']]['sv_a'].tolist())
            boi_svs |= set(task_df[task_df['is_boi_b']]['sv_b'].tolist())
            
            task_bodies = pd.unique(task_df[['body_a', 'body_b']].values.reshape(-1)).tolist()
            
            task = {
                # neu3 fields
                'task type': "merge review",
                'task id': hex(zlib.crc32(svs)),
                'supervoxel IDs': svs.tolist(),
                'boi supervoxel IDs': sorted(boi_svs),
                
                # Encode edge table as json
                "supervoxel IDs A": task_df['sv_a'].tolist(),
                "supervoxel IDs B": task_df['sv_b'].tolist(),
                "supervoxel points A": task_df[['xa', 'ya', 'za']].values.tolist(),
                "supervoxel points B": task_df[['xb', 'yb', 'zb']].values.tolist(),
                
                # Debugging fields
                'group_cc': int(group_cc),
                'cc_task': int(cc_task),
                'original_bodies': sorted(task_bodies),
                'total_body_count': len(task_bodies),
                'original_uuid': uuid,
            }
            group_tasks.append(task)

        num_bodies = group_size+1
        all_tasks[num_bodies] = group_tasks

    # Now that the task json data has been generated and split into groups (by body count),
    # write them into multiple directories (one per group), each of which has muliple files
    # (one per task batch, as specified by assignment_size)
    for num_bodies, group_tasks in all_tasks.items():
        output_subdir = f'{output_dir}/{num_bodies:02}-bodies'
        os.makedirs(output_subdir, exist_ok=True)
        for i, batch_start in enumerate(tqdm_proxy(range(0, len(group_tasks), assignment_size), leave=False)):
            output_path = f"{output_dir}/{num_bodies:02}-bodies/assignment-{i:04d}.json"

            batch_tasks = group_tasks[batch_start:batch_start+assignment_size]
            assignment = {
                "file type":"Neu3 task list",
                "file version":1,
                "task list": batch_tasks
            }

            with open(output_path, 'w') as f:
                #json.dump(assignment, f, indent=2)
                pretty_print_assignment_json_items(assignment.items(), f)
    
    return all_tasks


def generate_typereview_assignment(server, uuid, instance, df, output_path, comment='', generate_coords=False, specify_largest_sv=False, processes=8):
    """
    Generate a mergereview-like assignment for doing type comparison.
    
    By default
    """
    df = df.copy()
    assert df.columns.tolist() == ["body_a", "body_b", "score"]
    df['body_a'] = df['body_a'].astype(np.uint64)
    df['body_b'] = df['body_b'].astype(np.uint64)
    
    tasks = []
    for row in tqdm_proxy(df.itertuples(), total=len(df)):
        _index, task = _prepare_task( uuid,
                                      row.Index,
                                      row.body_a, row.body_b,
                                      row.score,
                                      comment )
        tasks.append(task)

    assignment = {
        "file type":"Neu3 task list",
        "file version":1,
        "task list": tasks
    }

    with open(output_path, 'w') as f:
        json.dump(assignment, f, indent=2)
        #pretty_print_assignment_json_items(assignment.items(), f)


def _prepare_task(uuid, index, body_a, body_b, score, comment=''):    
    task = {
        # neu3 fields
        'task type': "type review",
        'task result id': f"{body_a}_{body_b}",
        
         "body ID A": body_a,
         "body ID B": body_b,
        
        # Debugging fields
        'debug': {
            'original_uuid': uuid,
            'match_score': float(score),
            'comment': comment
        }
    }
    return (index, task)


def pretty_print_assignment_json_items(items, f, cur_indent=0):
    """
    Python's standard pretty-print is quite ugly if your data involves lots of lists.
    This function is hand-tuned to write a json assignment in a reasonably pretty way.
    """
    f.write(' '*cur_indent + '{\n')
    items = list(items)

    cur_indent += 2
    for i, (k,v) in enumerate(items):
        f.write(' '*cur_indent + f'"{k}": ')

        if k == 'task list':
            f.write('\n')
            cur_indent += 2
            f.write(' '*cur_indent + '[\n')
            cur_indent += 2
            for task_index, task in enumerate(v):
                pretty_print_assignment_json_items(task.items(), f, cur_indent)
                if task_index != len(v)-1:
                    f.write(',')
                f.write('\n')
            cur_indent -= 2
            f.write(' '*cur_indent + ']\n')
            cur_indent -= 2
        elif isinstance(v, str):
            f.write(f'"{v}"')
        elif isinstance(v, (int, float)):
            f.write(str(v))
        elif isinstance(v, list):
            json.dump(v, f)
        else:
            raise AssertionError(f"Can't pretty-print arbitrary values of type {type(v)}")
        
        if i != len(items)-1:
            f.write(',')

        f.write('\n')

    cur_indent -= 2
    f.write(' '*cur_indent + '}')
    if cur_indent == 0:
        f.write('\n')
    
    
def load_edges(edge_table):
    """
    - Read the edges from a file (if necessary)
    - Verify that the necessary columns are present
    - Drop any that were marked with group_cc == -1
      (they aren't part of any group_cc)
    - Normalize columns so that label_a < label_b
    """
    if isinstance(edge_table, str):
        assert edge_table.endswith('.npy')
        edges_df = pd.DataFrame(np.load(edge_table))
    else:
        assert isinstance(edge_table, pd.DataFrame)
        edges_df = edge_table

    required_cols = ['label_a', 'label_b', 'za', 'ya', 'xa', 'zb', 'yb', 'xb', 'distance', 'group', 'group_cc']
    assert set(edges_df.columns).issuperset(required_cols)
    
    # Edges with group_cc == -1 are 'nearby' edges that are too distant to be included in any group_cc.
    # The FindAdjacencies workflow emits them, but we don't want to use them.
    edges_df = edges_df.query('group_cc != -1').copy()
    
    swap_df_cols(edges_df, None, edges_df.eval('label_a > label_b'), ('a', 'b'))
    assert len(edges_df.query('label_a == label_b')) == 0
    assert edges_df.duplicated(['group', 'label_a', 'label_b']).sum() == 0
    return edges_df


def update_localized_edges(server, uuid, seg_instance, edges_df, processes=16):
    """
    Use the coordinates in the edge table to update the label_a/label_b
    columns (by fetching the labels from dvid at the given UUID).
    
    Then, since the labels MAY have changed, re-compute the central-most
    edges (for "direct" adjacencies) and closest-approaching edges (for
    nearby "adjacencies").  This takes a few minutes.
    """
    ref_seg = (server, uuid, seg_instance)

    # Update to latest node
    with Timer(f"Updating body labels for uuid {uuid[:4]}", logger):
        edges_df['label_a'] = fetch_labels_batched(*ref_seg, edges_df[['za', 'ya', 'xa']].values, processes=processes)
        edges_df['label_b'] = fetch_labels_batched(*ref_seg, edges_df[['zb', 'yb', 'xb']].values, processes=processes)
    swap_df_cols(edges_df, None, edges_df.eval('label_a > label_b'), ('a', 'b'))

    # Discard already-merged edges
    edges_df = edges_df.query('label_a != label_b')

    # Now that we've relabeled some points, there may be duplicate edges in the table.
    # De-duplicate them by choosing the best ones.
    # (This takes a while)

    with Timer("Re-selecting central-most direct edges", logger):
        direct_edges_df = edges_df.loc[edges_df['distance'] == 1.0].copy()
        
        # If we really want to choose the *best* edge, we should do a proper centroid calculation.
        # But that takes a long time, and there aren't likely to be all that many cases where it makes a difference.
        #direct_edges_df = select_central_edges(direct_edges_df)
        
        # Instead, just drop duplicates in arbitrary order.
        direct_edges_df.drop_duplicates(['group', 'label_a', 'label_b'], inplace=True)

    with Timer("Re-selecting closest-approaching nearby edges", logger):
        # This doesn't take as long, partly because there are
        # usually fewer nearby edges than direct edges.
        nearby_edges_df = edges_df.loc[edges_df['distance'] >= 1.0]
        nearby_edges_df = select_closest_edges(nearby_edges_df)

    # FIXME: Should we also update the group_cc?
    # If any splits have occurred, I guess the group_cc is no longer a single component.
    # Bu when we analyze it for 'fragments', the results will be correct.
    # append_group_ccs(...)

    # Combine (direct first)
    edges_df = pd.concat((direct_edges_df, nearby_edges_df))
    
    # After updating, it's technically possible that a nearby
    # edge now has the same labels as a direct edge.
    # Drop duplicates so we keep only the direct edge.
    edges_df = edges_df.drop_duplicates(['group', 'label_a', 'label_b'], keep='first')

    return edges_df


def filter_groups_for_min_boi_count(edges_df, bois, group_columns=['group_cc'], min_boi_count=2):
    """
    Group the given dataframe according to the columns listed in `group_columns`,
    and count how many BOIs exist in each group.
    
    Then drop rows from the original dataframe if the group they belong to didn't have enough BOIs.
    """
    with Timer("Filtering out groups with too few BOIs", logger):
        bois = np.fromiter(bois, np.uint64)
        bois.sort()
        assert isinstance(group_columns, (list, tuple))
    
        boi_counts_df = edges_df[['label_a', 'label_b', *group_columns]].copy()
        boi_counts_df['is_boi_a'] = boi_counts_df.eval('label_a in @bois')
        boi_counts_df['is_boi_b'] = boi_counts_df.eval('label_b in @bois')
        boi_counts_df['boi_count'] = boi_counts_df['is_boi_a'].astype(int) + boi_counts_df['is_boi_b'].astype(int)
        
        group_boi_counts = boi_counts_df.groupby(group_columns)['boi_count'].agg('sum')
        group_boi_counts = group_boi_counts[group_boi_counts >= min_boi_count]
    
        kept_groups_df = group_boi_counts.reset_index()[[*group_columns]]
        logger.info(f"Keeping {len(kept_groups_df)} groups ({group_columns}) out of {len(boi_counts_df)}")
    
        edges_df = edges_df.merge(kept_groups_df, 'inner', on=group_columns)
    return edges_df


def compute_fragment_edges(edges_df, bois, processes=0):
    """
    For each edge group, search for paths that can connect the BOIs in the group.
    Each group is a "fragment", a.k.a. "task".
    Return a new edge DataFrame, where each edge is associated with a group and
    a fragment within that group, indicated by group_cc and cc_task, respectively.
    
    Args:
        edges_df:
            An edge table as described in extract_assignment_fragments(), above,
            with the additional requirement that the table is in "normalized" form,
            i.e. label_a < label_b.

        bois:
            List of BOIs
    """
    fragments = extract_fragments(edges_df, bois, processes)
    
    with Timer("Extracting edges for each fragment from full table", logger):
        edges_df = edges_df.query('group_cc in @fragments.keys()')

        cc_col = []
        task_col = []
        frag_cols = []
        for group_cc, group_fragments in fragments.items():
            for task_index, frag in enumerate(group_fragments):
                cc_col.extend( [group_cc]*(len(frag)-1) )
                task_col.extend( [task_index]*(len(frag)-1) )
                frag_edges = list(zip(frag[:-1], frag[1:]))
                frag_cols.extend(frag_edges)

        frag_cols = np.array(frag_cols, dtype=np.uint64)
        frag_cols.sort(axis=1)

        fragment_edges_df = pd.DataFrame(frag_cols, columns=['label_a', 'label_b'])
        fragment_edges_df['group_cc'] = cc_col
        fragment_edges_df['cc_task'] = task_col
        
        fragment_edges_df = fragment_edges_df.merge(edges_df, 'left', ['group_cc', 'label_a', 'label_b'])
        return fragment_edges_df


def _extract_group_fragment_edges(frag_list, group_df):
    """
    Helper for compute_fragment_edges(), above.
    """
    fragment_edges_dfs = []
    for task_index, frag in enumerate(frag_list):
        frag_edges = list(zip(frag[:-1], frag[1:]))
        frag_edges = np.sort(frag_edges, axis=1)

        frag_edges_df = pd.DataFrame(frag_edges, columns=['label_a', 'label_b'])
        frag_edges_df = frag_edges_df.merge(group_df, 'left', ['label_a', 'label_b'])
        frag_edges_df['cc_task'] = task_index
        fragment_edges_dfs.append(frag_edges_df)
    return fragment_edges_dfs


def extract_fragments(edges_df, bois, processes):
    """
    For each connected component group (pre-labeled) in the given DataFrame,
    Search for paths that can connect the groups's BOIs to each other,
    possibly passing through non-BOI nodes in the group.
    
    Returns:
        dict {group_cc: [fragment, fragment, ...]}
        where each fragment is a tuple of N body IDs which form a path of
        adjacent bodies, with a BOI on each end (first node/last node) of
        the path, and non-BOIs for the intermediate nodes (if any).
    """
    assert isinstance(bois, set)
    assert edges_df.duplicated(['group', 'label_a', 'label_b']).sum() == 0
    
    def _prepare_group(group_cc, cc_df):
        group_bois = bois & set(cc_df[['label_a', 'label_b']].values.reshape(-1))
        return group_cc, cc_df, group_bois
        
    with Timer("Extracting fragments from each group", logger):
        num_groups = edges_df['group_cc'].nunique()
        group_and_bois = starmap(_prepare_group, edges_df.groupby('group_cc'))

        cc_and_frags = compute_parallel( extract_fragments_for_cc, group_and_bois, 1000,
                                         processes=processes, ordered=False, leave_progress=True,
                                         total=num_groups, starmap=True )

    fragments = dict(cc_and_frags)
    num_fragments = sum(len(frags) for frags in fragments.values())
    logger.info(f"Extracted {num_fragments} fragments")
    return fragments


def extract_fragments_for_cc(group_cc, cc_df, bois):
    g = prepare_graph(cc_df, bois)
    nodes = pd.unique(cc_df[['label_a', 'label_b']].values.reshape(-1))
    boi_nodes = {*filter(lambda n: g.nodes[n]['boi'], nodes)}
    fragments = []

    for n in boi_nodes:
        paths = nx.single_source_dijkstra_path(g,n, weight='distance')
        frags = [paths[target] for target in paths.keys() if target in boi_nodes]
        frags = [*filter(lambda frag: len(frag) > 1, frags)]
        frags = [*filter(lambda frag: not boi_nodes.intersection(frag[1:-1]), frags)]
        assert all([frag[0] in boi_nodes and frag[-1] in boi_nodes for frag in frags])
        fragments.extend(frags)

    for f in fragments:
        assert f[0] in boi_nodes and f[-1] in boi_nodes
        if f[0] > f[-1]:
            f[:] = f[::-1]

    fragments = {tuple(frag) for frag in fragments}
    return group_cc, fragments


def construct_mr_endpoint_df(mr_fragments_df, bois):
    """
    Each merge-review task group contains exactly two endpoint nodes.
    Locate each endpoint pair and return it in a DataFrame
    (with _a/_b columns as if it were an ordinary task dataframe).
    """
    assert isinstance(bois, set)
    mr_fragments_df = mr_fragments_df.copy()
    
    # Make sure our two BOI endpoints are always in body_a
    #_a_is_small = mr_fragments_df.eval('label_a not in @bois')
    
    # This is much faster than the above .eval() call if bois is large.
    _a_is_small = [(label_a not in bois) for label_a in mr_fragments_df['label_a']]
    _a_is_small = pd.Series(_a_is_small, index=mr_fragments_df.index)
    swap_df_cols(mr_fragments_df, None, _a_is_small, ('a', 'b'))
    
    # edge_area ends in 'a', which is inconvenient
    # for the column selection below,
    # and we don't need it anyway.  Drop it.
    fmr_df = mr_fragments_df.drop(columns=['edge_area'])
    
    # All columns ending in 'a'.
    cols_a = [col for col in fmr_df.columns if col.endswith('a')]
    
    num_tasks = len(mr_fragments_df.drop_duplicates(['group_cc', 'cc_task']))
    post_col = fmr_df[cols_a].columns.tolist().index('PostSyn_a')

    filtered_mr_endpoints = []
    for (group_cc, cc_task), task_df in tqdm_proxy(fmr_df.groupby(['group_cc', 'cc_task']), total=num_tasks):
        #assert task_df.eval('label_b not in @_bois').all()

        # Find the two rows that mention a BOI
        #selected_df = task_df.query('label_a in @_bois')
        
        # Apparently this is MUCH faster than .query() when bois is large
        _a_is_big = [(label_a in bois) for label_a in task_df['label_a']]
        selected_df = task_df.iloc[_a_is_big]
        
        selected_df = selected_df[cols_a]
        assert len(selected_df) == 2
    
        stats_a, stats_b = list(selected_df.itertuples(index=False))

        # Put the big body in the 'a' position.
        if stats_a[post_col] < stats_b[post_col]:
            stats_a, stats_b = stats_b, stats_a
    
        filtered_mr_endpoints.append( (group_cc, cc_task, len(task_df), *stats_a, *stats_b) )
    
    cols_b = [col[:-1] + 'b' for col in cols_a]
    combined_cols = ['group_cc', 'cc_task', 'num_edges', *cols_a, *cols_b]
    mr_endpoints_df = pd.DataFrame(filtered_mr_endpoints, columns=combined_cols)
    
    final_cols = ['group_cc', 'cc_task', 'num_edges', *sorted(combined_cols[3:])]
    mr_endpoints_df = mr_endpoints_df[final_cols]
    
    return mr_endpoints_df


def prepare_graph(cc_df, bois, consecutivize=False):
    """
    Create networkx Graph for the given edges,
    and annotate each node with its body ID and whether
    or not it is a 'boi' (body of interest).
    
    If consecutivize=True, do not use body IDs as the node IDs.
    Instead, use consecutive integers 1..N as the node IDs.
    This is useful when vizualizing the graph with hvplot,
    as a workaround for the following issue:
    https://github.com/pyviz/hvplot/issues/218
    """
    if isinstance(bois, pd.DataFrame):
        boi_df = bois
        assert boi_df.index.name == 'body'
        bois = boi_df.index
    else:
        boi_df = None

    if 'body_a' in cc_df.columns and 'body_b' in cc_df.columns:
        edges = cc_df[['body_a', 'body_b']].values
    elif 'label_a' in cc_df.columns and 'label_b' in cc_df.columns:
        edges = cc_df[['label_a', 'label_b']].values
    else:
        raise RuntimeError("Could not find label or body columns.")
    
    # Pre-sort nodes to avoid visualization issues such as:
    # https://github.com/pyviz/hvplot/issues/223
    bodies = np.sort(pd.unique(edges.reshape(-1)))
    nodes = bodies

    if consecutivize:
        assert bodies.dtype == edges.dtype == np.uint64
        nodes = np.arange(1, 1+len(bodies), dtype=np.uint32)
        mapper = LabelMapper(bodies, nodes)
        edges = mapper.apply(edges)

    g = nx.Graph()
    g.add_nodes_from(nodes.astype(np.int64))
    for (a, b), distance in zip(edges, cc_df['distance']):
        g.add_edge(a, b, distance=distance)

    for node, body in zip(nodes, bodies):
        g.nodes[node]['body'] = body
        g.nodes[node]['boi'] = (body in bois)

        # Append more node metadata if it's available.
        if boi_df is None:
            continue

        for col in boi_df.columns:
            if body in bois:
                g.nodes[node][col] = boi_df.loc[body, col]
            else:
                g.nodes[node][col] = -1
    return g


def filter_fragments_for_roi(server, uuid, fragment_rois, fragment_edges_df):
    """
    If any edge in a fragment falls outside of the given ROIs,
    discard the entire fragment.
    """
    edge_points_a = fragment_edges_df[['za', 'ya', 'xa']].rename(columns={'za': 'z', 'ya': 'y', 'xa': 'x'})
    edge_points_b = fragment_edges_df[['zb', 'yb', 'xb']].rename(columns={'zb': 'z', 'yb': 'y', 'xb': 'x'})

    roi_vol, roi_box, _ = fetch_combined_roi_volume(server, uuid, fragment_rois)
    determine_point_rois(server, uuid, fragment_rois, edge_points_a, roi_vol, roi_box)
    determine_point_rois(server, uuid, fragment_rois, edge_points_b, roi_vol, roi_box)
    
    fragment_edges_df['in_roi_a'] = (edge_points_a['roi_label'] == 1)
    fragment_edges_df['in_roi_b'] = (edge_points_b['roi_label'] == 1)

    # Note: There is a MUCH faster way to do this filter via groupby(...).all()
    #       followed by an inner merge, but that's more complicated and I'm not bothering.
    fragment_edges_df = (fragment_edges_df.groupby(['group_cc', 'cc_task'])
                         .filter(lambda df: df['in_roi_a'].all() and df['in_roi_b'].all()))
    return fragment_edges_df


def display_graph(cc_df, bois, width=500, hv=True, with_labels=True, big_bois=None):
    """
    Load the given edges into a graph and display it.
    Node colors distinguish between BOIs and non-BOIs.
    If hv=True, display with hvplot.
    Otherwise, display using networkx (matplotlib).
    """
    if hv:
        import holoviews as hv
        hv.extension('bokeh')
        import hvplot.networkx as hvnx
        draw = hvnx.draw
        g = prepare_graph(cc_df, bois, True)
    else:
        draw = nx.draw
        g = prepare_graph(cc_df, bois, False)

    node_colors = []
    for n in g.nodes:
        body = g.nodes[n]['body']
        if big_bois is not None and body in big_bois:
            node_colors.append('yellow')
        elif g.nodes[n]['boi']:
            node_colors.append('limegreen')
        else:
            node_colors.append('skyblue')

    edge_color = ['skyblue' if g.edges[a,b]['distance'] == 1.0 else 'red' for (a,b) in g.edges]

    if with_labels:
        labels = {n: str(g.nodes[n]['body']) for n in g.nodes}
    else:
        labels = None
    
    print(f"Drawing {len(g.nodes)} nodes and {len(g.edges)} edges")

    p = draw(g, node_color=node_colors, edge_color=edge_color, labels=labels, pos=nx.kamada_kawai_layout(g), with_labels=with_labels)
    
    if hv:
        p = p.opts(plot=dict(width=width, height=width))
    return p



if __name__ == "__main__":
    df = pd.DataFrame(np.load('/tmp/philip_small_not_tiny_df.npy', allow_pickle=True))
    df['body_a'] = fetch_mapping('emdata4:8900', '28e6', 'segmentation', df['sv_a'])
    df['body_b'] = fetch_mapping('emdata4:8900', '28e6', 'segmentation', df['sv_b'])
    bois = set(df[['body_a', 'body_b']].values.reshape(-1))
    generate_mergereview_assignments_from_df('emdata4:8900', '28e6', 'segmentation', df, bois, 10, '/tmp/philip-assignments')

