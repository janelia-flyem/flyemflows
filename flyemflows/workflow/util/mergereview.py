import logging

import numpy as np
import pandas as pd
import networkx as nx

from dvidutils import LabelMapper

from neuclease.util import Timer, tqdm_proxy, swap_df_cols
from neuclease.dvid import (fetch_instance_info, fetch_labels_batched, determine_bodies_of_interest,
                            fetch_combined_roi_volume, determine_point_rois)

from neuclease.mergereview.mergereview import generate_mergereview_assignments

from flyemflows.workflow.findadjacencies import select_central_edges, select_closest_edges

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
                                  seg_instance=None ):
    if seg_instance is None:
        syn_info = fetch_instance_info(server, uuid, syn_instance)
        seg_instance = syn_info["Base"]["Syncs"][0]

    ref_seg = (server, uuid, seg_instance)
    
    edges_df = load_edges(edge_table)
    edges_df = update_localized_edges(*ref_seg, edges_df, processes)
    
    bois = determine_bodies_of_interest( server, uuid, syn_instance,
                                         boi_rois,
                                         min_tbars_in_roi, min_psds_in_roi,
                                         processes,
                                         synapse_table=synapse_table )

    edges_df = filter_groups_for_min_boi_count(edges_df, bois, 2)
    fragment_edges_df = compute_fragment_edges(edges_df, bois)

    if fragment_rois is not None:
        fragment_edges_df = filter_fragments_for_roi(server, uuid, fragment_rois, fragment_edges_df)

    # Fetch the supervoxel IDs if necessary.
    if {'sv_a', 'sv_b'} - {*fragment_edges_df.columns}:
        with Timer("Sampling supervoxel IDs", logger):
            points_a = fragment_edges_df[['za', 'ya', 'xa']].values
            points_b = fragment_edges_df[['zb', 'yb', 'xb']].values
            fragment_edges_df['sv_a'] = fetch_labels_batched(*ref_seg, points_a, True, processes=processes)
            fragment_edges_df['sv_b'] = fetch_labels_batched(*ref_seg, points_b, True, processes=processes)
    
    # Divide into 'focused' and 'merge review' fragments,
    # i.e. single-edge fragments and multi-edge fragments
    focused_fragments_df = fragment_edges_df.groupby(['group_cc', 'cc_task']).filter(lambda task_df: len(task_df) == 1).copy()
    mr_fragments_df = fragment_edges_df.groupby(['group_cc', 'cc_task']).filter(lambda task_df: len(task_df) > 1).copy()

    return focused_fragments_df, mr_fragments_df


def generate_mergereview_assignments_from_df(server, uuid, instance, mr_fragments_df, bois, assignment_size, output_dir):
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


def load_edges(edge_table):
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
    edges_df = edges_df.query('group_cc != -1')
    
    swap_df_cols(edges_df, None, edges_df.eval('label_a > label_b'), ('a', 'b'))
    assert len(edges_df.query('label_a == label_b')) == 0
    assert edges_df.duplicated(['group', 'label_a', 'label_b']).sum() == 0
    return edges_df


def update_localized_edges(server, uuid, seg_instance, edges_df, processes=16):
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
        direct_edges_df = edges_df.query('distance == 1.0')
        direct_edges_df = select_central_edges(direct_edges_df)

    with Timer("Re-selecting closest-approaching nearby edges", logger):
        nearby_edges_df = edges_df.query('distance > 1.0')
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


def filter_groups_for_min_boi_count(edges_df, bois, min_boi_count=2):
    with Timer("Filtering out groups with too few BOIs", logger):
        # Drop CC with only 1 BOI
        cc_boi_counts = {}
        for cc, cc_df in tqdm_proxy(edges_df.groupby('group_cc'), total=len(edges_df['group_cc'].unique()), leave=False):
            labels = pd.unique(cc_df[['label_a', 'label_b']].values.reshape(-1))
            count = sum((label in bois) for label in labels)
            cc_boi_counts[cc] = count
    
        boi_counts_df = pd.DataFrame(list(cc_boi_counts.items()), columns=['group_cc', 'boi_count'])
        
        kept_ccs = set(boi_counts_df.query('boi_count >= @min_boi_count')['group_cc'])
        logger.info(f"Keeping {len(kept_ccs)} group_cc out of {len(boi_counts_df)}")
        edges_df = edges_df.query('group_cc in @kept_ccs')
        return edges_df


def compute_fragment_edges(edges_df, bois):
    fragments = extract_fragments(edges_df, bois)
    
    with Timer("Extracting edges for each fragment from full table", logger):
        edges_df = edges_df.query('group_cc in @fragments.keys()')
        
        fragment_edges_dfs = []
        for group_cc,  group_df in tqdm_proxy(edges_df.groupby('group_cc')):
            frag_list = fragments[group_cc]
            for task_index, frag in enumerate(frag_list):
                frag_edges = list(zip(frag[:-1], frag[1:]))
                frag_edges = np.sort(frag_edges, axis=1)
    
                frag_edges_df = pd.DataFrame(frag_edges, columns=['label_a', 'label_b'])
                frag_edges_df = frag_edges_df.merge(group_df, 'left', ['label_a', 'label_b'])
                frag_edges_df['cc_task'] = task_index
                fragment_edges_dfs.append(frag_edges_df)
    
        fragment_edges_df = pd.concat(fragment_edges_dfs, ignore_index=True)

    return fragment_edges_df


def extract_fragments(edges_df, bois):
    """
    For each connected component (pre-labeled) in the given 
    """
    assert edges_df.duplicated(['group', 'label_a', 'label_b']).sum() == 0
    
    groups = edges_df.groupby('group_cc')
    num_groups = edges_df['group_cc'].nunique()

    with Timer("Extracting fragments from each group", logger):
        fragments = {}
        for group_cc, cc_df in tqdm_proxy(groups, total=num_groups):
            fragments[group_cc] = extract_fragments_for_cc(cc_df, bois)
        
    num_fragments = sum(len(frags) for frags in fragments.values())
    logger.info(f"Extracted {num_fragments} fragments")
    return fragments


def extract_fragments_for_cc(cc_df, bois):
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
    return fragments


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
    edges = cc_df[['label_a', 'label_b']].values
    
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


def display_graph(cc_df, bois, width=500, hv=True):
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

    node_color = ['limegreen' if g.nodes[n]['boi'] else 'skyblue' for n in g.nodes]
    edge_color = ['skyblue' if g.edges[a,b]['distance'] == 1.0 else 'red' for (a,b) in g.edges]
    labels = {n: str(g.nodes[n]['body']) for n in g.nodes}
    
    print(f"Drawing {len(g.nodes)} nodes and {len(g.edges)} edges")

    p = draw(g, node_color=node_color, edge_color=edge_color, labels=labels, pos=nx.kamada_kawai_layout(g), with_labels=True)
    
    if hv:
        p = p.opts(plot=dict(width=width, height=width))
    return p
