import os
import json
import threading
from collections import defaultdict
from itertools import chain

from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials

import numpy as np
import snappy


# DEBUG 'logging' (Doesn't actually use logging module)
#import httplib2
#httplib2.debuglevel = 1

# Brainmaps Docs:
# https://developers.google.com/brainmaps
BRAINMAPS_API_VERSION = 'v1'
BRAINMAPS_BASE_URL = f'https://brainmaps.googleapis.com/{BRAINMAPS_API_VERSION}'

CACHED_BRAINMAPS_HTTP_HANDLES = {}

def get_brainmaps_http_interface(timeout):
    """
    Obtain an Http object for accessing BrainMaps from the global pool of such objects.
    This allows us to avoid re-initializing Http objects repeatedly if BrainMapsVolume
    instances are created (or unpickled) repeatedly in the same process/thread.
    """
    # One handle per thread/process
    thread_id = threading.current_thread().ident
    pid = os.getpid()
    key = (pid, thread_id, timeout)

    try:
        http = CACHED_BRAINMAPS_HTTP_HANDLES[key]
    except KeyError:
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if  not credentials_path:
            raise RuntimeError("To access BrainMaps volumes, you must define GOOGLE_APPLICATION_CREDENTIALS "
                               "in your environment, which must point to a google service account json credentials file.")

        scopes = ['https://www.googleapis.com/auth/brainmaps']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scopes)
        http = credentials.authorize(Http(timeout=timeout))
        CACHED_BRAINMAPS_HTTP_HANDLES[key] = http
    
    return http


class BrainMapsVolume:
    def __init__(self, project, dataset, volume_id, change_stack_id="", dtype=None, skip_checks=False, use_gzip=True, timeout=60.0):
        """
        Utility for accessing subvolumes of a BrainMaps volume.
        Instances of this class are pickleable, but they will have to re-authenticate after unpickling.
        
        For REST API details, see the BrainMaps API documentation:
        https://developers.google.com/brainmaps/v1/rest/
        
        (To access the docs, you need to email a BrainMaps developer at Google and
        ask them to add your email to brainmaps-tt@googlegroups.com.)
        
        Args:
            project, dataset, volume_id, and (optionally) change_stack_id can be extracted from a brainmaps volume uri:
            
            >>> url = 'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_seg_v2a:ffn_agglo_pass1_seg5663627_medt160'
            >>> full_id = url.split('://')[1]
            >>> project, dataset, volume_id, change_stack_id = full_id.split(':')

            dtype: (Optional.) If not provided, a separate request to the 
                   BrainMaps API will be made to determine the volume voxel type.

            skip_checks: If True, verify that the volume_id and change_stack_id exist on the server.
                         Otherwise, skip those checks, to minimize overhead.
        """
        self.project = str(project)
        self.dataset = dataset
        self.volume_id = volume_id
        self.change_stack_id = change_stack_id
        self.skip_checks = skip_checks
        self._dtype = None # Assigned *after* check below.
        self.use_gzip = use_gzip
        self.timeout = timeout

        # These members are lazily computed/memoized.
        self._http = None
        self._bounding_boxes = None
        self._geometries = None
        self._equivalence_mapping = None

        if not self.skip_checks:
            volume_list = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/volumes')
            try:
                vol_list = volume_list['volumeId']
            except KeyError:
                raise RuntimeError("Failed to fetch volume list.  Are you using the right project and credentials?")

            # We would LIKE to execute the following checks, but due to a known problem with the
            # brainmaps cache infrastructure, sometimes new-ish volumes are missing from the volume list.
            #
            # if f'{project}:{dataset}:{volume_id}' not in volume_list['volumeId']:
            #     raise RuntimeError(f"BrainMaps volume does not exist on server: {project}:{dataset}:{volume_id}\n"
            #                        f"Available volumes: {json.dumps(volume_list, indent=2)}")
            #
            # if change_stack_id:
            #     if change_stack_id not in self.get_change_stacks():
            #         raise RuntimeError(f"ChangeStackId doesn't exist on the server: '{change_stack_id}'")

            try:
                _ = self.geometries
            except Exception as ex:
                msg = f"Could not fetch the geometry for {self.volume_uri()}.\n"
                if f'{project}:{dataset}:{volume_id}' not in volume_list['volumeId']:
                    msg += f"BrainMaps volume does not exist on server: {project}:{dataset}:{volume_id}\n"
                    msg += f"Available volumes: {json.dumps(volume_list, indent=2)}"
                raise RuntimeError(msg) from ex

            if dtype:
                assert self.dtype == dtype, \
                    f"Provided dtype {dtype} doesn't match volume metadata ({self.dtype})"

        self._dtype = dtype


    @classmethod
    def from_volume_uri(cls, uri):
        """
        Convenience constructor.
        Construct from a BrainMaps volume URI, such as:
        
            brainmaps://274750196357:hemibrain:my_volume_name:some_changestack_name
        """
        assert uri.startswith('brainmaps://')
        
        components = uri.split('://')[1].split(':')
        if len(components) == 4:
            project, dataset, volume_id, change_stack_id = components
        elif len(components) == 3:
            project, dataset, volume_id = components
            change_stack_id = ""
        else:
            raise RuntimeError(f"Invalid volume URI: {uri}")
        
        return BrainMapsVolume(project, dataset, volume_id, change_stack_id)
    
    @classmethod
    def from_flyem_source_info(cls, d):
        """
        Convenience constructor.
        Construct from FlyEM JSON config data.
        """
        if 'use-gzip' not in d:
            d["use-gzip"] = True
        return BrainMapsVolume(d["project"], d["dataset"], d["volume-id"], d["change-stack-id"], use_gzip=d["use-gzip"])

    def flyem_source_info(self, as_str=False):
        """
        Convenience method.
        Return parameters as JSON for FlyEM config files.
        """
        info = {
            "service-type": "brainmaps",
            "project": self.project,
            "dataset": self.dataset,
            "volume-id": self.volume_id,
            "change-stack-id": self.change_stack_id,
            "bounding-box": self.bounding_box[:,::-1] # Print in xyz order
        }
        if not as_str:
            return info

        # The json so-called pretty-print makes the bounding-box ugly.
        # Make it pretty and re-insert it.
        del info["bounding-box"]
        json_text = json.dumps(info, indent=4, cls=NumpyConvertingEncoder)
        json_lines = json_text.split('\n')
        json_lines[-2] += ','
        json_lines.insert(-1, f'    "bounding-box": {self.bounding_box[:,::-1].tolist()}')
        return '\n'.join(json_lines)

    def volume_uri(self):
        parts = (self.project, self.dataset, self.volume_id)
        if self.change_stack_id:
            parts += (self.change_stack_id,)
        return 'brainmaps://' + ':'.join(parts)

    def get_subvolume(self, box_zyx, scale=0):
        """
        Fetch a subvolume from the remote BrainMaps volume.

        Args:
            box: (start, stop) tuple, in ZYX order.
            scale: Which scale to fetch the subvolume from.

        Returns:
            volume (ndarray), where volume.shape = (stop - start)
        """
        box_zyx = np.asarray(box_zyx)
        if (box_zyx[1] <= box_zyx[0]).any():
            raise RuntimeError(f"Invalid box: {box_zyx.tolist()}")

        bb = self.bounding_boxes[scale]
        if (box_zyx[0] < bb[0]).any() or (box_zyx[1] > bb[1]).any():
            msg = (f"Box ({box_zyx.tolist()}) exceeds "
                   f"volume bounding box ({bb.tolist()}) "
                   f"at scale {scale}")
            raise RuntimeError(msg)

        corner_zyx = box_zyx[0]
        shape_zyx = box_zyx[1] - box_zyx[0]

        corner_xyz = corner_zyx[::-1]
        shape_xyz = shape_zyx[::-1]

        snappy_data = fetch_subvol_data( self.http,
                                         self.project,
                                         self.dataset,
                                         self.volume_id,
                                         corner_xyz,
                                         shape_xyz,
                                         scale,
                                         self.change_stack_id,
                                         self.use_gzip )

        volume_buffer = snappy.decompress(snappy_data)
        volume = np.frombuffer(volume_buffer, dtype=self.dtype).reshape(shape_zyx)
        return volume


    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = np.dtype(self.geometries[0]['channelType'].lower())
        return self._dtype


    @property
    def bounding_box(self):
        """
        The bounding box [start, stop] of the volume at scale 0, in zyx order.
        """
        return self.bounding_boxes[0] # Scale 0


    @property
    def bounding_boxes(self):
        """
        A list of bounding boxes (one per scale), each in zyx order.
        """
        if self._bounding_boxes is None:
            self._bounding_boxes = list(map(self._extract_bounding_box, self.geometries))
        return self._bounding_boxes


    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()
        # Don't attempt to pickle the http connection, because
        # it would no longer be valid (authenticated) after it is unpickled.
        # Instead, set it to None so it will be lazily regenerated after unpickling.
        d['_http'] = None
        return d


    @property
    def http(self):
        """
        Returns an authenticated httplib2.Http object to use for all BrainMaps API requests.
        Memoized here instead of generated in the constructor,
        since we intentionally delete the _http member during pickling.
        """
        if self._http is not None:
            return self._http

        self._http = get_brainmaps_http_interface(self.timeout)
        return self._http


    @property
    def geometries(self):
        """
        The (memoized) geometry json for all scales.
        See get_geometry() for details.
        """
        if self._geometries is None:
            self._geometries = self.get_geometry()
            assert int(self._geometries[0]['channelCount']) == 1, \
                "Can't use this class on multi-channel volumes."
        return self._geometries


    def get_change_stacks(self):
        """
        Get the list of change_stacks 
        """
        msg_json = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/change_stacks')
        return msg_json['changeStackId']
        

    def get_geometry(self):
        """
        Returns a list of geometries (one per scale):
        
        [{
          'boundingBox': [{
            'corner': {},
            'size': {'x': '37911', 'y': '7731', 'z': '30613'}
          }],
          'channelCount': '1',
          'channelType': 'UINT64',
          'pixelSize': {'x': 8, 'y': 8, 'z': 8},
          'volumeSize': {'x': '37911', 'y': '7731', 'z': '30613'}
        },
        ...]

        (Notice that many of these numbers are strings, for some reason.) 
        """
        msg_json = fetch_json( self.http, f'{BRAINMAPS_BASE_URL}/volumes/{self.project}:{self.dataset}:{self.volume_id}')
        return msg_json['geometry']


    def _extract_bounding_box(self, geometry):
        """
        Return the bounding box [start, stop] in zyx order.
        """
        if 'boundingBox' in geometry:
            corner = geometry['boundingBox'][0]['corner']
            size = geometry['boundingBox'][0]['size']
            
            corner = defaultdict(lambda: 0, corner)
            size = defaultdict(lambda: 0, size)
        
            shape = [int(size[k]) for k in 'zyx']
            if not corner:
                offset = (0,)*len(size)
            else:
                offset = [int(corner[k]) for k in 'zyx']

            box = np.array((offset, offset))
            box[1] += shape
        else:
            size = geometry['volumeSize']
            shape = [int(size[k]) for k in 'zyx']

            box = np.zeros((2,3), int)
            box[1] = shape
            
        return box


    def get_equivalence_group(self, segment_id):
        """
        Return the set of segments (supervoxels) that are in the same group as the given segment_id.
        """
        assert self.change_stack_id, "No equivalences: This volume has no change_stack_id"
        url = f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/{self.change_stack_id}/equivalences:getgroups'

        # Technically we could give a list here, but we just ask for one at a time.
        equivalences = fetch_json(self.http, url, body={'segmentId': [segment_id]})
        
        # json data is
        # 
        # {
        #   'groups':
        #   [
        #     {
        #       'groupMembers':
        #       [
        #         '411313',
        #         '828042',
        #         '1239392',
        #         ...
        #       ]
        #     },
        #     ...
        #   ]
        # }
        
        # Convert from string to int
        siblings = set(map(int, equivalences["groups"][0]["groupMembers"]))
        assert segment_id in siblings, \
            f"Expected query segment ({segment_id}) to be in its own equivalence set: {siblings}"

        return siblings


    def get_all_equivalence_groups(self):
        """
        Like get_equivalences(), but returns results for all groups.
        
        Return a dictionary of { group_1 : [segment_1, segment_2, ...],
                                 group_2 : [segment_10, segment_11, ...],
                                 ... }
        """
        assert self.change_stack_id, "No equivalences: This volume has no change_stack_id"

        # There is no API for getting all groups, so we have to get the
        # full set of edges and run connected components ourselves.
        all_edges = self.get_equivalence_edges()
        return groups_from_edges(all_edges)


    def equivalence_mapping(self):
        if self._equivalence_mapping is not None:
            return self._equivalence_mapping
        groups = self.get_all_equivalence_groups()
        self._equivalence_mapping = mapping_from_groups(groups)
        return self._equivalence_mapping

    def set_equivalence_mapping(self, mapping):
        self._equivalence_mapping = mapping


    def get_equivalence_edges(self, segment_id=None):
        """
        Get the merged edges for the group that owns the given segment as a numpy array [[s1, s2], [s1, s2], ...].
        If segment_id is None, return all equivalence edges in the entired volume.
        """
        assert self.change_stack_id, "No equivalences: This volume has no change_stack_id"

        url = f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/{self.change_stack_id}/equivalences:list'
        if segment_id is None:
            edges_json = fetch_json(self.http, url, body={})
        else:
            edges_json = fetch_json(self.http, url, body={'segmentId': segment_id})

        # Data comes back in this format:
        # 
        # { 'edge': 
        #   [ 
        #     {'first': '411313', 'second': '1033743'},
        #     {'first': '411313', 'second': '1239364'},
        #     {'first': '411313', 'second': '1442313'},
        #     ...
        #   ]
        # }

        num_edges = len(edges_json['edge'])
        firsts = (int(edge['first']) for edge in edges_json['edge'])
        seconds = (int(edge['second']) for edge in edges_json['edge'])

        edges_flat = np.fromiter(chain(firsts, seconds), np.uint64, 2*num_edges)
        edges = edges_flat.reshape((2,-1)).transpose()
        return edges

def fetch_json(http, url, body=None):
    """
    Fetch JSON data from a BrainMaps API endpoint.
    
    Args:
        http: Authenticated httplib2.Http object.
        url: Full url to the endpoint.
        body: If the endpoint requires parameters in the body,
              give them here (as a dict). Forces method to be POST.
    
    Examples:
    
        projects = fetch_json(http, f'{BRAINMAPS_BASE_URL}/projects')
        volume_list = fetch_json(http, f'{BRAINMAPS_BASE_URL}/volumes')
    """
    if body is None:
        method = "GET"
    else:
        method = "POST"
        if not isinstance(body, (str, bytes)):
            body = json.dumps(body, cls=NumpyConvertingEncoder)

    response, content = http.request(url, method, body)
    if response['status'] != '200':
        raise RuntimeError(f"Bad response ({response['status']}):\n{content.decode('utf-8')}")

    return json.loads(content)


def fetch_subvol_data(http, project, dataset, volume_id, corner_xyz, size_xyz, scale, change_stack_id="", use_gzip=True):
    """
    Returns raw subvolume data (not decompressed).
    
    Clients should generally not call this function directly.
    Instead, use the BrainMapsVolume class.
    """
    url = f'{BRAINMAPS_BASE_URL}/volumes/{project}:{dataset}:{volume_id}/subvolume:binary'

    params = \
    {
        'geometry': {
            'corner': ','.join(str(x) for x in corner_xyz),
            'size': ','.join(str(x) for x in size_xyz),
            'scale': int(scale)
        },
        'subvolumeFormat': 'RAW_SNAPPY'
    }

    if change_stack_id:
        params["changeSpec"] = { "changeStackId": change_stack_id }

    if use_gzip:
        # GZIP is enabled by default in httplib2; but let's be explicit.
        headers = { 'accept-encoding': 'gzip' }
        response, content = http.request(url, "POST", headers=headers, body=json.dumps(params).encode('utf-8'))
    else:
        # GZIP is enabled by default in httplib2; this is the only way to disable it.
        headers = { 'accept-encoding': 'FAKE' }
        response, content = http.request(url, "POST", headers=headers, body=json.dumps(params).encode('utf-8'))
    
    if response['status'] != '200':
        raise RuntimeError(f"Bad response ({response['status']}):\n{content.decode('utf-8')}")
    return content

###
### Helper functions
###

def groups_from_edges(edges):
    """
    The given list of edges [(node_a, node_b),(node_a, node_b),...] encode a graph.
    Find the connected components in the graph and return them as a dict:
    
    { group_id : [node_id, node_id, node_id] }
    
    ...where each group_id is the minimum node_id of the group.
    """
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(edges)
    
    groups = {}
    for segment_set in nx.connected_components(g):
        # According to Jeremy, group_id == the min segment_id of the group.
        groups[min(segment_set)] = list(sorted(segment_set))

    return groups


def mapping_from_groups(groups):
    """
    Given a dict of { group_id: [node_a, node_b,...] },
    Return a reverse-mapping in the form of an ndarray:
        
        [[node_a, group_id],
         [node_b, group_id],
         [node_c, group_id],
         ...
        ]
    """
    element_count = sum(map(len, groups.values()))
    
    def generate():
        for group_id, members in groups.items():
            for member in members:
                yield member
                yield group_id

    mapping = np.fromiter( generate(), np.uint64, 2*element_count ).reshape(-1,2)
    return mapping


class NumpyConvertingEncoder(json.JSONEncoder):
    """
    Encoder that converts numpy arrays and scalars
    into their pure-python counterparts.
    
    (No attempt is made to preserve bit-width information.)
    
    Usage:
    
        >>> d = {"a": np.arange(3, dtype=np.uint32)}
        >>> json.dumps(d, cls=NumpyConvertingEncoder)
        '{"a": [0, 1, 2]}'
    """
    def default(self, o):
        if isinstance(o, (np.ndarray, np.number)):
            return o.tolist()
        return super().default(o)
