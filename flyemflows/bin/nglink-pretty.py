#!/usr/bin/env python3
"""
Parse a neuroglancer link (from stdin) into JSON text and print it to the console.

Note:
    On macOS, there's a limit to how many characters can be pasted directly into
    the terminal when feeding into a program stdin.  That limit is low: 1024 characters.
    Many neuroglancer links are longer than that, so you'll have to use a workaround.
    For example, first paste the link into a file:

        $ emacs /tmp/link.txt
        $ nglink-pretty.py < /tmp/link.txt

    OR you can use pbpaste to feed it straight from the clipboard (bypassing the terminal):

        $ pbpaste | nglink-pretty.py
"""
import sys
import json
import argparse
import urllib.parse

assert sys.version_info.major == 3, "Requires Python 3"

example_link = """\
https://hemibrain-dot-neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B15807.5%2C21274.5%2C18124.5%5D%2C%22crossSectionScale%22:54.37327962468417%2C%22crossSectionDepth%22:-37.62185354999912%2C%22projectionScale%22:109219.18067006872%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg%22%2C%22tab%22:%22source%22%2C%22name%22:%22emdata%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation%22%2C%22subsources%22:%7B%22default%22:true%2C%22properties%22:true%2C%22mesh%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22segments%22%2C%22objectAlpha%22:0.3%2C%22name%22:%22segmentation%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22segmentation%22%7D%2C%22layout%22:%22xy-3d%22%7D
"""

example_legacy_link = """\
https://neuroglancer-demo.appspot.com/#!{'layers':{'sec26_image':{'type':'image'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_image'}_'ffn+celis:mask100:threshold0':{'type':'segmentation'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask100_0'}_'ffn+celis:mask200:threshold0':{'type':'segmentation'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_seg_v2a:ffn_agglo_pass1_cpt5663627_medt160_with_celis_cx2-2048_r10_mask200_0'_'visible':false}_'ground_truth_bodies_20171017':{'type':'segmentation'_'source':'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_ground_truth_bodies_20171017'_'visible':false}}_'navigation':{'pose':{'position':{'voxelSize':[8_8_8]_'voxelCoordinates':[18955.5_3865.5_15306.5]}}_'zoomFactor':8}}
"""


def replace_commas(d):
    result = {}
    for k,v in d.items():
        new_key = k.replace(',', '_')
        new_val = v
        if isinstance(v, str):
            new_val = v.replace(',', '_')

        result[new_key] = new_val
    return result


def pseudo_json_to_data(pseudo_json):
    # Replace URL-encoding characters, e.g. '%7B' -> '{'
    pseudo_json = urllib.parse.unquote(pseudo_json)

    # Make the text valid json by replacing single-quotes
    # with double-quotes and underscores with commas.
    pseudo_json = pseudo_json.replace("'", '"')    # But underscores within strings should not have been replaced,
    # so change those ones back as we load the json data.
    try:
        data = json.loads(pseudo_json, object_hook=replace_commas)
    except:
        sys.stderr.write(f"Couldn't parse JSON:\n{pseudo_json}")
        raise
    else:
        return data


def parse_nglink(link):
    url_base, pseudo_json = link.split('#!')
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return url_base + '#!', data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', action='store_true', help='Output json only, without the http link prefix')
    parser.add_argument('--legacy', '-l', action='store_true', help="Assume link is in neuroglancer's old format, which used underscores in place ofe commas")
    args = parser.parse_args()

    link = sys.stdin.read()

    if args.legacy:
        url_base, pseudo_json = link.split('#!')
        url_base += '#!'
        data = pseudo_json_to_data(pseudo_json)
        pseudo_json = urllib.parse.unquote(pseudo_json)
        data = json.loads(pseudo_json)
    else:
        url_base, data = parse_nglink(link)

    if not args.json:
        # Sometimes the terminal prints our ctrl+d control
        # character to the screen as (^D), which messes up the JSON output.
        # Printing a blank line first keeps the json separate.
        print("")
        print(url_base)

    pretty_text = json.dumps(data, indent=4)
    print(pretty_text)


if __name__ == "__main__":
    main()
