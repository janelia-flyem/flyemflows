import os
import pytest

import numpy as np

from flyemflows.volumes import BrainMapsVolumeServiceReader
from flyemflows.volumes.brainmaps_volume import BrainMapsVolume

@pytest.mark.skipif(not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                    reason="Skipping BrainMaps test: GOOGLE_APPLICATION_CREDENTIALS is not defined")
def test_read(disable_auto_retry):
    """
    Minimal test to exercise the BrainMapsVolumeServiceReader.
    Reads a tiny subvolume from a known remote Brainmaps volume instance.
    
    If this is ever deleted or becomes unavailable,
    we'll have to find a different volume to test with.
    
    Note:
        For this test to work, you must have access to the flyem project,
        using a json credentials file pointed to by the
        GOOGLE_APPLICATION_CREDENTIALS environment variable.
    """
    config = {
      "brainmaps": {
        "project": "274750196357",
        "dataset": "janelia-flyem-cx-flattened-tabs",
        "volume-id": "sec26_seg_v2a",
        "change-stack-id": "ffn_agglo_pass1_seg5663627_medt160"
      },
      "geometry": {}
  }

    # raw volume handle
    bmv = BrainMapsVolume.from_flyem_source_info(config["brainmaps"])

    # Just verify that the 'service' wrapper is consistent with the low-level handle
    service = BrainMapsVolumeServiceReader(config)
    assert (service.bounding_box_zyx == bmv.bounding_box).all()
    assert service.dtype == bmv.dtype

    # Service INSERTS geometry into config if necessary
    assert config["geometry"]["bounding-box"] == bmv.bounding_box[:,::-1].tolist()

    start_xyz = np.array([18954, 3865, 15305])
    start_zyx = start_xyz[::-1]
    box_zyx = np.array([start_zyx, start_zyx + 256])

    subvol = service.get_subvolume(box_zyx)
    assert (subvol == bmv.get_subvolume(box_zyx)).all()
    assert subvol.any(), "Volume is all zeros"

    # Again, but this time with fetch-blockwise: true
    config["brainmaps"]["fetch-blockwise"] = True
    service = BrainMapsVolumeServiceReader(config)

    subvol = service.get_subvolume(box_zyx)
    assert (subvol == bmv.get_subvolume(box_zyx)).all()
    assert subvol.any(), "Volume is all zeros"


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_brainmaps_volume_service'])
