import os
import pytest

import numpy as np

from flyemflows.volumes import BossVolumeServiceReader
from intern.remote.boss.remote import BossRemote

@pytest.mark.skipif(not os.environ.get('BOSS_TOKEN', ''),
                    reason="Skipping BossVolumeService test: BOSS_TOKEN is not defined")
def test_read():
    """
    Minimal test to exercise the BossVolumeServiceReader.
    Reads a tiny subvolume from a known remote Boss volume.
    
    If this is ever deleted or becomes unavailable,
    we'll have to find a different volume to test with.
    
    Note:
        For this test to work, you must have access to the flyem project,
        using a json credentials file pointed to by the
        GOOGLE_APPLICATION_CREDENTIALS environment variable.
    """
    config = {
      "boss": {
        "host": "api.bossdb.org",
        "collection": "FIXME",
        "experiment": "FIXME",
        "channel": "FIXME"
      },
      "geometry":
        {
            "bounding-box": [[0,0,0], [10_000,10_000,10_000]]
        }
  }

    # raw volume handle
    boss_remote = BossRemote({
                        "protocol": "https",
                        "host": config["boss"]["host"],
                        "token": os.environ["BOSS_TOKEN"],
                })

    channel = boss_remote.get_channel(
                        config["boss"]["channel"],
                        config["boss"]["collection"],
                        config["boss"]["experiment"],
                )

    # Just verify that the 'service' wrapper is consistent with the low-level handle
    service = BossVolumeServiceReader(config)
    #assert (service.bounding_box_zyx == boss_remote.bounding_box).all()
    assert service.dtype == np.dtype(channel.datatype)

    start_xyz = np.array([5000, 4000, 3000])
    start_zyx = start_xyz[::-1]
    box_zyx = np.array([start_zyx, start_zyx + 256])
    subvol = service.get_subvolume(box_zyx)

    bounds_xyz = box_zyx[:,::-1].transpose()    
    cutout = boss_remote.get_cutout(channel, 0, *bounds_xyz.tolist())
    
    assert (subvol == cutout).all()
    assert subvol.any(), "Volume is all zeros"


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_boss_volume_service'])
