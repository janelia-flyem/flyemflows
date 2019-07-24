Quickstart for FlyEM users
==========================

0. Install Miniconda.

```
CONDA_BASE=${HOME}/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_BASE} 
```

Do NOT configure conda to be activated automatically in your `~/.bashrc`.
That would be problematic for cluster jobs, which you generally want to
keep the same environment as the submitting shell.

Instead, create a handy alias for yourself so you can quickly activate a conda environment:

```
echo "alias conact=\"source ${CONDA_BASE}/bin/activate\"" >> ~/.bashrc
```

1. Create an environment for flyem software, and install `flyemflows` to it.  Then activate your new environment.

```
conact base
conda create --name flyem -c flyem-forge -c conda-forge flyemflows
conact flyem
```

2. Try running a `flyemflows` job on the LSF cluster.  Jobs are configured by preparing a template directory with your config files, and running `launchflow`.  Try the following example, which will convert a stack of `.png` images to `.n5`:

```
conact flyem
cd /groups/flyem/data/scratchspace/vnc

less vnc02-to-n5/workflow.yaml
less vnc02-to-n5/dask-config.yaml

bsub launchflow -n 124 vnc02-to-n5
```

LSF will run `launchflow` on a cluster node, which will, in turn, launch a dask-cluster with 124 workers (31*4).  When the job starts, your template directory is copied and used as the working directory for the job.

The main process log file is streamed to `output.log`.  Also, the link to the dask dashboard is shown there.  For links to the Janelia RTM graphs for the job, see `graph-links.txt`.

```
$ cd vnc02-to-n5-20190724.145907

$ tail -f output.log
[2019-07-24 14:59:10,724] INFO Teeing output to /groups/flyem/data/scratchspace/vnc/vnc02-to-n5-20190724.145907/output.log
[2019-07-24 14:59:10,765] INFO Working dir: /groups/flyem/data/scratchspace/vnc/vnc02-to-n5-20190724.145907
[2019-07-24 14:59:10,766] INFO Running copygrayscale with 124 workers...
[2019-07-24 14:59:10,787] INFO Driver LSB_JOBID is: 62110075
[2019-07-24 14:59:10,787] INFO Driver host is: e10u21.int.janelia.org
[2019-07-24 14:59:10,787] INFO Driver RTM graphs: http://lsf-rtm/cacti/plugins/grid/grid_bjobs.php?action=viewjob&tab=jobgraph&clusterid=1&indexid=0&jobid=62110075&submit_time=1563994739
[2019-07-24 14:59:11,579] INFO Dashboard running on http://10.36.60.31:8787/status
[2019-07-24 14:59:11,579] INFO               a.k.a. http://e10u21.int.janelia.org:8787/status
[2019-07-24 14:59:11,936] INFO Waiting for 124 workers to launch...
[2019-07-24 14:59:18,045] INFO Waiting for 124 workers to launch took 0:00:06.108913
[2019-07-24 14:59:18,260] INFO Running get_job_submit_time on 4 workers...
[2019-07-24 14:59:24,238] INFO Running get_job_submit_time on 4 workers took 0:00:05.977545
[2019-07-24 14:59:24,306] INFO Running construct_rtm_url on 124 workers...
[2019-07-24 14:59:32,856] INFO Running construct_rtm_url on 124 workers took 0:00:08.549534
[2019-07-24 14:59:33,020] INFO Running update_env on 124 workers...
[2019-07-24 14:59:33,089] INFO Running update_env on 124 workers took 0:00:00.068429
[2019-07-24 14:59:34,756] INFO Output bounding box: [[0, 0, 1], [18438, 4717, 21137]]
[2019-07-24 14:59:34,757] INFO Processing volume in 21 slabs
[2019-07-24 14:59:34,757] INFO Slab 0: STARTING. [[0, 0, 1], [18438, 4717, 1024]]
[2019-07-24 14:59:34,758] INFO Slab 0: Aiming for partitions of 358759689 voxels
[2019-07-24 14:59:34,782] INFO Changing num_partitions to avoid power of two
[2019-07-24 14:59:37,311] INFO Initializing RDD of 1023 Bricks (over 257 partitions) with total volume 89.0 Gvox
[2019-07-24 14:59:37,711] INFO Slab 0: Downloading scale 0...
[2019-07-24 15:00:43,520] INFO Slab 0: Downloading scale 0 (N=1023, P=257, P_hist={4: 255, 1: 1, 2: 1}) took 0:01:05.809054
[2019-07-24 15:00:43,522] INFO Realigning to output grid...
[2019-07-24 15:02:29,129] INFO Realigning to output grid took 0:01:45.606579
[2019-07-24 15:02:29,138] INFO Slab 0: Writing scale 0...
[2019-07-24 15:02:39,635] INFO Slab 0: Writing scale 0 took 0:00:10.497041
[2019-07-24 15:02:39,635] INFO Slab 0: Scale 0 took 0:03:04.878319
[2019-07-24 15:02:39,872] INFO Slab 0: Aiming for partitions of 44844961 voxels
[2019-07-24 15:02:39,873] INFO Slab 0: Downsampling to scale 1...
[2019-07-24 15:03:21,929] INFO Slab 0: Downsampling to scale 1 (N=760, P=289, P_hist={1: 41, 3: 111, 2: 84, 4: 47, 5: 6}) took 0:00:42.055569
[2019-07-24 15:03:21,929] INFO Realigning to output grid...
[2019-07-24 15:03:30,533] INFO Realigning to output grid took 0:00:08.604025
[2019-07-24 15:03:30,533] INFO Slab 0: Writing scale 1...
[2019-07-24 15:03:33,224] INFO Slab 0: Writing scale 1 took 0:00:02.690947
[2019-07-24 15:03:33,226] INFO Slab 0: Scale 1 took 0:00:53.590942
[2019-07-24 15:03:33,251] INFO Slab 0: Aiming for partitions of 5605620 voxels
[2019-07-24 15:03:33,252] INFO Slab 0: Downsampling to scale 2...
```

Once you're satisfied that this test job seems to be working, you can kill it early by killing the main (client) process.  That will also kill the dask workers.

```
$ grep JOBID output.log
[2019-07-24 14:59:10,787] INFO Driver LSB_JOBID is: 62110075

$ bkill 62110075
```
