[![Travis CI Status](https://travis-ci.com/stuarteberg/flyemflows.svg?branch=master)](https://travis-ci.com/stuarteberg/flyemflows)

flyemflows
==========
Welcome to flyemflows, the center for distributed data processing for the FlyEM team at HHMI's Janelia Research Campus. This repo contains a suite of tools for processing large-scale volumetric image data using Dask. flyemflows operates on workflows defined in YAML documents, and those workflows can be run locally or on a cluster. Define your YAML file, launch your workflow, and you're off to the races!

Installation
==========
The flyemflows toolkit is available via the FlyEM team's conda channel, flyem-forge. To install flyemflows and get started, you can run a conda install from within your conda environment of choice. Here, we rely on the community-supported conda-forge channel to fill in any additional dependency needs.

`conda install -c flyem-forge -c conda-forge flyemflows`

Workflows?
==========
Yes, workflows! These are defined sequences of operations that are useful for all* of your volumetric image processing needs. Do you want to see all of the beautiful workflows available to you? Of course you do! Issue the following command, leveraging the ever-so-useful `launchflow` executable. This command will take a few seconds, as launchflow currently imports its libraries each time it is called.

`launchflow -w`

Running a Workflow
==========
The command above will display the built-in workflows, and it will also describe the process for using third-party workflows. Want to use a given workflow? This can be done by pointing `launchflow` at a template directory containing a workflow.yaml document. Once you know which workflow you want to run, you can use the following command to get an template directory named `mesh_template` that will contain the scaffolding of the YAML required for running the `CreateMeshes` workflow.

`mkdir mesh_template && launchflow -y CreateMeshes > mesh_template/workflow.yaml`

You can also get a more detailed YAML file by running eg `launchflow -v CreateMeshes` for the verbose version. Once you specify the details required of your workflow (input data settings, output data settings, workflow-specific parameters, etc), you can run the workflow! To run our example workflow with 10 workers, we would simply issue the following command.

`launchflow -n 10 mesh_template`

This will create a run directory based on the template with a timestamp appended to the name. That run directory will contain information about that run of the workflow, and you can reuse the template directory at a later date to run the job again.

Congratulations, you've run flyemflows!

Data Sources
==========

Downsampling
==========

Mesh Generation
==========

