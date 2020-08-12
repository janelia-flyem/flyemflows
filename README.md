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

Data Sources
==========

Downsampling
==========

Mesh Generation
==========

