LsfJobSchema = \
{
    "description": "dask-jobqueue config settings for LSF jobs.\n"
                   "https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask-jobqueue-lsfcluster",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "cores": {
            "description": "How many cores for each 'job', (typically an entire node's worth).\n"
                           "The 'workers' (processes) in each job will share these cores amongst them.",
            "type": "integer",
            "minimum": 1,
            "default": 1
        },
        "ncpus": {
            "description": "How many CPUs to reserve for each 'job'.\n"
                           "Typically, this should be the same as 'cores' (which is the default behavior if not specified),\n"
                           "unless you're worried about your RAM usage, in which case you may want it to be higher.\n"
                           "(This setting has no direct effect on dask behavior;\n"
                           "it is solely for fine-tuning resource reservations in the LSF scheduler.)\n",
            "type": "integer",
            "default": -1
        },
        "processes": {
            "description": "How many processes ('workers') per 'job'.\n"
                           "These processes will collectively share the 'cores' you specify for the job.\n"
                           "https://jobqueue.dask.org/en/latest/configuration-setup.html#processes",
            "type": "integer",
            "minimum": 1,
            "default": 1
        },
        "memory": {
            "description": "How much memory to allot to each 'job' (e.g. an entire\n"
                           "node's worth, if the job reserved all CPUs).\n"
                           "This memory will be divided up amongst the workers in the job,\n"
                           "so if you are setting 'cores' to more than one core, you should\n"
                           "increase this setting accordingly.\n"
                           "Specified as a string with a suffix for units, e.g. 4GB\n",
            "type": "string",
            "default": "15GB" # On the Janelia cluster, each slot gets 15 GB by default. 
        },
        "mem": {
            "description": "How much memory to reserve from LSF for each 'job'.\n"
                           "Typically should be the same as the dask 'memory' setting,\n"
                           "which is the default if not specified here.\n"
                           "(This setting has no direct effect on dask behavior;\n"
                           "it is solely for fine-tuning resource reservations in the LSF scheduler.)\n"
                           "Note: Must be specifed in bytes (as an integer, not string)",
            "type": "integer",
            "default": 0
        },
        "interface": {
            "description": "Network interface to use like eth0 or ib0",
            "oneOf": [
                {"type": "string"},
                {"type": "null"}
            ],
            "default": None
        },
        "use-stdin": {
            "description": "Implementation detail regarding how bsub is called by dask-jobqueue.\n"
                           "Under Janelia's LSF configuration, this must be set to 'True'.",
            "type": "boolean",
            "default": True
        },
        "log-directory": {
            "description": "Where LSF worker logs (from stdout) will be stored.",
            "type": "string",
            "default": "job-logs"
        },
        "local-directory": {
            "description": "Where dask should store temporary files when data spills to disk.\n"
                           "Note: Will also be used to configure Python's tempfile.tempdir",
            "type": "string",
            "default": ""
        },
        "walltime": {
            "description": "How much time to give the workers before killing them automatically.\n"
                           "Specified in HH:MM format.\n",
            "type": "string",
            "default": "24:00"
        },
        "death-timeout": {
            "description": "Seconds to wait for a scheduler before closing workers",
            "type": "integer",
            "default": 60
        },
        "name": {
            "description": "The name of the dask worker jobs when submitted to LSF.\n",
            "type": "string",
            "default": "dask-worker"
        }
    }
}


SgeJobSchema = \
{
    "description": "dask-jobqueue config settings for SGE jobs.\n"
                   "https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SGECluster.html#dask-jobqueue-sgecluster",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "walltime": {
            "description": "How much time to give the workers before killing them automatically.\n"
                           "Specified in HH:MM format.\n",
            "type": "string",
            "default": "24:00"
        },
        "name": {
            "description": "The name of the dask worker jobs when submitted to SGE.\n",
            "type": "string",
            "default": "dask-worker"
        },
        "cores": {
            "description": "How many cores for each 'job', (typically an entire node's worth).\n"
                           "The 'workers' (processes) in each job will share these cores amongst them.",
            "type": "integer",
            "minimum": 1,
            "default": 16
        },
        "memory": {
            "description": "How much memory to allot to each 'job' (e.g. an entire\n"
                           "node's worth, if the job reserved all CPUs).\n"
                           "This memory will be divided up amongst the workers in the job,\n"
                           "so if you are setting 'cores' to more than one core, you should\n"
                           "increase this setting accordingly.\n"
                           "Specified as a string with a suffix for units, e.g. 4GB\n",
            "type": "string",
            "default": "15GB" # On the Janelia cluster, each slot gets 15 GB by default. 
        },
        "processes": {
            "description": "How many processes ('workers') per 'job'.\n"
                           "These processes will collectively share the 'cores' you specify for the job.\n"
                           "https://jobqueue.dask.org/en/latest/configuration-setup.html#processes",
            "type": "integer",
            "minimum": 1,
            "default": 1
        },
        "log-directory": {
            "description": "Where SGE worker logs (from stdout) will be stored.",
            "type": "string",
            "default": "job-logs"
        },
        "local-directory": {
            "description": "Where dask should store temporary files when data spills to disk.\n"
                           "Note: Will also be used to configure Python's tempfile.tempdir",
            "type": "string",
            "default": ""
        }
    }
}


SlurmJobSchema = \
{
    "description": "dask-jobqueue config settings for SLURM jobs.\n"
                   "https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html#dask-jobqueue-slurmcluster",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "walltime": {
            "description": "How much time to give the workers before killing them automatically.\n"
                           "Specified in HH:MM format.\n",
            "type": "string",
            "default": "24:00"
        },
        "name": {
            "description": "The name of the dask worker jobs when submitted to SGE.\n",
            "type": "string",
            "default": "dask-worker"
        },
        "cores": {
            "description": "How many cores for each 'job', (typically an entire node's worth).\n"
                           "The 'workers' (processes) in each job will share these cores amongst them.",
            "type": "integer",
            "minimum": 1,
            "default": 16
        },
        "memory": {
            "description": "How much memory to allot to each 'job' (e.g. an entire\n"
                           "node's worth, if the job reserved all CPUs).\n"
                           "This memory will be divided up amongst the workers in the job,\n"
                           "so if you are setting 'cores' to more than one core, you should\n"
                           "increase this setting accordingly.\n"
                           "Specified as a string with a suffix for units, e.g. 4GB\n",
            "type": "string",
            "default": "15GB" # On the Janelia cluster, each slot gets 15 GB by default. 
        },
        "processes": {
            "description": "How many processes ('workers') per 'job'.\n"
                           "These processes will collectively share the 'cores' you specify for the job.\n"
                           "https://jobqueue.dask.org/en/latest/configuration-setup.html#processes",
            "type": "integer",
            "minimum": 1,
            "default": 1
        },
        "log-directory": {
            "description": "Where SLURM worker logs (from stdout) will be stored.",
            "type": "string",
            "default": "job-logs"
        },
        "local-directory": {
            "description": "Where dask should store temporary files when data spills to disk.\n"
                           "Note: Will also be used to configure Python's tempfile.tempdir",
            "type": "string",
            "default": ""
        },
        "job-cpu": {
            "description": "How many CPUs to reserve for each 'job'.\n"
                           "Typically, this should be the same as 'cores' (which is the default behavior if not specified),\n"
                           "unless you're worried about your RAM usage, in which case you may want it to be higher.\n"
                           "(This setting has no direct effect on dask behavior;\n"
                           "it is solely for fine-tuning resource reservations in the SLURM scheduler.)\n",
            "type": "integer",
            "default": -1
        },
        "job-mem": {
            "description": "How much memory to reserve from SLURM for each 'job'.\n"
                           "Typically should be the same as the dask 'memory' setting.\n",
            "type": "string",
            "default": "8GB"
        },
    }
}


JobQueueSchema = \
{
    "description": "dask-jobqueue config settings.",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "lsf": LsfJobSchema,
        "sge": SgeJobSchema,
        "slurm": SlurmJobSchema,
    }
}

DistributedSchema = \
{
    "description": "dask.distributed config section.",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "admin": {
            "description": "dask.distributed 'admin' config section.",
            "type": "object",
            "additionalProperties": True,
            "default": {},
            "properties": {
                'log-format': {
                    'type': 'string',
                    'default': '[%(asctime)s] %(levelname)s %(message)s'
                }
            }
        }
    }
}

DaskConfigSchema = \
{
    "description": "Dask config values to override the defaults in ~/.config/dask/ or /etc/dask/.\n"
                   "See https://docs.dask.org/en/latest/configuration.html for details.",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "distributed": DistributedSchema,
        "jobqueue": JobQueueSchema
    }
}

