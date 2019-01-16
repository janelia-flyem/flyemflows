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
            "default": 16
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
            "description": "How much memory to allot to each 'job' (typically an entire\n"
                           "node's worth, assuming the job reserved all CPUs).\n"
                           "This memory will be divided up amongst the workers in the job.\n"
                           "Specified as a string with a suffix for units, e.g. 4GB\n",
            "type": "string",
            "default": "128GB"
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

JobQueueSchema = \
{
    "description": "dask-jobqueue config settings.",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "lsf": LsfJobSchema
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

