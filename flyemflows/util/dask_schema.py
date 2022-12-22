#
# https://docs.dask.org/en/latest/configuration-reference.html
#

DaskBaseSchemaProperties = \
{
    "temporary-directory": {
        "description": "Temporary directory for local disk storage /tmp, /scratch, or /local.\n"
                       "This directory is used during dask spill-to-disk operations.\n"
                       "When the value is 'null' (default), dask will create a directory\n"
                       "from where dask was launched: `cwd/dask-worker-space`\n",
        "oneOf": [
            {"type": "string"},
            {"type": "null"}
        ],
        "default": None
    }
}

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
            "default": "15GB"  # On the Janelia cluster, each slot gets 15 GB by default.
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
            "default": "15GB"  # On the Janelia cluster, each slot gets 15 GB by default.
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
            "default": "15GB"  # On the Janelia cluster, each slot gets 15 GB by default.
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

FractionOrFalse = {
    "oneOf": [
        {"type": "number", "minValue": 0.01},
        {"type": "boolean", "enum": [False]}
    ],
    "default": False
}

DistributedSchema = \
{
    "description": "dask.distributed config section.",
    "type": "object",
    "additionalProperties": True,
    "default": {},
    "properties": {
        "worker": {
            "type": "object",
            "default": {},
            "properties": {
                "preload": {
                    "description": "See https://docs.dask.org/en/latest/setup/custom-startup.html",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [
                        "distributed.config"  # Make sure logging config is loaded.
                    ]
                },
                "preload-argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": []
                },
                "memory": {
                    "description": "Memory management settings. These can cause trouble and create \n"
                                   "failures that are difficult to diagnose, so we disable them by default.\n",
                    "type": "object",
                    "default": {},
                    "properties": {
                        "target":    FractionOrFalse,
                        "spill":     FractionOrFalse,
                        "pause":     FractionOrFalse,
                        "terminate": FractionOrFalse,
                    }
                }
            }
        },
        "scheduler": {
            "type": "object",
            "default": {},
            "properties": {
                "preload": {
                    "description": "See https://docs.dask.org/en/latest/setup/custom-startup.html",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [
                        "distributed.config"  # Make sure logging config is loaded.
                    ]
                },
                "preload-argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": []
                }
            }
        },
        "nanny": {
            "type": "object",
            "default": {},
            "properties": {
                "preload": {
                    "description": "See https://docs.dask.org/en/latest/setup/custom-startup.html",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [
                        "distributed.config"  # Make sure logging config is loaded.
                    ]
                },
                "preload-argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": []
                }
            }
        },
        "admin": {
            "description": "dask.distributed 'admin' config section.",
            "additionalProperties": True,
            "default": {},
            "properties": {
                'log-format': {
                    "description": "In the distributed.config code, this is referred to as part of the 'old-style'\n"
                                   "logging configuration, but it seems to be used unconditionally within\n"
                                   "the Worker (node.py), so I'm confused.",
                    'type': 'string',
                    'default': '[%(asctime)s] %(levelname)s %(message)s'
                }
            }
        },
        "logging": {
            "description": "dask.distributed 'new-style' logging config just uses the standard Python configuration dictionary schema.\n"
                           "See distributed.config.initialize_logging(), and the Python docs:\n"
                           "https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema\n",
            "type": "object",

            # For readability, here's the default configuration we use all in one place.
            # Each of these properties' schemas are also listed below, to enable default
            # value injection in case the user supplies one or more custom entries.
            "default": {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "timestamped": {
                        "format": "[%(asctime)s] %(levelname)s %(message)s"
                    }
                },
                "handlers": {
                    "console": {
                        "level": "DEBUG",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                        "formatter": "timestamped",
                    }
                },
                "root": {
                    "handlers": ["console"],
                    "level": "INFO"
                },
                "loggers": {
                    "distributed.client": {"level": "WARNING"},
                    "bokeh": {"level": "ERROR"},
                    "tornado": {"level": "CRITICAL"},
                    "tornado.application": {"level": "ERROR"},
                }
            },

            # List each property's schema independently to ensure their default values are
            # injected into the config, even if the user has supplied some of their
            # own logging options.
            "properties": {
                "version": {
                    "type": "integer",
                    "default": 1
                },
                "disable_existing_loggers": {
                    "description": "For reasons that are baffling to me, Python's logging.config.dictConfig()\n"
                                   "sets logger.disabled = True for all existing loggers unless you explicitly tell it not to.\n",
                    "type": "boolean",
                    "default": False
                },
                "formatters": {
                    "type": "object",
                    "default": {},
                    "properties": {
                        "timestamped": {
                            "default": {
                                "format": "[%(asctime)s] %(levelname)s %(message)s"
                            }
                        }
                    }
                }
            },
            "handlers": {
                "type": "object",
                "default": {},
                "properties": {
                    "console": {
                        "type": "object",
                        "default": {
                            "level": "DEBUG",
                            "class": "logging.StreamHandler",
                            "stream": "ext://sys.stdout",
                            "formatter": "timestamped",
                        }
                    }
                }
            },
            "root": {
                "type": "object",
                "default": {
                    "handlers": ['console'],
                    "level": "INFO"
                }
            },
            "loggers": {
                "type": "object",
                "default": {
                    "distributed.client": {"level": "WARNING"},
                    "bokeh": {"level": "ERROR"},
                    "tornado": {"level": "CRITICAL"},
                    "tornado.application": {"level": "ERROR"},
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
        "jobqueue": JobQueueSchema,
        "distributed": DistributedSchema,
        **DaskBaseSchemaProperties
    }
}
