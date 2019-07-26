from dvid_resource_manager.server import DEFAULT_CONFIG as DEFAULT_RESOURCE_MANAGER_CONFIG
from .dask_schema import JobQueueSchema

EnvironmentVariablesSchema = \
{
    "type": "object",
    "default": {},
    "additionalProperties": { "type": "string" },
    "description": "Extra environment variables to set on the driver and workers.\n"
                   "Some are provided by default, but you may add any others you like.\n",
    "properties": {
        "OMP_NUM_THREADS": {
            "description": "Some pandas and numpy functions will use OpenMP (via MKL or OpenBLAS),\n"
                           "which causes each process to use many threads.\n"
                           "That's bad, since you can end up with N^2 threads on a machine with N cores.\n"
                           "Unless you know what you're doing, it's best to force OpenMP to use only 1 core per process.\n",
            "type": "string",
            "default": "1"
        }
    }
}


ExitEmailSchema = {
    "type": "object",
    "default": {},
    "additionalProperties": { "type": "string" },
    "description": "Extra environment variables to set on the driver and workers.\n"
                   "Some are provided by default, but you may add any others you like.\n",
    "properties": {
        "send": {
            "description": "If True, an email will be sent to the list specified in 'addresses'.",
            "type": "boolean",
            "default": False # Disabled by default because it doesn't work on all machines
                             # (it works on Janelia cluster nodes, though.)
        },
        "addresses": {
            "description": "A list of email addresses to send the exit email to when the job completes.\n"
                           "The special keyword JANELIA_USER can be used as a standin for user@janelia.hhmi.org\n",
            "type": "array",
            "default": ['JANELIA_USER']
        },
        "include-log": {
            "description": "Whether or not to include the full client log in the email body.",
            "type": "boolean",
            "default": True
        }
    }
}

ResourceManagerSchema = \
{
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "description": "Which resource manager server to use (if any) and how to configure it (if launching on the driver).",
    "properties": {
        "server": {
            "description": "If provided, workflows MAY use this resource server to coordinate competing requests from worker nodes. \n"
                           "Set to the IP address of the (already-running) resource server, or use the special word 'driver' \n"
                           "to automatically start a new resource server on the driver node.",
            "type": "string",
            "default": ""
        },
        "port": {
            "description": "Which port the resource server is running on.  (See description above.)",
            "type": "integer",
            "default": 0
        },
        "config": {
            "type": "object",
            "default": DEFAULT_RESOURCE_MANAGER_CONFIG,
            "additionalProperties": True
        }
    }
}

WorkerInitSchema = \
{
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "description": "The given script will be called once per worker node, before the workflow executes.\n"
                   "If the process is long-running, it will be killed when the workflow completes via \n"
                   "SIGINT or SIGTERM (if necessary) or SIGKILL (if really necessary).\n",
    "properties": {
        "script-path": {
            "type": "string",
            "default": ""
        },
        "script-args": {
            "type": "array",
            "items": { "type": "string" },
            "default": []
        },
        "only-once-per-machine": {
            "description": "Depending on your cluster configuration, dask might start multiple workers on a single machine.\n"
                           "Use this setting you only want this initialization script to be run ONCE per machine\n"
                           "(even if there are multiple workers on that machine)",
            "type": "boolean",
            "default": False
        },
        "launch-delay": {
            "description": "By default, wait for the script to complete before continuing.\n"
                           "Otherwise, launch the script asynchronously and then pause for N seconds before continuing.",
            "type": "integer",
            "default": -1 # default: blocking execution
        },
        "log-dir": {
            "type": "string",
            "default": "script-logs"
        },
        "also-run-on-driver": {
            "description": "Also run this initialization script on the driver machine.\n",
            "type": "boolean",
            "default": False
        }
    }
}

JOBQUEUE_CLUSTERS = ["lsf", "sge", "slurm"]
assert set(JobQueueSchema["properties"].keys()) == set(JOBQUEUE_CLUSTERS)

BaseSchema = \
{
    "type": "object",
    "description": "Workflow base config",
    "default": {},
    "additionalProperties": False,
    "required": ["workflow-name", "cluster-type"],
    "properties": {
        "workflow-name": {
            "description": "The class name of the workflow which will be executed using this config.",
            "type": "string",
            "minLength": 1
        },
        "cluster-type": {
            "description": "Whether or not to use an LSF cluster or a local cluster.\n"
                           "Choices: lsf, local-cluster, synchronous, processes",
            "type": "string",
            "enum": [*JOBQUEUE_CLUSTERS, "local-cluster", "synchronous", "processes"]
            # No default
        },
        "cluster-max-wait": {
            "description": "How many minutes to wait for all cluster workers to become available.\n"
                           "If it takes longer than this, the workflow will be aborted.\n"
                           "This timer doesn't start until the main client process starts.\n",
            "type": "integer",
            "default": 60
        },
        "resource-manager": ResourceManagerSchema,
        "worker-initialization": WorkerInitSchema,
        "environment-variables": EnvironmentVariablesSchema,
        "exit-email": ExitEmailSchema
    }
}
