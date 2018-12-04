#!/bin/bash

DVID_TOML=worker-dvid-config.toml

LAUNCH_DIR="worker-dvid-logs/$(uname -n)"
mkdir -p ${LAUNCH_DIR}
cp "${DVID_TOML}" ${LAUNCH_DIR}

echo "cd ${LAUNCH_DIR}"
cd ${LAUNCH_DIR}

echo "dvid -verbose serve \"${DVID_TOML}\" &"
dvid -verbose serve "${DVID_TOML}" &

DVID_PID=$!

# If we're terminated from the outside via SIGTERM, send SIGTERM to the subprocess.
trap 'kill -TERM $DVID_PID; echo "Exiting in response to external signal"' EXIT

wait $DVID_PID
DVID_EXIT_CODE=$?
echo "DVID exited with code ${DVID_EXIT_CODE}"
