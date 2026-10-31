#!/bin/bash
# Start the shared task-spooler queue that the AutoLLM UI submits GPU jobs to.
# Usage: scripts/start_ts_queue.sh [num_slots]   (default: number of GPUs)
#
# Safe to re-run: if a server is already listening on the socket, it does nothing.
set -euo pipefail

export TS_SOCKET="${TS_SOCKET:-/tmp/shared_gpu_queue}"
SLOTS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$SLOTS" -ge 1 ] 2>/dev/null || SLOTS=1

is_listening() {
    ss -xl 2>/dev/null | grep -qF " $TS_SOCKET "
}

if is_listening; then
    echo "task-spooler queue already running on $TS_SOCKET, nothing to do."
    exit 0
fi

if [ -e "$TS_SOCKET" ]; then
    # File exists but nothing is listening: leftover from a dead server, no jobs to lose.
    echo "Removing stale socket $TS_SOCKET (no server listening)."
    rm -f "$TS_SOCKET"
    STALE=1
fi

# Any ts command starts the server; -S sets how many jobs run at once.
TS_SLOTS="$SLOTS" ts -S "$SLOTS"

for _ in 1 2 3 4 5; do
    is_listening && break
    sleep 0.5
done
if ! is_listening; then
    echo "ERROR: task-spooler server did not come up on $TS_SOCKET." >&2
    exit 1
fi

echo "task-spooler queue started on $TS_SOCKET with $SLOTS slot(s)."
if [ "${STALE:-0}" = 1 ]; then
    if docker ps -q -f name=^autollm-ui$ 2>/dev/null | grep -q .; then
        echo "Restarting autollm-ui container so it picks up the new socket..."
        docker restart autollm-ui >/dev/null
    else
        echo "If the autollm-ui container is running, restart it: docker restart autollm-ui"
    fi
fi
