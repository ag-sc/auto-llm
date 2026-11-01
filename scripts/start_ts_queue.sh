#!/bin/bash
# Start the shared task-spooler queue that AutoLLM GPU jobs run on, plus the host relay
# that lets the Docker UI submit jobs to it (docker/ts_relay.py).
# Usage: scripts/start_ts_queue.sh [num_slots]   (default: number of GPUs)
#
# Safe to re-run: anything already running is left alone.
#
# Jobs inherit the environment of the shell that runs this script (via the relay), as they
# did when the UI ran outside Docker. The job scripts load venv/ and env.sh themselves.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export TS_SOCKET="${TS_SOCKET:-/tmp/shared_gpu_queue}"
RELAY_DIR="${TS_RELAY_DIR:-/home/ubuntu/.autollm_ts_relay}"
RELAY_SOCKET="$RELAY_DIR/relay.sock"
SLOTS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$SLOTS" -ge 1 ] 2>/dev/null || SLOTS=1

# True if some process listens on this unix socket path. Uses `ss` rather than a `ts`
# command: a `ts` call that can't connect would start a second server on the same path.
is_listening() {
    ss -xl 2>/dev/null | grep -qF " $1 "
}

wait_listening() {
    for _ in $(seq 1 10); do
        is_listening "$1" && return 0
        sleep 0.5
    done
    return 1
}

# --- 1. task-spooler queue
STALE=0
if is_listening "$TS_SOCKET"; then
    echo "task-spooler queue already running on $TS_SOCKET."
else
    if [ -e "$TS_SOCKET" ]; then
        # File exists but nothing is listening: leftover from a dead server, no jobs to lose.
        echo "Removing stale socket $TS_SOCKET (no server listening)."
        rm -f "$TS_SOCKET"
        STALE=1
    fi
    # Any ts command starts the server; -S sets how many jobs run at once.
    TS_SLOTS="$SLOTS" ts -S "$SLOTS"
    if ! wait_listening "$TS_SOCKET"; then
        echo "ERROR: task-spooler server did not come up on $TS_SOCKET." >&2
        exit 1
    fi
    echo "task-spooler queue started on $TS_SOCKET with $SLOTS slot(s)."
fi

# --- 2. host relay for the Docker UI
if is_listening "$RELAY_SOCKET"; then
    echo "ts relay already running on $RELAY_SOCKET."
else
    mkdir -p "$RELAY_DIR" && chmod 700 "$RELAY_DIR"
    # setsid + nohup: keeps running after this terminal closes.
    TS_RELAY_DIR="$RELAY_DIR" setsid nohup python3 "$REPO_DIR/docker/ts_relay.py" \
        >> "$RELAY_DIR/relay.log" 2>&1 < /dev/null &
    if ! wait_listening "$RELAY_SOCKET"; then
        echo "ERROR: ts relay did not start, see $RELAY_DIR/relay.log" >&2
        exit 1
    fi
    echo "ts relay started on $RELAY_SOCKET (log: $RELAY_DIR/relay.log)."
fi

if [ "$STALE" = 1 ]; then
    echo "Note: the queue was restarted, so earlier queued jobs are gone."
fi
