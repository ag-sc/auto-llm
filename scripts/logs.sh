#!/bin/bash
# Usage:
#   ./scripts/logs.sh           → tail the latest log file
#   ./scripts/logs.sh -f        → follow (live stream) the latest log
#   ./scripts/logs.sh JOBID     → tail a specific job's log
#   ./scripts/logs.sh -f JOBID  → follow a specific job's log
#   ./scripts/logs.sh -l        → list all your log files (newest first)

LOG_DIR="/vol/auto_llm/logs"
USER=$(whoami)

if [[ "$1" == "-l" ]]; then
    echo "=== Your logs (newest first) ==="
    ls -lt "$LOG_DIR"/*-"$USER"-*.out 2>/dev/null | head -20
    exit 0
fi

FOLLOW=false
JOBID=""

for arg in "$@"; do
    if [[ "$arg" == "-f" ]]; then
        FOLLOW=true
    else
        JOBID="$arg"
    fi
done

if [[ -n "$JOBID" ]]; then
    FILE=$(ls -t "$LOG_DIR"/*-"$USER"-"$JOBID".out 2>/dev/null | head -1)
else
    FILE=$(ls -t "$LOG_DIR"/*-"$USER"-*.out 2>/dev/null | head -1)
fi

if [[ -z "$FILE" ]]; then
    echo "No log files found in $LOG_DIR for user $USER"
    exit 1
fi

echo "=== $FILE ==="

if $FOLLOW; then
    tail -f "$FILE"
else
    tail -80 "$FILE"
fi
