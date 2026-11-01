#!/bin/bash
# Container equivalent of scripts/autollm_ui.sh / scripts/autollm_ui_prod.sh.
# Usage: entrypoint [prod|dev]   (default: prod)
set -euo pipefail

MODE="${1:-prod}"
REPO_DIR="${AUTOLLM_REPO_DIR:-/home/ubuntu/auto-llm}"
ENV_VARIABLES_PATH="${ENV_VARIABLES_PATH:-/home/ubuntu/env.sh}"

# The UI queues GPU jobs through the host ts relay (docker/ts_relay.py), which runs `ts`
# on the host so the jobs run there. Without it, job submission would fail.
RELAY_SOCKET="${TS_RELAY_DIR:-/home/ubuntu/.autollm_ts_relay}/relay.sock"
if [ ! -S "$RELAY_SOCKET" ]; then
    echo "ERROR: host ts relay socket $RELAY_SOCKET not found." >&2
    echo "Run scripts/start_ts_queue.sh on the host first, then restart this container." >&2
    exit 1
fi

# Same secrets file the host scripts use (HF_TOKEN, WANDB_KEY, ...).
if [ -f "$ENV_VARIABLES_PATH" ]; then
    # shellcheck disable=SC1090
    source "$ENV_VARIABLES_PATH"
    # On the host wandb authenticates via ~/.netrc, which the container doesn't have.
    # wandb reads WANDB_API_KEY; env.sh names the same key WANDB_KEY.
    if [ -z "${WANDB_API_KEY:-}" ] && [ -n "${WANDB_KEY:-}" ]; then
        export WANDB_API_KEY="$WANDB_KEY"
    fi
else
    echo "WARNING: $ENV_VARIABLES_PATH not found, starting without it." >&2
fi

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR/ui"

case "$MODE" in
    prod)
        reflex export
        exec reflex run --env prod --frontend-port 3000 --backend-port 8000
        ;;
    dev)
        exec reflex run --frontend-port 3000 --backend-port 8000
        ;;
    *)
        exec "$@"
        ;;
esac
