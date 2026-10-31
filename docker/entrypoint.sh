#!/bin/bash
# Container equivalent of scripts/autollm_ui.sh / scripts/autollm_ui_prod.sh.
# Usage: entrypoint [prod|dev]   (default: prod)
set -euo pipefail

MODE="${1:-prod}"
REPO_DIR="${AUTOLLM_REPO_DIR:-/home/ubuntu/auto-llm}"
ENV_VARIABLES_PATH="${ENV_VARIABLES_PATH:-/home/ubuntu/env.sh}"

# The UI queues GPU jobs on the host's task-spooler server. If the socket is missing,
# `ts` would quietly start its own server inside this container (no GPUs, wrong env),
# so refuse to start instead.
if [ ! -S "${TS_SOCKET:-}" ]; then
    echo "ERROR: task-spooler socket '${TS_SOCKET:-<unset>}' not found." >&2
    echo "Start the ts server on the host first, then restart this container." >&2
    exit 1
fi
if ! command -v ts >/dev/null; then
    echo "ERROR: 'ts' binary not mounted into the container." >&2
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
