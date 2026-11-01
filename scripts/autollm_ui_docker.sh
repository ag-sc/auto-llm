#!/bin/bash
# Start the AutoLLM UI in Docker (replaces autollm_ui_prod.sh / autollm_ui.sh).
# Usage: scripts/autollm_ui_docker.sh [prod|dev]   (default: prod)
set -euo pipefail

cd "$(dirname "$0")/.."
export AUTOLLM_UI_MODE="${1:-prod}"
export UID GID="$(id -g)"

# The queue and the host relay must be up before the UI starts (it refuses to start otherwise).
scripts/start_ts_queue.sh

# Free the ports if the non-docker UI is still running (same as the old scripts).
fuser -k 3000/tcp 8000/tcp 2>/dev/null || true

docker compose -f docker/docker-compose.yml up -d --build
echo "AutoLLM UI starting ($AUTOLLM_UI_MODE). Logs: docker logs -f autollm-ui"
