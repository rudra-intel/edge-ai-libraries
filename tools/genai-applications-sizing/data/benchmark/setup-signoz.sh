#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# setup-signoz.sh
# Stands up a local SigNoz stack (OTLP backend + UI) via Docker Compose.
#
# SigNoz bundles an OpenTelemetry Collector, ClickHouse, query service and UI.
# OpenClaw (with the observability plugin) exports OTLP to the collector, and
# you read the KPI spans/metrics in the SigNoz UI.
#
# Exposed endpoints after startup:
#   OTLP gRPC : 127.0.0.1:4317
#   OTLP HTTP : 127.0.0.1:4318
#   SigNoz UI : http://127.0.0.1:8080   (older builds use http://127.0.0.1:3301)
#
# Usage:
#   ./setup-signoz.sh          # clone (if needed) and start SigNoz
#   ./setup-signoz.sh down     # stop SigNoz (keeps data)
#   ./setup-signoz.sh purge    # stop SigNoz and delete volumes

set -euo pipefail

SIGNOZ_DIR="${SIGNOZ_DIR:-$HOME/signoz}"
DEPLOY_DIR="${SIGNOZ_DIR}/deploy/docker"

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed. See https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Start it with: sudo systemctl start docker"
    exit 1
fi

ACTION="${1:-up}"

if [ ! -d "${DEPLOY_DIR}" ]; then
    echo "Cloning SigNoz into ${SIGNOZ_DIR} ..."
    git clone -b main https://github.com/SigNoz/signoz.git "${SIGNOZ_DIR}"
fi

cd "${DEPLOY_DIR}"

case "${ACTION}" in
    up)
        echo "Starting SigNoz (first run pulls several images and may take a while)..."
        compose up -d
        echo ""
        echo "=== SigNoz is starting ==="
        echo "  OTLP gRPC : 127.0.0.1:4317"
        echo "  OTLP HTTP : 127.0.0.1:4318"
        echo "  UI        : http://127.0.0.1:8080  (or http://127.0.0.1:3301 on older builds)"
        echo ""
        echo "Give the stack ~1-2 minutes, then open the UI and create the default login."
        ;;
    down)
        echo "Stopping SigNoz (data kept)..."
        compose down
        ;;
    purge)
        echo "Stopping SigNoz and deleting volumes..."
        compose down -v
        ;;
    *)
        echo "Usage: $0 [up|down|purge]"
        exit 1
        ;;
esac
