#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${ENV_FILE:-/opt/sarc_drone_backend/.env}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
DRONE_ID="${DRONE_ID:-sarc_drone_001}"
LIVE_TELEMETRY_CHECK="${LIVE_TELEMETRY_CHECK:-0}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${MQTT_HOST:?MQTT_HOST missing in $ENV_FILE}"
: "${MQTT_PORT:?MQTT_PORT missing in $ENV_FILE}"
: "${DRONE_MQTT_PASSWORD:?Set DRONE_MQTT_PASSWORD for the drone_client MQTT user}"

DRONE_MQTT_USERNAME="${DRONE_MQTT_USERNAME:-drone_client}"

echo "=== SARC VMware E2E Suite ==="
echo "ENV_FILE=$ENV_FILE"
echo "API_BASE=$API_BASE"
echo "DRONE_ID=$DRONE_ID"

echo "[1/3] Running smoke test (health + command publish + DB persistence)..."
ENV_FILE="$ENV_FILE" API_BASE="$API_BASE" DRONE_ID="$DRONE_ID" DRONE_MQTT_USERNAME="$DRONE_MQTT_USERNAME" DRONE_MQTT_PASSWORD="$DRONE_MQTT_PASSWORD" \
  "$SCRIPT_DIR/smoke_test_vmware.sh"

echo "[2/3] Running ACK test (command status transition + ACK event)..."
ENV_FILE="$ENV_FILE" API_BASE="$API_BASE" DRONE_ID="$DRONE_ID" DRONE_MQTT_USERNAME="$DRONE_MQTT_USERNAME" DRONE_MQTT_PASSWORD="$DRONE_MQTT_PASSWORD" \
  "$SCRIPT_DIR/ack_test_vmware.sh"

echo "[3/3] Optional live telemetry check..."
if [[ "$LIVE_TELEMETRY_CHECK" == "1" ]]; then
  echo "Waiting up to 12s for one telemetry message on sarc/drone/telemetry..."
  if mosquitto_sub -h "$MQTT_HOST" -p "$MQTT_PORT" -u "$DRONE_MQTT_USERNAME" -P "$DRONE_MQTT_PASSWORD" \
    -t 'sarc/drone/telemetry' -C 1 -W 12 -v; then
    echo "Telemetry check OK"
  else
    echo "ERROR: Telemetry check failed. Ensure Android app is running and connected." >&2
    exit 1
  fi
else
  echo "Skipped (set LIVE_TELEMETRY_CHECK=1 to enable)."
fi

echo "=== E2E Suite OK ==="
echo "- Smoke test passed"
echo "- ACK test passed"
if [[ "$LIVE_TELEMETRY_CHECK" == "1" ]]; then
  echo "- Telemetry check passed"
fi
