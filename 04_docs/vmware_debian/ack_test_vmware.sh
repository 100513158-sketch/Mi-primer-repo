#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="${ENV_FILE:-/opt/sarc_drone_backend/.env}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
DRONE_ID="${DRONE_ID:-sarc_drone_001}"
ACK_STATUS="${ACK_STATUS:-ACK}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${PGHOST:?PGHOST missing in $ENV_FILE}"
: "${PGPORT:?PGPORT missing in $ENV_FILE}"
: "${PGDATABASE:?PGDATABASE missing in $ENV_FILE}"
: "${PGUSER:?PGUSER missing in $ENV_FILE}"
: "${PGPASSWORD:?PGPASSWORD missing in $ENV_FILE}"
: "${MQTT_HOST:?MQTT_HOST missing in $ENV_FILE}"
: "${MQTT_PORT:?MQTT_PORT missing in $ENV_FILE}"

DRONE_MQTT_USERNAME="${DRONE_MQTT_USERNAME:-drone_client}"
: "${DRONE_MQTT_PASSWORD:?Set DRONE_MQTT_PASSWORD for the drone_client MQTT user}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

command_id="cmd-ack-$(date +%s)"
timestamp_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"

echo "[1/4] Sending command through backend API..."
curl -fsS -X POST "$API_BASE/command/$DRONE_ID" \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$command_id\",\"type\":\"REQUEST_TELEMETRY\"}" > "$tmp_dir/command.json"

echo "[2/4] Publishing ACK as drone_client..."
ack_payload="{\"drone_id\":\"$DRONE_ID\",\"command_id\":\"$command_id\",\"status\":\"$ACK_STATUS\",\"timestamp\":$timestamp_ms,\"source\":\"ack_test_vmware\"}"
mosquitto_pub -h "$MQTT_HOST" -p "$MQTT_PORT" -u "$DRONE_MQTT_USERNAME" -P "$DRONE_MQTT_PASSWORD" -t 'sarc/drone/ack' -m "$ack_payload"

echo "[3/4] Waiting for backend to persist ACK..."
command_status=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  command_status="$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc "SELECT status FROM sarc_drone.commands WHERE command_id = '$command_id' ORDER BY updated_at DESC LIMIT 1;")"
  if [[ "$command_status" == "$ACK_STATUS" ]]; then
    break
  fi
  sleep 1
done

if [[ "$command_status" != "$ACK_STATUS" ]]; then
  echo "ERROR: command status did not transition to $ACK_STATUS (current: ${command_status:-<empty>})" >&2
  exit 1
fi

ack_row="$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc "SELECT event_type || '|' || drone_id || '|' || COALESCE(payload->>'command_id','') || '|' || COALESCE(payload->>'status','') FROM sarc_drone.events WHERE event_type = 'ack' AND payload->>'command_id' = '$command_id' ORDER BY received_at DESC LIMIT 1;")"

echo "[4/4] Validating persisted ACK event..."
[[ "$ack_row" == *"ack|$DRONE_ID|$command_id|$ACK_STATUS"* ]]

echo "ACK test OK"
echo "- Command $command_id moved to status $ACK_STATUS"
echo "- ACK event persisted in sarc_drone.events"