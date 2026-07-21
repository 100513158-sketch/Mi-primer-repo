#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="${ENV_FILE:-/opt/sarc_drone_backend/.env}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
DRONE_ID="${DRONE_ID:-sarc_drone_001}"

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
: "${MQTT_USERNAME:?MQTT_USERNAME missing in $ENV_FILE}"
: "${MQTT_PASSWORD:?MQTT_PASSWORD missing in $ENV_FILE}"

DRONE_MQTT_USERNAME="${DRONE_MQTT_USERNAME:-drone_client}"
: "${DRONE_MQTT_PASSWORD:?Set DRONE_MQTT_PASSWORD for the drone_client MQTT user}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

health_file="$tmp_dir/health.json"
mqtt_file="$tmp_dir/mqtt.txt"

command_id="cmd-smoke-$(date +%s)"
topic="sarc/commands/drone/${DRONE_ID}"

echo "[1/5] Checking health endpoint..."
curl -fsS "$API_BASE/health" -o "$health_file"
python3 - "$health_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload.get("status") == "ok", payload
assert payload.get("database") == "sarc_drone", payload
assert payload.get("schema") == "sarc_drone", payload
assert payload.get("mqtt_ready") is True, payload

print("health_ok")
PY

echo "[2/5] Subscribing to MQTT commands topic..."
mosquitto_sub -h "$MQTT_HOST" -p "$MQTT_PORT" -u "$DRONE_MQTT_USERNAME" -P "$DRONE_MQTT_PASSWORD" -t "$topic" -C 2 -v > "$mqtt_file" &
sub_pid=$!
trap 'kill "$sub_pid" 2>/dev/null || true; rm -rf "$tmp_dir"' EXIT

echo "[3/5] Sending REQUEST_TELEMETRY and FOLLOW_TARGET..."
curl -fsS -X POST "$API_BASE/command/$DRONE_ID" \
  -H 'Content-Type: application/json' \
  -d "{\"id\":\"$command_id\",\"type\":\"REQUEST_TELEMETRY\"}" > "$tmp_dir/command.json"

curl -fsS -X POST "$API_BASE/follow/$DRONE_ID?enabled=true" > "$tmp_dir/follow.json"

wait "$sub_pid"
trap 'rm -rf "$tmp_dir"' EXIT

echo "[4/5] Validating MQTT payloads..."
grep -F "$command_id" "$mqtt_file" > /dev/null
grep -F 'REQUEST_TELEMETRY' "$mqtt_file" > /dev/null
grep -F 'FOLLOW_TARGET' "$mqtt_file" > /dev/null

python3 - "$tmp_dir/follow.json" "$DRONE_ID" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload.get("status") == "sent", payload
assert payload.get("drone_id") == sys.argv[2], payload
assert payload.get("topic") == f"sarc/commands/drone/{sys.argv[2]}", payload

print(payload.get("command_id"))
PY

follow_id="$(python3 - "$tmp_dir/follow.json" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    payload = json.load(handle)

print(payload['command_id'])
PY
)"

grep -F "$follow_id" "$mqtt_file" > /dev/null

echo "[5/5] Validating PostgreSQL persistence..."
command_row="$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc "SELECT command_id || '|' || command_type || '|' || drone_id || '|' || status FROM sarc_drone.commands WHERE command_id = '$command_id' ORDER BY created_at DESC LIMIT 1;")"
follow_row="$(PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc "SELECT command_id || '|' || command_type || '|' || drone_id || '|' || status FROM sarc_drone.commands WHERE command_id = '$follow_id' ORDER BY created_at DESC LIMIT 1;")"

[[ "$command_row" == *"$command_id|REQUEST_TELEMETRY|$DRONE_ID|"* ]]
[[ "$follow_row" == *"$follow_id|FOLLOW_TARGET|$DRONE_ID|"* ]]

echo "Smoke test OK"
echo "- Health endpoint passed"
echo "- MQTT command topic received REQUEST_TELEMETRY and FOLLOW_TARGET"
echo "- PostgreSQL persisted both commands"