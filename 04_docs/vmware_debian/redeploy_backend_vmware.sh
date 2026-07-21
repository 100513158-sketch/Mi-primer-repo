#!/usr/bin/env bash

set -euo pipefail

SOURCE_VMWARE_DIR="${SOURCE_VMWARE_DIR:-}"
TARGET_ROOT="${TARGET_ROOT:-/opt/sarc_drone_backend}"
TARGET_VMWARE_DIR="$TARGET_ROOT/sarc_drone/04_docs/vmware_debian"
TARGET_BACKEND_DIR="$TARGET_VMWARE_DIR/backend"
VENV_DIR="${VENV_DIR:-$TARGET_ROOT/venv}"
SERVICE_NAME="${SERVICE_NAME:-sarc-backend}"
SERVICE_USER="${SERVICE_USER:-sarcsvc}"
ENV_FILE="${ENV_FILE:-$TARGET_ROOT/.env}"

if [[ -z "$SOURCE_VMWARE_DIR" ]]; then
  echo "ERROR: set SOURCE_VMWARE_DIR to the source 04_docs/vmware_debian directory" >&2
  echo "Example: SOURCE_VMWARE_DIR=/mnt/hgfs/SARC-Drone/04_docs/vmware_debian $0" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_VMWARE_DIR/backend/app.py" ]]; then
  echo "ERROR: backend/app.py not found under $SOURCE_VMWARE_DIR" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

echo "[1/6] Syncing VMware deployment files into /opt..."
sudo mkdir -p "$TARGET_VMWARE_DIR"
sudo rsync -av --delete --exclude '.env' "$SOURCE_VMWARE_DIR/" "$TARGET_VMWARE_DIR/"

echo "[2/6] Ensuring backend virtualenv exists..."
if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  sudo python3 -m venv "$VENV_DIR"
fi

echo "[3/6] Installing backend dependencies..."
sudo "$VENV_DIR/bin/pip" install --upgrade pip
sudo "$VENV_DIR/bin/pip" install -r "$TARGET_BACKEND_DIR/requirements.txt"

echo "[4/6] Fixing ownership and permissions..."
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$TARGET_ROOT/sarc_drone" "$VENV_DIR"
sudo chmod 750 "$TARGET_ROOT" "$TARGET_ROOT/sarc_drone" "$VENV_DIR"
sudo chown root:"$SERVICE_USER" "$ENV_FILE"
sudo chmod 640 "$ENV_FILE"

echo "[5/6] Restarting backend service..."
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager -l | sed -n '1,20p'

echo "[6/6] Checking health..."
curl -fsS http://127.0.0.1:8000/health