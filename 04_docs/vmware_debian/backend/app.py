from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from paho.mqtt import client as mqtt
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import SimpleConnectionPool

APP_NAME = "SARC Edge Backend"
PGHOST = os.getenv("PGHOST", "postgres")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "sarc_drone")
PGUSER = os.getenv("PGUSER", "sarc_admin")
PGPASSWORD = os.getenv("PGPASSWORD", "change_me_postgres")
PGSCHEMA = os.getenv("PGSCHEMA", "sarc_drone")

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "backend_service")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "change_me_backend")
MQTT_TOPIC_BASE = os.getenv("MQTT_TOPIC_BASE", "sarc/drone/#")
MQTT_COMMAND_PREFIX = os.getenv("MQTT_COMMAND_PREFIX", "sarc/commands/drone")
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "sarc-edge-backend")

app = FastAPI(title=APP_NAME, version="1.0.0")

_db_pool: SimpleConnectionPool | None = None
_mqtt_client: mqtt.Client | None = None
_mqtt_ready = threading.Event()


def _schema_qualified(table_name: str) -> str:
    return f"{PGSCHEMA}.{table_name}"


def _db_dsn() -> str:
    return (
        f"host={PGHOST} port={PGPORT} dbname={PGDATABASE} "
        f"user={PGUSER} password={PGPASSWORD}"
    )


def _get_db_conn():
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized")
    return _db_pool.getconn()


def _put_db_conn(conn) -> None:
    if _db_pool is not None:
        _db_pool.putconn(conn)


def init_db_pool() -> None:
    global _db_pool
    if _db_pool is None:
        _db_pool = SimpleConnectionPool(1, 10, _db_dsn())


def ensure_schema() -> None:
    conn = _get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {PGSCHEMA};")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {PGSCHEMA}.events (
                        id BIGSERIAL PRIMARY KEY,
                        drone_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        source_timestamp TIMESTAMPTZ NULL,
                        received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {PGSCHEMA}.pose_events (
                        id BIGSERIAL PRIMARY KEY,
                        drone_id TEXT NOT NULL,
                        class_name TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        topic TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        source_timestamp TIMESTAMPTZ NULL,
                        received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {PGSCHEMA}.commands (
                        id BIGSERIAL PRIMARY KEY,
                        command_id TEXT NOT NULL UNIQUE,
                        drone_id TEXT NOT NULL,
                        command_type TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        payload JSONB NOT NULL,
                        status TEXT NOT NULL DEFAULT 'SENT',
                        requested_by TEXT NOT NULL DEFAULT 'backend_api',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_events_drone_time ON {PGSCHEMA}.events (drone_id, received_at DESC);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_events_type_time ON {PGSCHEMA}.events (event_type, received_at DESC);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_pose_events_drone_time ON {PGSCHEMA}.pose_events (drone_id, received_at DESC);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_commands_drone_status ON {PGSCHEMA}.commands (drone_id, status, updated_at DESC);"
                )
    finally:
        _put_db_conn(conn)


def _safe_json(payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
        if isinstance(decoded, dict):
            return decoded
        return {"value": decoded}
    except Exception:
        return {"raw": payload.decode("utf-8", errors="replace")}


def _event_type_from_topic(topic: str) -> str:
    if topic.endswith("/telemetry"):
        return "telemetry"
    if topic.endswith("/detections"):
        return "detections"
    if topic.endswith("/tracking"):
        return "tracking"
    if topic.endswith("/pose"):
        return "pose"
    if topic.endswith("/ack"):
        return "ack"
    return "unknown"


def _extract_drone_id(topic: str, payload: dict[str, Any]) -> str:
    candidates = [
        payload.get("drone_id"),
        payload.get("device_id"),
        payload.get("source_id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    topic_parts = topic.split("/")
    if len(topic_parts) >= 4 and topic_parts[-2] == "drone":
        return topic_parts[-1]
    return "unknown"


def store_event(topic: str, payload: dict[str, Any]) -> None:
    drone_id = _extract_drone_id(topic, payload)
    event_type = _event_type_from_topic(topic)
    source_ts = payload.get("timestamp")
    source_dt = None
    if isinstance(source_ts, (int, float)):
        source_dt = datetime.fromtimestamp(float(source_ts) / 1000.0, tz=timezone.utc)

    conn = _get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {PGSCHEMA}.events (
                        drone_id, event_type, topic, payload, source_timestamp
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (drone_id, event_type, topic, Json(payload), source_dt),
                )

                if event_type == "pose":
                    cur.execute(
                        f"""
                        INSERT INTO {PGSCHEMA}.pose_events (
                            drone_id, class_name, confidence, topic, payload, source_timestamp
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            drone_id,
                            str(payload.get("class", payload.get("class_name", "unknown"))),
                            float(payload.get("confidence", 0.0)),
                            topic,
                            Json(payload),
                            source_dt,
                        ),
                    )

                if event_type == "ack" and payload.get("command_id"):
                    cur.execute(
                        f"""
                        UPDATE {PGSCHEMA}.commands
                           SET status = %s,
                               updated_at = NOW()
                         WHERE command_id = %s
                        """,
                        (str(payload.get("status", "ACK")), str(payload.get("command_id"))),
                    )
    finally:
        _put_db_conn(conn)


def store_command(drone_id: str, command: dict[str, Any], topic: str) -> dict[str, Any]:
    command_id = str(command.get("id") or command.get("command_id") or f"cmd-{int(time.time() * 1000)}")
    command_type = str(command.get("type") or "UNKNOWN")
    payload = dict(command)
    payload["id"] = command_id
    payload["command_id"] = command_id
    payload["drone_id"] = drone_id

    conn = _get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {PGSCHEMA}.commands (
                        command_id, drone_id, command_type, topic, payload, status, requested_by
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (command_id)
                    DO UPDATE SET
                        drone_id = EXCLUDED.drone_id,
                        command_type = EXCLUDED.command_type,
                        topic = EXCLUDED.topic,
                        payload = EXCLUDED.payload,
                        status = EXCLUDED.status,
                        updated_at = NOW()
                    """,
                    (
                        command_id,
                        drone_id,
                        command_type,
                        topic,
                        Json(payload),
                        "SENT",
                        "backend_api",
                    ),
                )
    finally:
        _put_db_conn(conn)

    return payload


def _build_follow_command(drone_id: str, enabled: bool = True) -> dict[str, Any]:
    return {
        "id": f"follow-{drone_id}-{int(time.time() * 1000)}",
        "type": "FOLLOW_TARGET",
        "enabled": enabled,
        "target_drone_id": drone_id,
    }


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        client.subscribe(MQTT_TOPIC_BASE, qos=1)
        _mqtt_ready.set()


def on_message(client, userdata, msg):
    payload = _safe_json(msg.payload)
    store_event(msg.topic, payload)


def start_mqtt() -> None:
    global _mqtt_client
    if _mqtt_client is not None:
        return

    try:
        _mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
        _mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        _mqtt_client.on_connect = on_connect
        _mqtt_client.on_message = on_message
        _mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
        _mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        _mqtt_client.loop_start()
    except Exception as exc:
        _mqtt_ready.clear()
        _mqtt_client = None
        print(f"[WARN] MQTT no disponible aun: {exc}")


@app.on_event("startup")
def on_startup() -> None:
    init_db_pool()
    ensure_schema()
    start_mqtt()


@app.on_event("shutdown")
def on_shutdown() -> None:
    global _mqtt_client, _db_pool
    if _mqtt_client is not None:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        _mqtt_client = None
    if _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mqtt_ready": _mqtt_ready.is_set(),
        "database": PGDATABASE,
        "schema": PGSCHEMA,
    }


@app.get("/events/{drone_id}")
def latest_events(drone_id: str, limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 500))
    conn = _get_db_conn()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, drone_id, event_type, topic, payload, source_timestamp, received_at
                      FROM {PGSCHEMA}.events
                     WHERE drone_id = %s
                     ORDER BY received_at DESC
                     LIMIT %s
                    """,
                    (drone_id, limit),
                )
                rows = cur.fetchall()
                return [dict(row) for row in rows]
    finally:
        _put_db_conn(conn)


def _console_html() -> str:
        return """
<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SARC Drone Console</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #0b1020;
            --panel: #121a33;
            --panel-2: #182241;
            --accent: #7dd3fc;
            --accent-2: #34d399;
            --text: #e5eefc;
            --muted: #9fb0d0;
            --danger: #f87171;
            --border: rgba(125, 211, 252, 0.18);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Inter, Segoe UI, system-ui, sans-serif;
            background: radial-gradient(circle at top, #172554 0%, var(--bg) 45%, #050816 100%);
            color: var(--text);
        }
        .wrap {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }
        .hero {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        .card {
            background: linear-gradient(180deg, rgba(18,26,51,.96), rgba(10,15,32,.96));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 12px 48px rgba(0,0,0,.3);
        }
        h1, h2, h3 { margin: 0 0 12px 0; }
        h1 { font-size: 28px; }
        p { margin: 8px 0; color: var(--muted); }
        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 16px;
        }
        .col-7 { grid-column: span 7; }
        .col-5 { grid-column: span 5; }
        .field-row {
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 10px;
            align-items: end;
        }
        label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
        input, select, button, textarea {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--panel-2);
            color: var(--text);
            padding: 12px 14px;
            font: inherit;
        }
        textarea {
            min-height: 150px;
            resize: vertical;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            white-space: pre-wrap;
        }
        button {
            cursor: pointer;
            background: linear-gradient(135deg, var(--accent), #60a5fa);
            color: #04111f;
            font-weight: 700;
            border: none;
        }
        button.secondary {
            background: transparent;
            color: var(--text);
            border: 1px solid var(--border);
            font-weight: 600;
        }
        .btn-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 12px;
        }
        .status {
            padding: 10px 12px;
            border-radius: 12px;
            background: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.22);
            color: #d1fae5;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .status.error {
            background: rgba(248, 113, 113, 0.12);
            border-color: rgba(248, 113, 113, 0.25);
            color: #fee2e2;
        }
        .event-list {
            display: grid;
            gap: 10px;
            max-height: 680px;
            overflow: auto;
        }
        .event {
            border: 1px solid var(--border);
            background: rgba(24,34,65,.7);
            border-radius: 14px;
            padding: 12px;
        }
        .event-head {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
            margin-bottom: 8px;
            font-size: 13px;
            color: var(--muted);
        }
        .pill {
            display: inline-flex;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(125, 211, 252, 0.12);
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
        }
        pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            color: #dbeafe;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            font-size: 12px;
        }
        @media (max-width: 980px) {
            .hero, .col-7, .col-5 { grid-template-columns: 1fr; grid-column: span 12; }
            .field-row, .btn-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="hero">
            <div class="card">
                <h1>SARC Drone Console</h1>
                <p>Consola remota para ver eventos estructurados y mandar comandos al dron desde la VM Debian.</p>
                <div id="health" class="status">Conectando...</div>
            </div>
            <div class="card">
                <h3>Comandos remotos</h3>
                <p>FOLLOW_TARGET, ABORT_MISSION y REQUEST_TELEMETRY</p>
                <div class="btn-row">
                    <button onclick="sendPreset('FOLLOW_TARGET', true)">Follow ON</button>
                    <button class="secondary" onclick="sendPreset('FOLLOW_TARGET', false)">Follow OFF</button>
                    <button onclick="sendPreset('REQUEST_TELEMETRY')">Telemetry</button>
                    <button style="background:linear-gradient(135deg,#f87171,#fb7185); color:#1f0a0a;" onclick="sendPreset('ABORT_MISSION')">Abort</button>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card col-7">
                <h3>Eventos recientes</h3>
                <div class="field-row">
                    <div>
                        <label for="droneId">Drone ID</label>
                        <input id="droneId" value="sarc_drone_001" />
                    </div>
                    <div>
                        <label for="limit">Límite</label>
                        <input id="limit" type="number" min="1" max="500" value="25" />
                    </div>
                    <button onclick="loadEvents()">Actualizar</button>
                </div>
                <div style="margin-top:12px" id="events" class="event-list"></div>
            </div>

            <div class="card col-5">
                <h3>Respuesta / payload</h3>
                <textarea id="result" readonly>Listo para enviar comandos.</textarea>
            </div>
        </div>
    </div>

    <script>
        const healthEl = document.getElementById('health');
        const eventsEl = document.getElementById('events');
        const resultEl = document.getElementById('result');
                            <div>
                                <label for="filter">Filtro</label>
                                <select id="filter">
                                    <option value="all">Todos</option>
                                    <option value="telemetry">Telemetry</option>
                                    <option value="detections">Detections</option>
                                    <option value="tracking">Tracking</option>
                                    <option value="pose">Pose</option>
                                    <option value="ack">ACK</option>
                                </select>
                            </div>
                            <button onclick="loadEvents()">Actualizar</button>
        function droneId() { return document.getElementById('droneId').value.trim(); }
                        <div class="btn-row" style="grid-template-columns: repeat(3, 1fr); margin-top: 12px;">
                            <button class="secondary" onclick="exportJson()">Exportar JSON</button>
                            <button class="secondary" onclick="rememberDroneId()">Guardar Drone ID</button>
                            <button class="secondary" onclick="forgetDroneId()">Olvidar Drone ID</button>
                        </div>
                        <div class="stats">
                            <div class="stat"><div class="value" id="stat-events">0</div><div class="label">Eventos</div></div>
                            <div class="stat"><div class="value" id="stat-telemetry">0</div><div class="label">Telemetry</div></div>
                            <div class="stat"><div class="value" id="stat-tracking">0</div><div class="label">Tracking</div></div>
                            <div class="stat"><div class="value" id="stat-pose">0</div><div class="label">Pose</div></div>
                        </div>
        function limit() { return parseInt(document.getElementById('limit').value || '25', 10); }

        function setStatus(message, isError = false) {
            healthEl.className = isError ? 'status error' : 'status';
            healthEl.textContent = message;
        }

        async function refreshHealth() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                setStatus(`OK | MQTT: ${data.mqtt_ready ? 'ready' : 'down'} | DB: ${data.database}.${data.schema}`);
            } catch (error) {
                setStatus(`Error de salud: ${error}`, true);
                const statEventsEl = document.getElementById('stat-events');
                const statTelemetryEl = document.getElementById('stat-telemetry');
                const statTrackingEl = document.getElementById('stat-tracking');
                const statPoseEl = document.getElementById('stat-pose');
                const droneIdEl = document.getElementById('droneId');
                const filterEl = document.getElementById('filter');

                const savedDroneId = localStorage.getItem('sarc_console_drone_id');
                if (savedDroneId) {
                    droneIdEl.value = savedDroneId;
                }
            }
        }

                function selectedFilter() { return filterEl.value; }
        function pretty(obj) {
            return JSON.stringify(obj, null, 2);
        }

        async function loadEvents() {
            const id = droneId();
            if (!id) {
                resultEl.value = 'Debes indicar un drone_id';
                return;
            }
            try {
                const res = await fetch(`/events/${encodeURIComponent(id)}?limit=${limit()}`);
                const data = await res.json();
                eventsEl.innerHTML = '';
                if (!data.length) {
                    eventsEl.innerHTML = '<div class="event">Sin eventos recientes.</div>';
                    return;
                }
                for (const item of data) {
                    const event = document.createElement('div');
                    event.className = 'event';
                    event.innerHTML = `
                        <div class="event-head">
                            <span class="pill">${item.event_type}</span>
                            <span>${item.received_at || ''}</span>
                        </div>
                        <div><strong>${item.topic}</strong></div>
                        <pre>${pretty(item.payload)}</pre>
                    `;
                        const filtered = data.filter(eventMatchesFilter);
                    eventsEl.appendChild(event);
                        if (!filtered.length) {
            } catch (error) {
                resultEl.value = `Error cargando eventos: ${error}`;
            }
                        renderQuickViews(filtered);
                        for (const item of filtered) {

        async function sendCommand(command) {
            const id = droneId();
            if (!id) {
                resultEl.value = 'Debes indicar un drone_id';
                return;
            }
            const response = await fetch(`/command/${encodeURIComponent(id)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(command)
            });
            const payload = await response.json();
            resultEl.value = pretty(payload);
            await loadEvents();
        }

        async function sendPreset(type, enabled = true) {
            if (type === 'FOLLOW_TARGET') {
                const response = await fetch(`/follow/${encodeURIComponent(droneId())}?enabled=${enabled}`, { method: 'POST' });
                const payload = await response.json();
                resultEl.value = pretty(payload);
                await loadEvents();
                return;
            }
            await sendCommand({ id: `ui-${Date.now()}`, type });
        }

        refreshHealth();
        loadEvents();
        setInterval(loadEvents, 5000);
        setInterval(refreshHealth, 10000);
    </script>
</body>
</html>
"""


@app.get("/console", response_class=HTMLResponse)
def console() -> HTMLResponse:
    return HTMLResponse(_console_html())


@app.post("/command/{drone_id}")
def send_command(drone_id: str, command: dict[str, Any]) -> dict[str, Any]:
    if "type" not in command:
        raise HTTPException(status_code=400, detail="command.type es obligatorio")

    topic = f"{MQTT_COMMAND_PREFIX}/{drone_id}"
    payload = store_command(drone_id, command, topic)

    if _mqtt_client is None:
        raise HTTPException(status_code=503, detail="MQTT no inicializado")

    result = _mqtt_client.publish(topic, json.dumps(payload), qos=1)
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        raise HTTPException(status_code=502, detail="No se pudo publicar el comando")

    return {
        "status": "sent",
        "topic": topic,
        "command_id": payload["command_id"],
        "drone_id": drone_id,
    }


@app.post("/follow/{drone_id}")
def follow_drone(drone_id: str, enabled: bool = True) -> dict[str, Any]:
    return send_command(drone_id, _build_follow_command(drone_id, enabled=enabled))
