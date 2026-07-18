CREATE SCHEMA IF NOT EXISTS sarc_drone AUTHORIZATION CURRENT_USER;

CREATE TABLE IF NOT EXISTS sarc_drone.events (
    id BIGSERIAL PRIMARY KEY,
    drone_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    topic TEXT NOT NULL,
    payload JSONB NOT NULL,
    source_timestamp TIMESTAMPTZ NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sarc_drone.commands (
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

CREATE INDEX IF NOT EXISTS idx_events_drone_time
    ON sarc_drone.events (drone_id, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON sarc_drone.events (event_type, received_at DESC);

CREATE INDEX IF NOT EXISTS idx_commands_drone_status
    ON sarc_drone.commands (drone_id, status, updated_at DESC);
