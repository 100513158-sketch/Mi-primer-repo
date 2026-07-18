# SARC-Drone VMware Debian Quickstart

Guia corta para levantar la solucion en la VM Debian y probar la consola remota.

## 1) Preparar la VM Debian

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
```

Cierra sesion y vuelve a entrar para que aplique el grupo `docker`.

## 2) Configurar la carpeta del proyecto

```bash
cd /ruta/a/04_docs/vmware_debian
cp .env.example .env
```

Edita `.env` con tus valores reales:

- `POSTGRES_PASSWORD`
- `MQTT_PASSWORD`
- `MQTT_HOST` si cambias la IP del broker
- `MQTT_CLIENT_ID` si cambia el dron

## 3) Crear passwords de Mosquitto

```bash
docker run --rm -it \
  -v "$PWD/mosquitto:/mosquitto/config" \
  eclipse-mosquitto \
  mosquitto_passwd -c /mosquitto/config/passwords drone_client

docker run --rm -it \
  -v "$PWD/mosquitto:/mosquitto/config" \
  eclipse-mosquitto \
  mosquitto_passwd /mosquitto/config/passwords backend_service
```

## 4) Levantar la pila

```bash
docker compose up -d --build
```

Servicios esperados:

- `sarc-postgres`
- `sarc-mosquitto`
- `sarc-edge-backend`

## 5) Verificar salud

```bash
curl http://IP_DE_LA_VM:8000/health
```

Debe responder con `status=ok` y `mqtt_ready=true`.

## 6) Abrir la consola remota

```text
http://IP_DE_LA_VM:8000/console
```

Desde ahi puedes:

- ver eventos recientes
- filtrar por `telemetry`, `detections`, `tracking`, `pose` o `ack`
- guardar el `drone_id`
- exportar JSON
- enviar `FOLLOW_TARGET`
- enviar `ABORT_MISSION`
- enviar `REQUEST_TELEMETRY`

## 7) Pruebas rapidas

### Enviar seguimiento remoto

```bash
curl -X POST "http://IP_DE_LA_VM:8000/follow/sarc_drone_001?enabled=true"
```

### Pedir telemetria

```bash
curl -X POST http://IP_DE_LA_VM:8000/command/sarc_drone_001 \
  -H 'Content-Type: application/json' \
  -d '{"id":"cmd-1","type":"REQUEST_TELEMETRY"}'
```

### Abortar mision

```bash
curl -X POST http://IP_DE_LA_VM:8000/command/sarc_drone_001 \
  -H 'Content-Type: application/json' \
  -d '{"id":"cmd-2","type":"ABORT_MISSION"}'
```

## 8) Comprobaciones esperadas

- En MQTT aparece el comando en `sarc/commands/drone/sarc_drone_001`.
- La app responde con ACK en `sarc/drone/ack`.
- PostgreSQL guarda registros en `sarc_drone.events` y `sarc_drone.commands`.
- Si luego publicas `pose`, se guarda en `sarc_drone.pose_events`.

## 9) Parar la pila

```bash
docker compose down
```

Si quieres reiniciar desde cero:

```bash
docker compose down -v
```
