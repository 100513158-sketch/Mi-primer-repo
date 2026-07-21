# Checklist E2E - VMware Debian

Este checklist se usa para validar el backend VMware Debian de extremo a extremo.

Estado de esta sesion:
- [x] Codigo backend revisado
- [x] `config.yaml` alineado con VMware Debian
- [x] `/opt/sarc_drone_backend/.env` preparado desde `.env.example`
- [x] Mosquitto instalado en la VM
- [x] Backend Python arrancado contra PostgreSQL local
- [x] `GET /health` responde `ok`
- [x] `GET /console` carga correctamente
- [x] Publicacion MQTT de prueba verificada
- [x] Insercion en PostgreSQL verificada
- [x] Comando `FOLLOW_TARGET` verificado
- [ ] ACK del dron verificado

## 1) Preparar entorno

```bash
cp /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/.env.example /opt/sarc_drone_backend/.env
```

Revisar variables criticas en `.env`:
- `PGHOST=127.0.0.1`
- `PGPORT=5432`
- `PGDATABASE=sarc_drone`
- `PGUSER=sarc_admin`
- `PGPASSWORD`
- `PGSCHEMA=sarc_drone`
- `MQTT_USERNAME`
- `MQTT_PASSWORD`
- `MQTT_CLIENT_ID`
- `MQTT_HOST=127.0.0.1`

## 2) Crear entorno Python

```bash
cd /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/backend
python3 -m venv /opt/sarc_drone_backend/venv
source /opt/sarc_drone_backend/venv/bin/activate
pip install -r requirements.txt
```

## 3) Arrancar backend

```bash
sudo systemctl restart sarc-backend
```

## 4) Verificar salud del backend

```bash
curl http://192.168.1.134:8000/health
```

Criterio:
- `status = ok`
- `database = sarc_drone`
- `schema = sarc_drone`
- `mqtt_ready = true`

## 5) Verificar consola web

Abrir:

```text
http://192.168.1.134:8000/console
```

Criterio:
- La pagina carga.
- Se muestran eventos para `drone_id`.
- Los botones `Follow ON`, `Telemetry` y `Abort` responden.

## 6) Instalar Mosquitto y verificar MQTT

Verificar MQTT:

Suscribirse a todos los topics del dron:

```bash
mosquitto_sub -h 127.0.0.1 -p 1883 -u drone_client -P TU_PASSWORD -t 'sarc/commands/drone/#' -C 1 -v
```

Enviar comando de prueba desde la API:

```bash
curl -s -X POST http://127.0.0.1:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-e2e-1","type":"REQUEST_TELEMETRY"}'
```

Criterio:
- MQTT recibe `sarc/commands/drone/sarc_drone_001`.
- El payload contiene `REQUEST_TELEMETRY`.

## 7) Verificar PostgreSQL

Consultar la base local:

```bash
sudo -u postgres psql -d sarc_drone -c "SELECT command_id, command_type, drone_id, status, created_at FROM sarc_drone.commands ORDER BY created_at DESC LIMIT 10;"
```

Criterio:
- Aparece `cmd-e2e-1` con `REQUEST_TELEMETRY`.
- El estado queda `SENT` o el valor mas reciente asociado al comando.

## 8) Verificar comandos remotos

Enviar telemetria solicitada:

```bash
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-1","type":"REQUEST_TELEMETRY"}'
```

Activar follow:

```bash
curl -X POST "http://192.168.1.134:8000/follow/sarc_drone_001?enabled=true"
```

Abortar mision:

```bash
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-2","type":"ABORT_MISSION"}'
```

Criterio:
- El comando aparece en MQTT `sarc/commands/drone/{drone_id}`.
- El backend guarda el registro en `sarc_drone.commands`.
- Si llega ACK, se actualiza `status`.

## 9) Criterio de cierre

La prueba E2E queda aprobada cuando:
- El backend arranca sin errores.
- La consola web abre.
- MQTT recibe telemetria y comandos.
- PostgreSQL guarda eventos y comandos.
- El dron o simulador devuelve ACK.

## 10) Smoke test reutilizable

```bash
chmod +x /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/smoke_test_vmware.sh
DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/smoke_test_vmware.sh
```

## Pendiente operativo

Solo queda validar ACK real desde el dron o simulador publicando en `sarc/drone/ack`.
