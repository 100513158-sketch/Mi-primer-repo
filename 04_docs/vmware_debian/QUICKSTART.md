# SARC-Drone VMware Debian Quickstart

Guia corta para correr el backend con Python directo en la VM Debian usando la estructura validada en produccion: `/opt/sarc_drone_backend/venv`, `/opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/backend` y `/opt/sarc_drone_backend/.env`.

## 1) Preparar la VM Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Si luego vas a activar la parte MQTT en la misma VM, instala tambien:

```bash
sudo apt install -y mosquitto mosquitto-clients
```

## 2) Configurar el backend

```bash
cp /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/.env.example /opt/sarc_drone_backend/.env
```

Edita `.env` con estos valores mínimos:

- `PGHOST=127.0.0.1`
- `PGPORT=5432`
- `PGDATABASE=sarc_drone`
- `PGUSER=sarc_admin`
- `PGPASSWORD`
- `MQTT_HOST=127.0.0.1`
- `MQTT_PASSWORD`

## 3) Crear entorno Python

```bash
cd /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Arrancar el backend

```bash
cd /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/backend
source /opt/sarc_drone_backend/venv/bin/activate
export $(grep -v '^#' /opt/sarc_drone_backend/.env | xargs)
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 5) Verificar salud

```bash
curl http://192.168.1.134:8000/health
```

Esperado:

- `status=ok`
- `database=sarc_drone`
- `schema=sarc_drone`
- `mqtt_ready=true` cuando Mosquitto y credenciales ya estan bien

## 6) Abrir la consola remota

```text
http://192.168.1.134:8000/console
```

Desde ahi puedes ver eventos, enviar comandos y validar el flujo completo si Mosquitto ya esta operativo.

## 7) Verificar MQTT y base de datos

Suscribirse a comandos del dron:

```bash
mosquitto_sub -h 127.0.0.1 -p 1883 -u drone_client -P TU_PASSWORD -t 'sarc/commands/drone/#' -C 1 -v
```

Consultar ultimos comandos persistidos:

```bash
sudo -u postgres psql -d sarc_drone -c "SELECT command_id, command_type, drone_id, status, created_at FROM sarc_drone.commands ORDER BY created_at DESC LIMIT 10;"
```

## 8) Probar comandos

```bash
curl -X POST "http://192.168.1.134:8000/follow/sarc_drone_001?enabled=true"
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-1","type":"REQUEST_TELEMETRY"}'
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-2","type":"ABORT_MISSION"}'
```

## 9) Ejecutar smoke test completo

```bash
chmod +x /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/smoke_test_vmware.sh
DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/smoke_test_vmware.sh
```

## 10) Validar ACK del dron o simulador

```bash
chmod +x /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/ack_test_vmware.sh
DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/ack_test_vmware.sh
```

## 11) Redeploy rapido hacia /opt

Si actualizas `04_docs/vmware_debian` desde una carpeta compartida o un snapshot del repo:

```bash
chmod +x /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/redeploy_backend_vmware.sh
SOURCE_VMWARE_DIR=/mnt/hgfs/SARC-Drone/04_docs/vmware_debian /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/redeploy_backend_vmware.sh
```

## 12) Ejecutar suite E2E completa en un comando

```bash
chmod +x /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/e2e_suite_vmware.sh
DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/e2e_suite_vmware.sh
```

Si quieres exigir tambien telemetria en vivo desde la app Android:

```bash
LIVE_TELEMETRY_CHECK=1 DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/e2e_suite_vmware.sh
```
