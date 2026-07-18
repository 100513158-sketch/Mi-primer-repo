# SARC-Drone VMware Debian Quickstart

Guia corta para correr el backend con Python directo en la VM Debian. PostgreSQL ya esta instalado en `192.168.1.134`; Mosquitto todavia no, asi que el backend puede arrancar pero `mqtt_ready` quedara en `false` hasta instalar el broker.

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
cd /ruta/a/SARC-Drone/04_docs/vmware_debian/backend
cp .env.example .env
```

Edita `.env` con estos valores mínimos:

- `POSTGRES_HOST=192.168.1.134`
- `POSTGRES_PASSWORD`
- `MQTT_HOST=192.168.1.134` cuando tengas Mosquitto instalado
- `MQTT_PASSWORD`

## 3) Crear entorno Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Arrancar el backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 5) Verificar salud

```bash
curl http://192.168.1.134:8000/health
```

Esperado:

- `status=ok`
- `database=sarc_drone`
- `mqtt_ready=false` mientras Mosquitto no este disponible

## 6) Abrir la consola remota

```text
http://192.168.1.134:8000/console
```

Desde ahi puedes ver eventos y preparar la consola; para enviar comandos remotos hace falta que Mosquitto ya este instalado.

## 7) Cuando instales Mosquitto

1. Configura usuario y password.
2. Ajusta `MQTT_HOST=192.168.1.134` en `.env`.
3. Reinicia el backend.
4. Verifica que `mqtt_ready=true`.

## 8) Probar comandos

```bash
curl -X POST "http://192.168.1.134:8000/follow/sarc_drone_001?enabled=true"
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-1","type":"REQUEST_TELEMETRY"}'
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-2","type":"ABORT_MISSION"}'
```
