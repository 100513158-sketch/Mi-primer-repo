# Checklist E2E - VMware Debian

Este checklist se usa para validar el backend VMware Debian de extremo a extremo.

Estado de esta sesion:
- [x] Codigo backend revisado
- [x] `config.yaml` alineado con VMware Debian
- [x] `04_docs/vmware_debian/.env` preparado desde `.env.example`
- [ ] Mosquitto instalado en la VM
- [ ] Backend Python arrancado contra PostgreSQL local
- [ ] `GET /health` responde `ok`
- [ ] `GET /console` carga correctamente
- [ ] Publicacion MQTT de prueba verificada
- [ ] Insercion en PostgreSQL verificada
- [ ] Comando `FOLLOW_TARGET` verificado
- [ ] ACK del dron verificado

## 1) Preparar entorno

```powershell
cd C:\SARC-Drone\04_docs\vmware_debian\backend
copy .env.example .env
```

Revisar variables criticas en `.env`:
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `POSTGRES_SCHEMA`
- `MQTT_USERNAME`
- `MQTT_PASSWORD`
- `MQTT_CLIENT_ID`
- `POSTGRES_HOST=192.168.1.134`
- `MQTT_HOST=192.168.1.134` cuando Mosquitto ya este instalado

## 2) Crear entorno Python

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Arrancar backend

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 4) Verificar salud del backend

```powershell
curl http://192.168.1.134:8000/health
```

Criterio:
- `status = ok`
- `database = sarc_drone`
- `mqtt_ready = false` hasta instalar Mosquitto

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

Si Mosquitto aun no esta instalado en la VM:

```powershell
sudo apt install -y mosquitto mosquitto-clients
```

Luego verifica MQTT:

Suscribirse a todos los topics del dron:

```powershell
docker exec -it sarc-mosquitto mosquitto_sub -h localhost -p 1883 -u drone_client -P TU_PASSWORD -t 'sarc/drone/#' -v
```

Publicar telemetria de prueba:

```powershell
docker exec -it sarc-mosquitto mosquitto_pub -h localhost -p 1883 -u drone_client -P TU_PASSWORD -t sarc/drone/telemetry -m '{"drone_id":"sarc_drone_001","timestamp":1730000000000,"lat":0.0,"lon":0.0,"alt":20.0,"search_pattern":"lawnmower","fps":12.5}'
```

Criterio:
- El backend recibe el mensaje.
- Se ve en `/console`.

## 7) Verificar PostgreSQL

Entrar al contenedor y consultar:

```powershell
docker exec -it sarc-postgres psql -U sarc_admin -d sarc_drone -c "select count(*) from sarc_drone.events;"
```

Criterio:
- La tabla `sarc_drone.events` aumenta filas.
- Si se publica `pose`, tambien aumenta `sarc_drone.pose_events`.

## 8) Verificar comandos remotos

Enviar telemetria solicitada:

```powershell
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 -H 'Content-Type: application/json' -d '{"id":"cmd-1","type":"REQUEST_TELEMETRY"}'
```

Activar follow:

```powershell
curl -X POST "http://192.168.1.134:8000/follow/sarc_drone_001?enabled=true"
```

Abortar mision:

```powershell
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

## Bloqueo actual en esta maquina

No se puede completar la parte MQTT aqui porque `docker` no esta instalado y Mosquitto aun no existe en la VM.
