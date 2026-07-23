# SARC-Drone en VMware Debian

Esta carpeta contiene la variante recomendada para desplegar el backend del proyecto en una VM Debian dentro de VMware con IP privada independiente.

Si quieres una guía corta para levantarlo y probarlo en pocos minutos, abre [QUICKSTART.md](QUICKSTART.md).

Arquitectura objetivo:

- Android app en el dron: publica telemetria, inferencia, seguimiento y ACK por MQTT.
- Mosquitto en la VM Debian: broker MQTT local de la solucion.
- PostgreSQL en la misma VM Debian: persistencia historica.
- Backend FastAPI en la misma VM Debian: suscribe MQTT, escribe en PostgreSQL y expone API para enviar comandos.

Flujo extremo a extremo:

```mermaid
flowchart LR
  A[Android app en el dron] -->|telemetria, detecciones, tracking, pose, ack| B[Mosquitto]
  B --> C[Backend FastAPI]
  C --> D[PostgreSQL sarc_drone.sarc_drone]
  E[Consola remota web] -->|FOLLOW_TARGET, ABORT_MISSION, REQUEST_TELEMETRY| C
  C -->|publica comandos| B
  B -->|comandos MQTT| A
```

Base de datos requerida:

- Nombre de la base: `sarc_drone`
- Nombre del esquema: `sarc_drone`

Filosofia de intercambio de datos:

- No se envian imagenes al VM.
- Se publican solo datos estructurados: clases detectadas, confianza, cajas, pose, tracking, telemetria y ACK.
- El VM puede ordenar al dron entrar en modo seguimiento mediante MQTT/API.

## Estructura de carpetas

```text
04_docs/vmware_debian/
├── .env.example
├── README.md
├── QUICKSTART.md
├── CHECKLIST_E2E.md
├── e2e_suite_vmware.sh
├── backend/
│   ├── app.py
│   └── requirements.txt
├── mosquitto/
│   ├── aclfile
│   └── mosquitto.conf
└── postgres/
    └── init/
        └── 001_init.sql
```

## 1) Prerrequisitos en Debian

Instalar Python y las herramientas base en la VM Debian.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Mosquitto todavia no esta instalado en la VM. Cuando quieras activar MQTT en la misma maquina:

```bash
sudo apt install -y mosquitto mosquitto-clients
```

## 1.1) Visión general del flujo

```mermaid
flowchart LR
  A[Android app en el dron] -->|MQTT: telemetry/detections/tracking/pose/ack| B[Broker Mosquitto]
  B --> C[Backend FastAPI]
  C --> D[PostgreSQL sarc_drone.sarc_drone]
  E[Consola remota web / API] -->|POST /follow /command| C
  C -->|MQTT: comandos| B
  B -->|FOLLOW_TARGET / ABORT_MISSION / REQUEST_TELEMETRY| A
```

Puntos clave:

- La app no envía imágenes.
- El backend persiste solo datos estructurados.
- La consola remota controla el dron a través del backend.
- Mosquitto y PostgreSQL pueden convivir en la misma VM Debian.

## 2) IP privada de la VM

Asigna una IP privada estable a la VM, por ejemplo `192.168.1.134` o la que use tu red VMware.

Los puertos que se exponen son:

- `1883` MQTT
- `5432` PostgreSQL
- `8000` API backend

## 3) Variables de entorno

Copia el archivo de ejemplo:

```bash
cp .env.example .env
```

Edita los secretos antes de levantar el stack.

## 4) Configurar Mosquitto

La configuracion base esta en `mosquitto/mosquitto.conf`.

Esto quedara pendiente hasta que instales Mosquitto en la VM.

Debes crear el archivo de contraseñas del broker en la VM cuando llegue ese paso.

Ejemplo para crear usuarios:

```bash
mosquitto_passwd -c /etc/mosquitto/passwords drone_client
```

Luego repite para el backend:

```bash
mosquitto_passwd /etc/mosquitto/passwords backend_service
```

ACL esperado:

- `drone_client` puede escribir telemetria, detecciones, tracking, pose y ack.
- `backend_service` puede leer eventos y escribir comandos.
- No dejar `allow_anonymous true` en producción.

## 5) Levantar el backend con Python directo

```bash
cd /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/backend
source /opt/sarc_drone_backend/venv/bin/activate
export $(grep -v '^#' /opt/sarc_drone_backend/.env | xargs)
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Servicios esperados ahora:

- PostgreSQL ya disponible en `192.168.1.134`
- Backend FastAPI corriendo en `8000`

Si quieres reiniciar el backend desde cero:

```bash
pkill -f "uvicorn app:app" || true
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 6) Verificacion rapida

Health del backend:

```bash
curl http://192.168.1.134:8000/health
```

Consola remota web:

```bash
http://192.168.1.134:8000/console
```

Desde esa pagina puedes:

- Ver eventos recientes del dron.
- Guardar el `drone_id` en el navegador.
- Filtrar eventos por `telemetry`, `detections`, `tracking`, `pose` o `ack`.
- Exportar los eventos en JSON.
- Ver vistas rapidas de `tracking` y `pose`.
- Enviar `FOLLOW_TARGET`.
- Enviar `ABORT_MISSION`.
- Enviar `REQUEST_TELEMETRY`.

Probar MQTT cuando Mosquitto ya exista:

```bash
mosquitto_sub -h 192.168.1.134 -p 1883 -u drone_client -P TU_PASSWORD -t 'sarc/drone/#' -v
```

Probar publicación de evento ficticio:

```bash
mosquitto_pub -h 192.168.1.134 -p 1883 -u drone_client -P TU_PASSWORD \
  -t sarc/drone/telemetry -m '{"drone_id":"sarc_drone_001","timestamp":1730000000000,"lat":0.0,"lon":0.0,"alt":20.0,"search_pattern":"lawnmower","fps":12.5}'
```

Enviar comando de prueba:

```bash
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 \
  -H 'Content-Type: application/json' \
  -d '{"id":"cmd-1","type":"REQUEST_TELEMETRY"}'
```

Probar seguimiento remoto:

```bash
curl -X POST "http://192.168.1.134:8000/follow/sarc_drone_001?enabled=true"
```

Probar aborto de mision:

```bash
curl -X POST http://192.168.1.134:8000/command/sarc_drone_001 \
  -H 'Content-Type: application/json' \
  -d '{"id":"cmd-2","type":"ABORT_MISSION"}'
```

Suite E2E en un solo comando (smoke + ACK + telemetria opcional):

```bash
chmod +x /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/e2e_suite_vmware.sh
DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/e2e_suite_vmware.sh
```

Para exigir tambien telemetria en vivo desde Android:

```bash
LIVE_TELEMETRY_CHECK=1 DRONE_MQTT_PASSWORD=TU_PASSWORD_DRONE /opt/sarc_drone_backend/sarc_drone/04_docs/vmware_debian/e2e_suite_vmware.sh
```

## 7) Topics MQTT usados

- `sarc/drone/telemetry`
- `sarc/drone/detections`
- `sarc/drone/tracking`
- `sarc/drone/pose`
- `sarc/drone/ack`
- `sarc/commands/drone/{drone_id}`

Eventos que publica la app:

- `telemetry`: posicion, altitud, FPS y patron de busqueda.
- `detections`: clase, confianza, bounding box y posicion del dron.
- `tracking`: objetivo, caja, centro del blanco y estado de seguimiento.
- `pose`: pose o keypoints si se habilitan en una etapa posterior.
- `ack`: confirmacion de comandos recibidos o ejecutados.

Comandos que publica el backend:

- `FOLLOW_TARGET`
- `ABORT_MISSION`
- `REQUEST_TELEMETRY`
- `MANUAL_CONTROL`
- `SET_SEARCH_PATTERN`
- `CHANGE_TARGET`

Semántica recomendada:

- `telemetry`: lat/lon/alt, modo de búsqueda, FPS y estado general.
- `detections`: clase, confianza, bbox y posición del dron.
- `tracking`: objetivo, bbox, estado de seguimiento y comando sugerido.
- `pose`: datos de pose/landmarks si luego se añade el modelo de pose.
- `ack`: confirmación de ejecución de comandos.

Comando de seguimiento remoto:

- `POST /follow/{drone_id}`
- Publica un comando `FOLLOW_TARGET` al topic del dron.

Consola web remota:

- `GET /console`
- Lee eventos desde `GET /events/{drone_id}`.
- Envía comandos a `POST /command/{drone_id}` y `POST /follow/{drone_id}`.

## 8) Tablas PostgreSQL

Se crean bajo el esquema `sarc_drone`:

- `events`
- `commands`
- `pose_events`

Uso de tablas:

- `events`: histórico general de todo lo que llega por MQTT.
- `pose_events`: vista optimizada para eventos de pose.
- `commands`: auditoria de comandos enviados y su estado de ACK.

Ejemplo de esquema lógico:

```text
sarc_drone.events
  - drone_id
  - event_type
  - topic
  - payload
  - source_timestamp
  - received_at

sarc_drone.commands
  - command_id
  - drone_id
  - command_type
  - topic
  - payload
  - status
  - requested_by
  - created_at
  - updated_at

sarc_drone.pose_events
  - drone_id
  - class_name
  - confidence
  - topic
  - payload
  - source_timestamp
  - received_at
```

Esquema recomendado:

- Base de datos: `sarc_drone`
- Esquema: `sarc_drone`

Campos principales:

- `events.payload` guarda JSONB con el contenido completo del evento.
- `commands.payload` guarda el comando enviado y su estado.
- `pose_events.payload` guarda los datos de pose sin imágenes.

## 9) Integracion con la app Android

La app debe apuntar al broker de la VM Debian:

- host: `192.168.1.134` o la IP real de tu VM
- puerto: `1883`
- usuario: `drone_client`

La consola remota debe usar la API del backend en vez de tocar PostgreSQL directamente.

La app no debe enviar imagenes al VM; solo metadatos de inferencia, pose, seguimiento y estado.

Campos recomendados que debe incluir la app en los JSON:

- `drone_id`
- `timestamp`
- `class` o `class_name`
- `confidence`
- `bbox`
- `target`
- `drone_position`
- `fps`
- `tracking_enabled`
- `source`
- `command` cuando aplique

Para que la app funcione con la VM:

- Cambiar el broker MQTT al IP privado de la VM.
- Mantener el mismo `drone_id` en la app y en la consola.
- Verificar que el topic de comandos sea `sarc/commands/drone/{drone_id}`.

## 10) Orden de despliegue recomendado

1. Confirmar PostgreSQL ya accesible en `192.168.1.134`.
2. Crear el entorno Python del backend.
3. Ejecutar `uvicorn app:app --host 0.0.0.0 --port 8000`.
4. Verificar `GET /health`.
5. Instalar Mosquitto en la VM cuando pases a la parte MQTT.
6. Ajustar `MQTT_HOST=192.168.1.134` en `.env`.
7. Probar telemetria, detecciones, tracking y comandos.
8. Abrir `http://192.168.1.134:8000/console` y validar eventos y comandos.

## 11) Referencias rapidas

Para el flujo paso a paso y las pruebas exactas, usa [QUICKSTART.md](QUICKSTART.md).

Para la descripcion completa de la pila, payloads, tablas y troubleshooting, este README es la referencia principal.

Scripts operativos disponibles:

- `smoke_test_vmware.sh`: valida health, MQTT y persistencia de comandos.
- `ack_test_vmware.sh`: fuerza un ACK de prueba y confirma la transicion `SENT -> ACK`.
- `redeploy_backend_vmware.sh`: resincroniza `04_docs/vmware_debian` hacia `/opt` y reinicia `sarc-backend`.

## 12) Troubleshooting rapido

### La consola web no carga

- Verifica que `uvicorn` siga ejecutandose en la terminal.
- Revisa la salida del proceso del backend.
- Comprueba `http://192.168.1.134:8000/health`.

### MQTT no conecta

- Revisa usuario, contraseña y ACL de Mosquitto.
- Comprueba que el broker ya esté instalado en la VM.
- Comprueba que la app use la IP privada correcta de la VM.

### No se guardan eventos en PostgreSQL

- Verifica que `PGDATABASE=sarc_drone` y `PGSCHEMA=sarc_drone`.
- Revisa la salida del backend o `journalctl` si lo ejecutas como servicio.
- Comprueba que llegue `drone_id` en el payload.

### No responde FOLLOW_TARGET

- Verifica que la app esté suscrita a `sarc/commands/drone/{drone_id}`.
- Revisa el `ACK` en `sarc/drone/ack`.
- Asegúrate de que `trackingEnabled` se active desde el comando.
