# Backend de ingestion MQTT + PostgreSQL

Este backend pertenece a la pila recomendada para la VM Debian de VMware.

Responsabilidades:

- Suscribirse a los topics MQTT del dron.
- Persistir eventos en PostgreSQL.
- Publicar comandos hacia el dron.
- Exponer una API HTTP para la consola remota.
- Trabajar solo con datos estructurados, no con imagenes.

## Filosofia de datos

- La app publica metadatos de inferencia y control.
- No se guardan frames ni capturas en la VM por ahora.
- Toda la persistencia historica se hace en PostgreSQL.

Base de datos:

- DB: `sarc_drone`
- Schema: `sarc_drone`

Topics principales:

- `sarc/drone/telemetry`
- `sarc/drone/detections`
- `sarc/drone/tracking`
- `sarc/drone/pose`
- `sarc/drone/ack`
- `sarc/commands/drone/{drone_id}`

### Contratos de payload

Ejemplo `telemetry`:

```json
{
	"drone_id": "sarc_drone_001",
	"timestamp": 1730000000000,
	"lat": 0.0,
	"lon": 0.0,
	"alt": 20.0,
	"search_pattern": "lawnmower",
	"fps": 12.5,
	"source": "android_vision_app"
}
```

Ejemplo `detections`:

```json
{
	"drone_id": "sarc_drone_001",
	"timestamp": 1730000000000,
	"class": "person",
	"confidence": 0.92,
	"bbox": {"x": 120.0, "y": 80.0, "w": 42.0, "h": 110.0},
	"drone_position": {"lat": 0.0, "lon": 0.0, "alt": 20.0},
	"fps": 11.8,
	"source": "android_vision_app"
}
```

Ejemplo `tracking`:

```json
{
	"drone_id": "sarc_drone_001",
	"timestamp": 1730000000000,
	"class": "person",
	"confidence": 0.91,
	"target": {"x": 330.0, "y": 220.0},
	"bbox": {"x": 320.0, "y": 200.0, "w": 55.0, "h": 130.0},
	"drone_position": {"lat": 0.0, "lon": 0.0, "alt": 20.0},
	"fps": 11.7,
	"tracking_enabled": true,
	"command": "TRACK:330.0:220.0",
	"source": "android_vision_app"
}
```

Ejemplo `pose`:

```json
{
	"drone_id": "sarc_drone_001",
	"timestamp": 1730000000000,
	"class": "person",
	"confidence": 0.87,
	"pose": {"keypoints": []},
	"source": "android_vision_app"
}
```

Endpoints:

- `GET /health`
- `GET /events/{drone_id}`
- `POST /command/{drone_id}`
- `POST /follow/{drone_id}`
- `GET /console`

Uso operativo de la consola:

1. Abre `http://IP_DE_LA_VM:8000/console`.
2. Escribe el `drone_id`.
3. Usa `Follow ON`, `Telemetry` o `Abort`.
4. Verifica el ACK y el registro en PostgreSQL.


## Esquema de tablas

### `events`

- `id`: clave primaria.
- `drone_id`: identificador del dron.
- `event_type`: telemetry/detections/tracking/pose/ack.
- `topic`: topic MQTT origen.
- `payload`: JSONB completo del evento.
- `source_timestamp`: timestamp original del dron.
- `received_at`: timestamp de ingesta.

### `commands`

- `id`: clave primaria.
- `command_id`: id único del comando.
- `drone_id`: dron objetivo.
- `command_type`: tipo lógico del comando.
- `topic`: topic MQTT publicado.
- `payload`: JSONB completo del comando.
- `status`: SENT / ACK / ABORTED / etc.
- `requested_by`: origen de la petición.

### `pose_events`

- `id`: clave primaria.
- `drone_id`: dron origen.
- `class_name`: clase detectada.
- `confidence`: confianza de pose/clasificación.
- `topic`: topic MQTT origen.
- `payload`: JSONB completo.
- `source_timestamp`: timestamp original.
- `received_at`: timestamp de ingesta.

Ejemplo de arranque local dentro de la VM:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Consola web

La consola minimalista vive en `/console` y permite:

- Ver eventos recientes por `drone_id`.
- Filtrar por `telemetry`, `detections`, `tracking`, `pose` o `ack`.
- Guardar el `drone_id` en el navegador.
- Exportar los eventos a JSON.
- Enviar `FOLLOW_TARGET`, `ABORT_MISSION` y `REQUEST_TELEMETRY`.

Uso típico:

1. Abrir `http://IP_DE_LA_VM:8000/console`.
2. Escribir el `drone_id`.
3. Pulsar `Follow ON` o cualquier otro comando.
4. Confirmar que aparece el `ACK` en MQTT y en PostgreSQL.
