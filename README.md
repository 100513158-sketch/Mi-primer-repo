# SARC-Drone: Sistema Autonomo de Rescate con Vision por Computadora

Guia completa de implementacion con configuracion centralizada en config.yaml.
Estado del documento: actualizado al 2026-06-27.

## Indice

- Fase 0: Estructura de directorios y configuracion
- Fase 1: Preparacion de datasets
- Fase 2: Entrenamiento curricular con pipeline secuencial (NUEVO)
- Fase 3: Optimizacion y exportacion
- Fase 4: Aplicacion Android
- Fase 5: Integracion IoT, nube y control remoto
- Fase 6: Autonomia avanzada y produccion
- Anexos

## Fase 0: Estructura y configuracion

> Documentación complementaria general: [04_docs/descripcion_general_proyecto.md](04_docs/descripcion_general_proyecto.md)

### 0.1 Estructura final

```text
C:\SARC-Drone\
|
|-- requirements.txt
|-- config.yaml                          <- fuente unica de verdad
|
|-- 00_datasets/
|   |-- raw/
|   |   |-- C2A/         {train,val,test}/{images,labels}/
|   |   |-- NITC/        {train,val,test}/{images,labels}/
|   |   |-- NOMAD/       {train,val,test}/{images,labels}/
|   |   |-- OKUTAMA/     {train,val,test}/{images,labels}/
|   |   |-- RESDataset/  {train,val,test}/{images,labels}/
|   |   |-- SeaDronesSee/{train,val,test}/{images,labels}/
|   |   |-- VisDrone/    {train,val,test}/{images,labels}/
|   |   `-- prepare_rescue_datasets.py
|   |-- processed/
|   |   |-- C2A/         {train,val,test}/{images,labels}/  data.yaml
|   |   |-- NITC/        ...
|   |   |-- NOMAD/       ...
|   |   |-- OKUTAMA/     ...
|   |   |-- RESDataset/  ...
|   |   |-- SeaDronesSee/...
|   |   `-- VisDrone/    ...
|   `-- data.yaml                        <- data.yaml global (referencia / uso manual)
|
|-- 01_training/
|   |-- scripts/
|   |   |-- config_utils.py              <- carga config.yaml
|   |   |-- integrity_check.py           <- NUEVO: valida datasets antes de entrenar
|   |   |-- generate_dataset_yamls.py    <- NUEVO: genera data.yaml por dataset
|   |   |-- train_pipeline.py            <- NUEVO: pipeline curricular completo
|   |   |-- train.py                     <- entrenamiento unico (uso manual/pose)
|   |   |-- export_model.py
|   |   |-- extreme_optimization.py
|   |   |-- convert_visdrone.py
|   |   |-- convert_okutama.py
|   |   |-- merge_datasets.py            <- conservado, no se usa en pipeline
|   |   `-- download_visdrone.ps1
|   `-- runs/
|       `-- detection/
|           |-- VisDrone/   weights/ results.csv
|           |-- NOMAD/      ...
|           |-- OKUTAMA/    ...
|           |-- SeaDronesSee/...
|           |-- NITC/       ...
|           |-- RESDataset/ ...
|           `-- C2A/        ...
|
|-- 02_models/
|   |-- exported/{tflite,onnx,openvino}/
|   `-- weights/
|       |-- best_VisDrone.pt             <- checkpoint por etapa
|       |-- best_NOMAD.pt
|       |-- best_OKUTAMA.pt
|       |-- best_SeaDronesSee.pt
|       |-- best_NITC.pt
|       |-- best_RESDataset.pt
|       |-- best_C2A.pt                  <- modelo final para exportar
|       `-- results_<DATASET>.csv        <- metricas por etapa
|
|-- 03_android_app/
|   `-- SARCApp/
|
|-- 04_docs/
`-- 05_tests/
```

### 0.2 requirements.txt en raiz

Archivo: C:\SARC-Drone\requirements.txt

Instalacion:

```powershell
cd C:\SARC-Drone
python -m venv venv_sarc
venv_sarc\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 0.3 Configuracion central

Archivo: C:\SARC-Drone\config.yaml

Este archivo centraliza:

- Rutas del proyecto
- URLs de datasets
- Mapeo de clases por dataset
- Hiperparametros de entrenamiento global y por etapa
- Orden del pipeline curricular (`training_pipeline.order`)
- Parametros de exportacion, Android y dron

Todos los scripts Python en 01_training/scripts leen este archivo.

### 0.4 Verificacion base

```powershell
cd C:\SARC-Drone
venv_sarc\Scripts\activate
python -c "import torch; from ultralytics import YOLO; print('CUDA:', torch.cuda.is_available()); print('Ultralytics OK')"
```

## Fase 1: Preparacion de datasets

### 1.1 Datasets disponibles

Todos los datasets estan en `00_datasets/raw/` con estructura ya normalizada
`{train,val,test}/{images,labels}/`.

| Dataset | Naturaleza | Clases | Especificidad SARC |
|---|---|---|---|
| VisDrone | Vigilancia aerea urbana, multiobjeto | person | Generica |
| NOMAD | Seguimiento aereo de personas en exterior | person | Baja |
| OKUTAMA | Reconocimiento de accion aerea | person | Baja |
| SeaDronesSee | SAR maritimo: nadadores, embarcaciones | swimmer, boat, jetski, life_saving_appliances, buoy | Media |
| NITC | Rescate de personas, escenas variadas | person | Alta |
| RESDataset | Escenas de emergencia y rescate | person | Alta |
| C2A | Especifico SARC, rescate en area | person | Maxima |

### 1.2 Verificacion de integridad

Antes de entrenar, verificar que todos los datasets estan correctamente
formateados para YOLO26:

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\integrity_check.py
```

Opciones:

```powershell
python .\integrity_check.py --dataset VisDrone   # un solo dataset
python .\integrity_check.py --strict             # falla si hay imagenes sin label
```

El script valida por cada dataset:

- Existencia de `train/images/`, `train/labels/`, `val/images/`, `val/labels/`
- Correspondencia imagen <-> label (archivo .txt vacio = imagen sin objetos, valido)
- Formato YOLO por linea: `<int> <float> <float> <float> <float>` con valores en `[0, 1]`
- Ausencia de archivos de imagen corruptos (0 bytes)

Salida esperada:

```
=== Verificacion de integridad YOLO ===
[OK]  VisDrone         train:12345  val:1870  test:548   clases:[0]
[OK]  NOMAD            train:8200   val:910   test:320   clases:[0]
[OK]  OKUTAMA          train:6100   val:700             clases:[0]
[OK]  SeaDronesSee     train:4100   val:620   test:80    clases:[0, 1, 2, 3, 4]
[OK]  NITC             train:2800   val:350   test:150   clases:[0]
[OK]  RESDataset       train:1900   val:240   test:110   clases:[0]
[OK]  C2A              train:1200   val:150   test:80    clases:[0]

Integridad OK - 7/7 datasets validos. Listo para entrenar.
```

### 1.3 Generar data.yaml individual por dataset

Cada dataset necesita su propio `data.yaml` en `processed/<DATASET>/`.
El pipeline lo genera automaticamente, pero tambien puede ejecutarse a mano:

```powershell
python .\generate_dataset_yamls.py              # genera los que faltan
python .\generate_dataset_yamls.py --overwrite  # regenera todos
python .\generate_dataset_yamls.py --dataset SeaDronesSee
```

Ejemplo de archivo generado para `processed/VisDrone/data.yaml`:

```yaml
path: C:/SARC-Drone/00_datasets/processed/VisDrone
train: train/images
val:   val/images
test:  test/images
nc: 1
names:
  0: person
```

Para SeaDronesSee el script detecta automaticamente las 5 clases:

```yaml
path: C:/SARC-Drone/00_datasets/processed/SeaDronesSee
train: train/images
val:   val/images
test:  test/images
nc: 5
names:
  0: swimmer
  1: boat
  2: jetski
  3: life_saving_appliances
  4: buoy
```

### 1.4 data.yaml global (referencia)

El archivo `00_datasets/data.yaml` queda como referencia para usos manuales
o pruebas puntuales con `train.py`. El pipeline curricular usa los
`data.yaml` individuales generados en el paso anterior.

## Fase 2: Entrenamiento curricular con pipeline secuencial

### 2.1 Estrategia

En lugar de mezclar todos los datasets en un unico entrenamiento, el pipeline
entrena cada dataset de forma **independiente y secuencial**, de menor a mayor
especificidad SARC. Los pesos de cada etapa se transfieren a la siguiente
(curriculum learning con transferencia progresiva).

```
VisDrone -> NOMAD -> OKUTAMA -> SeaDronesSee -> NITC -> RESDataset -> C2A
(generica)                     (SAR maritimo)          (rescate)    (SARC)
```

A medida que el modelo se acerca al dominio SARC:
- La tasa de aprendizaje disminuye
- Aumenta el numero de capas congeladas (freeze)
- Los epochs se reducen (el modelo ya tiene buena base)

### 2.2 Parametros por etapa (config.yaml > training_pipeline)

| Dataset | Epochs | Batch | Freeze | lr0 |
|---|---|---|---|---|
| VisDrone | 300 | 32 | 0 | 0.001 |
| NOMAD | 200 | 32 | 0 | 0.001 |
| OKUTAMA | 200 | 32 | 0 | 0.001 |
| SeaDronesSee | 150 | 32 | 5 | 0.0005 |
| NITC | 100 | 16 | 10 | 0.0001 |
| RESDataset | 100 | 16 | 10 | 0.0001 |
| C2A | 100 | 16 | 10 | 0.00005 |

Todos estos valores son editables en `config.yaml` bajo `training_pipeline.per_dataset`
sin tocar ningun script.

### 2.3 Ejecutar el pipeline completo

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\train_pipeline.py
```

El pipeline ejecuta automaticamente en este orden:

1. Verificacion de integridad de todos los datasets del pipeline
2. Generacion de `data.yaml` individuales si faltan
3. Deteccion de GPU y ajuste de batch efectivo
4. Entrenamiento secuencial etapa por etapa con transferencia de pesos
5. Si una etapa falla, continua con el ultimo checkpoint valido disponible

### 2.4 Opciones del pipeline

```powershell
# Reanudar desde una etapa especifica (usa el checkpoint previo existente)
python .\train_pipeline.py --start-from NITC

# Ejecutar solo una etapa concreta
python .\train_pipeline.py --only SeaDronesSee

# Simular el pipeline completo sin entrenar (verificacion de configuracion)
python .\train_pipeline.py --dry-run
```

### 2.5 Salidas por etapa

Cada etapa produce tres artefactos:

```
01_training/runs/detection/<DATASET>/weights/best.pt  <- run de Ultralytics
02_models/weights/best_<DATASET>.pt                   <- copia del checkpoint
02_models/weights/results_<DATASET>.csv               <- curvas de loss y mAP
```

Salida en consola por etapa:

```
========================================================
Dataset : VisDrone
Pesos   : yolo26m.pt
Epochs  : 300  |  Batch: 32  |  Freeze: 0 capas
lr0     : 0.001  |  lrf: 0.01
========================================================
... (logs Ultralytics) ...
Metricas finales - mAP50: 0.682  mAP50-95: 0.421
Pesos guardados: best_VisDrone.pt

[1/7] OK - VisDrone completado en 142.3 min | Pesos: best_VisDrone.pt
```

### 2.6 Modelo final

El modelo resultante del pipeline completo es `best_C2A.pt`, que ha recorrido
todo el curriculum. Es el candidato para exportar a Android.

### 2.7 Entrenamiento manual de un solo dataset (uso avanzado)

`train.py` queda disponible para entrenar manualmente sobre el `data.yaml` global
o para el entrenamiento de pose cuando haya datasets con keypoints:

```powershell
# Entrenamiento unico sobre data.yaml global
python .\train.py

# Entrenamiento de pose (pendiente de dataset con keypoints)
python .\train.py  # adaptar data.yaml con kpt_shape antes de ejecutar
```

Salidas de train.py (uso manual):

- `02_models/weights/best_detection.pt`
- `02_models/weights/best_pose.pt`

## Fase 3: Exportacion

Script: C:\SARC-Drone\01_training\scripts\export_model.py

Script adicional: C:\SARC-Drone\01_training\scripts\extreme_optimization.py

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\export_model.py
python .\extreme_optimization.py
```

Salidas:

- C:\SARC-Drone\02_models\exported\tflite\
- C:\SARC-Drone\02_models\exported\onnx\
- C:\SARC-Drone\02_models\exported\openvino\

## Fase 4: Android

- Proyecto: C:\SARC-Drone\03_android_app\SARCApp
- Copiar modelo TFLite exportado a app/src/main/assets/

Build:

```powershell
cd C:\SARC-Drone\03_android_app\SARCApp
.\gradlew assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

## Fase 5: Integracion IoT, nube y control remoto

La Fase 5 agrega comunicacion bidireccional con centro de mando, recepcion de comandos remotos y backend cloud para almacenamiento de eventos.

### 5.1 Configuracion central

El archivo config.yaml incluye estas secciones nuevas:

- iot.mqtt (broker, credenciales, topics)
- iot.database (postgresql para la pila VMware Debian)
- autonomy (simulacion, bateria, patrones)

### 5.2 Android: modulos IoT

Archivos agregados en 03_android_app/SARCApp/app/src/main/java/com/sarc/drone/vision:

- MqttManager.kt
- CommandProcessor.kt
- DroneSimulator.kt

Integracion aplicada en VisionActivity.kt:

- Conexion MQTT al iniciar actividad
- Suscripcion a topic de comandos remotos
- Procesamiento de comandos (cambio objetivo, abortar, control manual, telemetria)
- Publicacion de telemetria y detecciones al broker

### 5.3 VMware Debian: backend, broker y persistencia

Archivos VMware Debian:

- 04_docs/vmware_debian/README.md
- 04_docs/vmware_debian/QUICKSTART.md

Flujo Node-RED de ejemplo:


Ejecucion de la pila VMware Debian:

```powershell
cd 04_docs\vmware_debian
docker compose up -d --build
```

## Fase 6: Autonomia avanzada y produccion

La Fase 6 agrega planificacion de busqueda, evitacion de obstaculos, gestion de bateria y pruebas sin dron real.

### 6.1 Android: modulos de autonomia

Archivos agregados:

- SearchPattern.kt
- ObstacleAvoidance.kt
- BatteryManager.kt

Soporte adicional:

- Config.kt ampliado con parametros MQTT/autonomia
- MavlinkController.kt con hooks para setVelocityTarget y requestTelemetry

### 6.2 Simulacion sin hardware real

La simulacion se habilita por Config.SIMULATION_ENABLED (integrada en VisionActivity + DroneSimulator).

Prueba rapida:

```powershell
cd C:\SARC-Drone\03_android_app\SARCApp
.\gradlew installDebug
adb shell am start -n com.sarc.drone.vision/.VisionActivity
```

### 6.3 Objetivo operativo

Con Fase 5/6 integradas, el sistema soporta:

- Vision + tracking local en Android
- Telemetria y detecciones en tiempo real hacia la nube
- Comandos remotos desde centro de mando
- Ejecucion en modo simulado para desarrollo continuo

## Anexos

### Guia operativa recomendada

Para ejecucion paso a paso y validaciones previas, usar:

- 04_docs/flujo_comandos.md

### Flujo de comandos completo (primera fase)

```powershell
# 0. Entorno
cd C:\SARC-Drone
venv_sarc\Scripts\activate

# 1. Integridad (opcional, el pipeline la ejecuta automaticamente)
python 01_training\scripts\integrity_check.py

# 2. data.yaml por dataset (opcional, el pipeline los genera si faltan)
python 01_training\scripts\generate_dataset_yamls.py

# 3. Pipeline de entrenamiento curricular
python 01_training\scripts\train_pipeline.py

# 4. Exportar mejor modelo
python 01_training\scripts\export_model.py

# 5. Build Android
cd 03_android_app\SARCApp
.\gradlew assembleRelease
adb install -r app\build\outputs\apk\release\app-release.apk
```

### Chequeo rapido de artefactos tras entrenamiento

```powershell
cd C:\SARC-Drone
@('VisDrone','NOMAD','OKUTAMA','SeaDronesSee','NITC','RESDataset','C2A') | ForEach-Object {
    $p = "02_models\weights\best_$_.pt"
    if (Test-Path $p) { "OK  $p" } else { "FALTA  $p" }
}
```

### Resumen de arquitectura y cambios respecto a version anterior

| Elemento | Antes | Ahora |
|---|---|---|
| Datasets usados | 3 (VisDrone, OKUTAMA, NITC) | 7 (todos los de raw/) |
| Estrategia de datos | Merge en un unico conjunto | Independiente por dataset |
| data.yaml | 1 global | 1 por dataset + 1 global de referencia |
| Entrenamiento | 1 run unico | 7 runs secuenciales con transfer |
| Pesos producidos | best_detection.pt | best_<DATASET>.pt x7 |
| Integridad | No existia | Obligatoria antes de cualquier entrenamiento |
| Scripts nuevos | - | integrity_check.py, generate_dataset_yamls.py, train_pipeline.py |
| config.yaml | Sin pipeline | Seccion training_pipeline, go_nogo e iot alineada con VMware Debian |

Componentes del sistema:

- requirements.txt en raiz
- config.yaml como unica fuente de verdad
- Scripts Python adaptados a config.yaml
- Pipeline curricular de 7 etapas
- Android con IoT/MQTT + autonomia
- Backend VMware Debian con FastAPI + MQTT + PostgreSQL
