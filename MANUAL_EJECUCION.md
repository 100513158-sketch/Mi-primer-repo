# Manual de Ejecución — SARC-Drone Pipeline Curricular

**Estado del documento:** Actualizado 2026-06-28  
**Versión:** 1.0 — Pipeline de 7 etapas con entrenamiento curricular  
**Responsable:** Sistema SARC-Drone (Visión + Rescate)

---

## Índice rápido

1. [Prerequisitos](#prerequisitos)
2. [Antes de empezar](#antes-de-empezar)
3. [Ejecución paso a paso](#ejecución-paso-a-paso)
4. [Referencia de comandos](#referencia-de-comandos)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisitos

### Hardware mínimo

- **CPU:** Intel i7 / AMD Ryzen 7 o superior
- **RAM:** 16 GB (32 GB recomendado)
- **GPU:** NVIDIA (CUDA 11.8+) — altamente recomendado
  - ~6 GB VRAM mínimo por etapa
  - ~8-10 GB VRAM recomendado
- **SSD:** 50 GB libres (datasets + modelos + runs)

### Software requerido

```powershell
# Verificar versión de Windows
[System.Environment]::OSVersion.VersionString

# Verificar Python 3.10+
python --version

# Verificar NVIDIA CUDA (si tienes GPU NVIDIA)
nvidia-smi
```

**Resultado esperado:**
```
Windows 10/11
Python 3.10.x o superior
CUDA 11.8+ (si GPU disponible)
cuDNN 8.x (si GPU disponible)
```

---

## Antes de empezar

### 1. Clonar / descargar el proyecto

```powershell
# Si es un repositorio Git
git clone <url-repo> C:\SARC-Drone
cd C:\SARC-Drone

# O navega manualmente si ya lo tienes
cd C:\SARC-Drone
```

### 2. Crear entorno virtual

```powershell
python -m venv venv_sarc
.venv\Scripts\activate
```

**Resultado esperado:** El prompt muestra `(venv_sarc) C:\SARC-Drone>`

### 3. Instalar dependencias

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Tiempo estimado:** 10-15 minutos  
**Tamaño descargado:** ~3-4 GB

### 4. Verificación rápida

```powershell
python -c "import torch; from ultralytics import YOLO; from yaml import safe_load; print('✓ PyTorch OK'); print('✓ Ultralytics OK'); print('✓ PyYAML OK'); print('GPU disponible:', torch.cuda.is_available())"
```

**Resultado esperado:**
```
✓ PyTorch OK
✓ Ultralytics OK
✓ PyYAML OK
GPU disponible: True  (o False si sin GPU)
```

---

## Ejecución paso a paso

### PASO 0: Verificación de estructura

```powershell
cd C:\SARC-Drone
venv_sarc\Scripts\activate

# Verificar que existan los directorios clave
@('00_datasets\processed\VisDrone', '01_training\scripts', '02_models\weights', 'config.yaml') | ForEach-Object {
    if (Test-Path $_) { "✓ $_" } else { "✗ FALTA: $_" }
}
```

**Resultado esperado:** Todos con ✓

### PASO 1: Verificación de integridad de datasets

```powershell
cd 01_training\scripts
python .\integrity_check.py
```

**Tiempo esperado:** 2-5 minutos  
**Salida esperada:**

```
=== Verificación de integridad YOLO ===

[OK]  VisDrone         train:12345  val:1870  test:548   clases:[0]
[OK]  NOMAD            train:8200   val:910   test:320   clases:[0]
[OK]  OKUTAMA          train:6100   val:700             clases:[0]
[OK]  SeaDronesSee     train:4100   val:620   test:80    clases:[0, 1, 2, 3, 4]
[OK]  NITC             train:2800   val:350   test:150   clases:[0]
[OK]  RESDataset       train:1900   val:240   test:110   clases:[0]
[OK]  C2A              train:1200   val:150   test:80    clases:[0]

Integridad OK — 7/7 datasets válidos. Listo para entrenar.
```

**Si falla alguno:** El script indicará el dataset, línea exacta y tipo de error. Corrige antes de continuar.

### PASO 2: Generar data.yaml individuales

```powershell
python .\generate_dataset_yamls.py
```

**Tiempo esperado:** <1 minuto  
**Salida esperada:**

```
=== Generando data.yaml por dataset ===
Directorio: C:/SARC-Drone/00_datasets/processed

  [OK]    VisDrone/data.yaml          nc=1  (0:person)       [train:12345  val:1870  test:548]  [generado]
  [OK]    NOMAD/data.yaml             nc=1  (0:person)       [train:8200   val:910   test:320]  [generado]
  [OK]    OKUTAMA/data.yaml           nc=1  (0:person)       [train:6100   val:700]             [generado]
  [OK]    SeaDronesSee/data.yaml      nc=5  (0:swimmer, ...) [train:4100   val:620   test:80]   [generado]
  [OK]    NITC/data.yaml              nc=1  (0:person)       [train:2800   val:350   test:150]  [generado]
  [OK]    RESDataset/data.yaml        nc=1  (0:person)       [train:1900   val:240   test:110]  [generado]
  [OK]    C2A/data.yaml               nc=1  (0:person)       [train:1200   val:150   test:80]   [generado]
```

Verifica que todos muestren `[generado]` o `[actualizado]`.

### PASO 3: Ejecutar pipeline completo de entrenamiento

```powershell
python .\train_pipeline.py
```

**Tiempo estimado por etapa (con GPU NVIDIA):**
- VisDrone: 3-4 horas (300 epochs)
- NOMAD: 2-2.5 horas (200 epochs)
- OKUTAMA: 2-2.5 horas (200 epochs)
- SeaDronesSee: 1-1.5 horas (150 epochs)
- NITC: 40-50 minutos (100 epochs)
- RESDataset: 40-50 minutos (100 epochs)
- C2A: 40-50 minutos (100 epochs)

**Total estimado:** ~13-16 horas con GPU  
**Total estimado:** ~2-3 días con CPU

**Salida en consola (inicio):**

```
======================================================
SARC-Drone — Pipeline curricular de entrenamiento
Etapas (7): VisDrone -> NOMAD -> OKUTAMA -> SeaDronesSee -> NITC -> RESDataset -> C2A
Modelo base : yolo26n.pt
Dry-run     : False
======================================================

[1/7] Iniciando etapa: VisDrone
========================================================
Dataset : VisDrone
Pesos   : yolo26n.pt
Epochs  : 300  |  Batch: 32  |  Freeze: 0 capas
lr0     : 0.001  |  lrf: 0.01
========================================================
```

Luego verás logs de Ultralytics con progreso por epoch:

```
Epoch 1/300
      all      45.2      0.889      0.858      0.789      0.701
      ...
Epoch 300/300
      all      15.8      0.925      0.917      0.854      0.823

Métricas finales — mAP50: 0.854  mAP50-95: 0.623
Pesos guardados: best_VisDrone.pt

[1/7] OK — VisDrone completado en 192.3 min | Pesos: best_VisDrone.pt
```

**La pipeline continúa automáticamente con la siguiente etapa.**

**Si una etapa falla:** El pipeline lo registra, intenta usar el último checkpoint válido y continúa. Al final muestra un resumen.

**Salida final:**

```
========================================================
Pipeline finalizado en 815.2 min
Etapas OK   : 7 / 7
Mejor modelo: best_C2A.pt
========================================================
```

### PASO 4: Verificar artefactos de entrenamiento

```powershell
cd C:\SARC-Drone

# Listar pesos generados
Get-ChildItem -Path "02_models\weights\best_*.pt" | Select-Object Name
```

**Resultado esperado:**

```
Name
────
best_VisDrone.pt
best_NOMAD.pt
best_OKUTAMA.pt
best_SeaDronesSee.pt
best_NITC.pt
best_RESDataset.pt
best_C2A.pt
```

Todos deben existir después del pipeline completo.

### PASO 5: Exportar modelos

```powershell
cd 01_training\scripts
python .\export_model.py
```

**Tiempo estimado:** 10-20 minutos (genera TFLite, ONNX, OpenVINO)  
**Salida esperada:**

```
TFLite exportado a: 02_models/exported/tflite/best_C2A/best_C2A.tflite
ONNX exportado a: 02_models/exported/onnx/best_C2A.onnx
OpenVINO exportado a: 02_models/exported/openvino/best_C2A/
```

El modelo **best_C2A.tflite** (de transferencia completa) es el candidato para Android.

### PASO 6: Build Android (opcional para esta fase)

```powershell
cd C:\SARC-Drone\03_android_app\SARCApp

# Copiar modelo TFLite a assets
Copy-Item "..\..\02_models\exported\tflite\best_C2A\best_C2A.tflite" "app\src\main\assets\best_model.tflite"

# Build
.\gradlew assembleRelease

# Instalar en dispositivo/emulador
adb install -r app\build\outputs\apk\release\app-release.apk
```

---

## Referencia de comandos

### Ejecución completa (recomendado)

```powershell
cd C:\SARC-Drone
venv_sarc\Scripts\activate
cd 01_training\scripts

# Integridad → data.yaml → entrenamiento
python .\integrity_check.py
python .\generate_dataset_yamls.py
python .\train_pipeline.py
```

### Opciones avanzadas del pipeline

```powershell
# Reanudar desde una etapa específica (con checkpoint previo)
python .\train_pipeline.py --start-from NITC

# Ejecutar solo una etapa
python .\train_pipeline.py --only SeaDronesSee

# Simular sin entrenar (verificar configuración)
python .\train_pipeline.py --dry-run
```

### Verificación individual de datasets

```powershell
# Verificar solo un dataset
python .\integrity_check.py --dataset VisDrone

# Modo estricto (falla si hay imágenes sin label)
python .\integrity_check.py --strict
```

### Generar/regenerar data.yaml

```powershell
# Generar solo los faltantes
python .\generate_dataset_yamls.py

# Regenerar todos
python .\generate_dataset_yamls.py --overwrite

# Un dataset específico
python .\generate_dataset_yamls.py --dataset SeaDronesSee
```

### Entrenamiento manual (uso avanzado)

```powershell
# Entrenar sobre data.yaml global
python .\train.py

# Entrenar pose (cuando haya keypoints disponibles)
python .\train.py  # (adaptar data.yaml primero)
```

### Exportación completa

```powershell
python .\export_model.py
python .\extreme_optimization.py  # optimizaciones adicionales
```

### Evaluación Go/No-Go (obligatorio antes de despliegue)

```powershell
cd C:\SARC-Drone

# Gate de candidato de modelo (permite continuar aunque aún no exportes)
python .\05_tests\go_nogo_gate.py --profile model_candidate

# Gate de liberación edge/android (requiere exportaciones finales)
python .\05_tests\go_nogo_gate.py --profile release_edge_android

# Reporte completo en JSON
python .\05_tests\go_nogo_gate.py --profile release_edge_android --json
```

Estados posibles del gate:

- `GO`: cumple umbrales y artefactos requeridos.
- `NO-GO`: no cumple métricas críticas.
- `PENDING`: entrenamiento o exportación aún incompletos.

### Orquestador post-entrenamiento (recomendado)

Script único para ejecutar automáticamente:

1. Gate `model_candidate`
2. Exportación de modelos
3. Optimización extrema (opcional)
4. Gate `release_edge_android`
5. Build Android (opcional)

```powershell
cd C:\SARC-Drone

# Flujo completo
python .\05_tests\run_post_training_release.py

# Sin build Android
python .\05_tests\run_post_training_release.py --skip-build

# Sin optimización extrema
python .\05_tests\run_post_training_release.py --no-extreme
```

Reportes automáticos:

- Se guardan en `05_tests/reports/post_training_release_*.json`.
- Incluyen estado final, salidas por paso y motivo de fallo si aplica.

---

## Troubleshooting

### Error: `No module named 'torch'`

**Causa:** PyTorch no está instalado  
**Solución:**
```powershell
cd C:\SARC-Drone
venv_sarc\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: `CUDA out of memory`

**Causa:** GPU sin suficiente VRAM  
**Soluciones (en orden):**

1. Reducir batch_size en `config.yaml`:
   ```yaml
   training_pipeline:
     per_dataset:
       VisDrone:
         batch_size: 16  # de 32 a 16
   ```

2. Ejecutar solo una etapa a la vez:
   ```powershell
   python .\train_pipeline.py --only VisDrone
   ```

3. Usar CPU (muy lento):
   ```powershell
   python .\train_pipeline.py  # automáticamente detecta CPU
   ```

### Error: `data.yaml not found`

**Causa:** Los archivos data.yaml no fueron generados  
**Solución:**
```powershell
python .\generate_dataset_yamls.py
```

### Error: `[FAIL] Dataset <NAME> con errores de integridad`

**Causa:** Imagen/label corrupta o formato inválido  
**Solución:** El script muestra la línea exacta. Corrige o elimina el archivo problemático:

```powershell
# Revisar el error reportado, e.g.
# "00_datasets/processed/VisDrone/labels/img_001.txt:3: campo[1]=1.234567 fuera de [0,1]"

# Editar o eliminar ese archivo y reejecutar integridad
python .\integrity_check.py --dataset VisDrone
```

### Error: `[WARN] No GPU detectada`

**Causa:** Sin NVIDIA/CUDA o drivers no configurados  
**Impacto:** El entrenamiento será muy lento (CPU)  
**Solución:** Instalar CUDA 11.8+ y cuDNN 8.x si tienes GPU NVIDIA

### El pipeline se congela

**Causa:** A veces ocurre con tensores muy grandes  
**Solución:** Presiona `Ctrl+C` para interrumpir, luego reanuda:
```powershell
python .\train_pipeline.py --start-from <DATASET_SIGUIENTE>
```

### Error: `FileNotFoundError: config.yaml`

**Causa:** El script no está en `01_training/scripts/` o config.yaml no existe  
**Solución:**
```powershell
# Verificar ruta actual
cd C:\SARC-Drone\01_training\scripts
ls .\config*.py  # verificar que config_utils.py exista
cd ..\..\
ls config.yaml  # verificar que config.yaml exista en raíz
```

---

## Flujo rápido (resumen)

```powershell
# 1. Entorno
cd C:\SARC-Drone
python -m venv venv_sarc
venv_sarc\Scripts\activate
pip install -r requirements.txt

# 2. Verificación
cd 01_training\scripts
python .\integrity_check.py

# 3. data.yaml
python .\generate_dataset_yamls.py

# 4. Entrenamiento (central)
python .\train_pipeline.py              # ~13-16 horas con GPU

# 5. Exportación
python .\export_model.py

# 5.1 Go/No-Go
cd ..\..\
python .\05_tests\go_nogo_gate.py --profile model_candidate
python .\05_tests\go_nogo_gate.py --profile release_edge_android

# 6. Android (opcional)
cd ..\..\03_android_app\SARCApp
.\gradlew assembleRelease
adb install -r app\build\outputs\apk\release\app-release.apk
```

---

## Monitoreo en tiempo real

### Logs principales

Durante la ejecución, revisa estos archivos:

```powershell
# Runs de Ultralytics (por etapa)
# 01_training/runs/detection/<DATASET>/weights/

# Métricas finales por etapa
# 02_models/weights/results_<DATASET>.csv

# Pesos finales (después del pipeline)
# 02_models/weights/best_*.pt
```

### Dashboard de progreso (opcional)

```powershell
# Monitor en tiempo real (si usas VSCode)
# Terminal → Panel derecho → selecciona la terminal de ejecución
# O abre varios terminales, uno por etapa

# Monitoreo de GPU en paralelo (en otra terminal)
nvidia-smi -l 1  # actualiza cada segundo
```

---

## Checklist de validación

Antes de reportar éxito, verifica:

- [ ] Los 7 `best_*.pt` existen en `02_models/weights/`
- [ ] Los 7 `results_*.csv` existen en `02_models/weights/`
- [ ] Go/No-Go `model_candidate` en estado `GO`
- [ ] Go/No-Go `release_edge_android` en estado `GO`
- [ ] Archivos exportados existen:
  - [ ] `02_models/exported/tflite/best_C2A/best_C2A.tflite`
  - [ ] `02_models/exported/onnx/best_C2A.onnx`
  - [ ] `02_models/exported/openvino/best_C2A/`
- [ ] Integridad sin errores (todos verde)
- [ ] data.yaml generados para los 7 datasets
- [ ] Pipeline completó todas las 7 etapas

---

## Configuración personalizada

Para modificar hiperparámetros sin editar scripts, solo modifica `config.yaml`:

```yaml
training_pipeline:
  per_dataset:
    VisDrone:
      epochs: 300          # cambiar aquí
      batch_size: 32       # o aquí
      lr0: 0.001           # o aquí
      freeze: 0            # o aquí
```

Luego reinicia el pipeline. Los cambios se aplican automáticamente.

---

## Soporte y próximos pasos

**Primera fase completada:** ✓ Entrenamiento curricular con 7 etapas

**Próximas fases (no incluidas en este manual):**
- Fase 3-5: Optimizaciones, exportación avanzada, Android integración
- Fase 6: IoT/MQTT, cloud backend, autonomía del dron

Para detalles adicionales, revisa [README.md](README.md) y [04_docs/flujo_comandos.md](04_docs/flujo_comandos.md).

---

**Fin del manual. ¡Bienvenido al pipeline SARC-Drone!** 🚀
