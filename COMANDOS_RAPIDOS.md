# Comandos de ejecución — SARC-Drone (orden exacto)

## Fase 0: Activar entorno

```powershell
cd C:\SARC-Drone
.venv\Scripts\activate
```

**Verificar:**
```powershell
python -c "import torch; from ultralytics import YOLO; print('✓ OK')"
```

---

## Fase 1: Conversión raw → YOLO (primera vez o al cambiar datos)

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\convert_all_datasets.py
```

Limpia `00_datasets/processed/` y regenera los 7 datasets desde `raw/`.

**Opciones:**
```powershell
python .\convert_all_datasets.py --no-clean          # conservar lo que ya está
python .\convert_all_datasets.py --dataset VisDrone  # solo un dataset
```

**Esperar:** 7 líneas [OK] con conteo de imágenes por split

---

## Fase 2: Verificación de integridad

```powershell
python .\integrity_check.py
```

**Esperar:** Todo verde [OK]

---

## Fase 3: Generar data.yaml

```powershell
python .\generate_dataset_yamls.py
```

**Esperar:** 7 archivos generados

---

## Fase 4: Entrenamiento (central - 13-16h con GPU)

```powershell
python .\train_pipeline.py
```

**Esperar:** 7 etapas completadas. Salida final muestra:
```
Etapas OK   : 7 / 7
Mejor modelo: best_C2A.pt
```

---

## Fase 5: Exportación

```powershell
python .\export_model.py
```

**Esperar:** 3 tipos de modelos exportados (TFLite, ONNX, OpenVINO)

---

## Fase 5.1: Gate Go/No-Go

```powershell
cd C:\SARC-Drone

# Candidato de modelo
python .\05_tests\go_nogo_gate.py --profile model_candidate

# Liberación edge/android (requiere exportaciones)
python .\05_tests\go_nogo_gate.py --profile release_edge_android
```

**Esperar:** Estado `GO` en ambos perfiles antes de desplegar

---

## Fase 5.2: Orquestador post-entrenamiento (automático)

```powershell
cd C:\SARC-Drone

# Ejecuta gate candidato -> export -> gate release -> build
python .\05_tests\run_post_training_release.py

# Opcional: sin build Android
python .\05_tests\run_post_training_release.py --skip-build

# Opcional: sin optimización extrema
python .\05_tests\run_post_training_release.py --no-extreme
```

**Salida:** reporte JSON en `05_tests\reports\post_training_release_*.json`

---

## Fase 6: Verificar resultados

```powershell
cd C:\SARC-Drone
Get-ChildItem "02_models\weights\best_*.pt"
Get-ChildItem "02_models\exported\tflite\best_C2A\best_C2A.tflite"
```

---

## Comandos opcionales / debugging

```powershell
# Reconvertir solo un dataset
python .\convert_all_datasets.py --dataset C2A

# Integridad de un solo dataset
python .\integrity_check.py --dataset VisDrone

# Reanudar pipeline desde etapa específica
python .\train_pipeline.py --start-from NITC

# Solo una etapa
python .\train_pipeline.py --only SeaDronesSee

# Simular sin entrenar
python .\train_pipeline.py --dry-run

# Monitorear GPU en paralelo (otra terminal)
nvidia-smi -l 1
```

---

## Resumen en 1 línea (pipeline completo)

```powershell
cd C:\SARC-Drone && .venv\Scripts\activate && cd 01_training\scripts && python .\convert_all_datasets.py && python .\integrity_check.py && python .\generate_dataset_yamls.py && python .\train_pipeline.py && python .\export_model.py
```

---

**Duración total estimada:**
- Fase 0 (activar entorno): 5 seg
- Fase 1 (conversión raw→YOLO): **30-60 min** (depende del tamaño de VisDrone)
- Fase 2 (integridad): 5 min
- Fase 3 (data.yaml): 1 min
- Fase 4 (entrenamiento): **13-16 horas** (GPU) o 2-3 días (CPU)
- Fase 5 (exportación): 20 min

**Total:** ~15-18 horas (con GPU) ⏱️
