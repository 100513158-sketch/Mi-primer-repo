# Flujo de comandos SARC-Drone (configuracion centralizada)

Este flujo usa config.yaml como fuente unica de rutas y parametros.

## Estado del documento

- Ultima actualizacion: 2026-04-18
- Estado: alineado con scripts actuales en 01_training/scripts y config.yaml en raiz.

## Regla de mantenimiento

Cuando cambie cualquier script o ruta del proyecto, actualizar primero este archivo y luego README.md.
Checklist minimo de sincronizacion:

1. Verificar que los comandos apunten a rutas reales.
2. Verificar que los scripts usados existan en 01_training/scripts.
3. Verificar que las rutas de datos y modelos sigan definidas en config.yaml.
4. Verificar comandos de build Android y rutas de APK.

## Verificacion rapida pre-ejecucion

Ejecuta este bloque antes de correr cualquier fase para confirmar que la estructura minima existe:

```powershell
cd C:\SARC-Drone

$checks = @(
	@{ Name = "requirements"; Path = "C:\SARC-Drone\requirements.txt"; Type = "file" },
	@{ Name = "config"; Path = "C:\SARC-Drone\config.yaml"; Type = "file" },
	@{ Name = "dataset_config"; Path = "C:\SARC-Drone\00_datasets\data.yaml"; Type = "file" },
	@{ Name = "raw_dir"; Path = "C:\SARC-Drone\00_datasets\raw"; Type = "dir" },
	@{ Name = "processed_dir"; Path = "C:\SARC-Drone\00_datasets\processed"; Type = "dir" },
	@{ Name = "scripts_dir"; Path = "C:\SARC-Drone\01_training\scripts"; Type = "dir" },
	@{ Name = "train_script"; Path = "C:\SARC-Drone\01_training\scripts\train.py"; Type = "file" },
	@{ Name = "export_script"; Path = "C:\SARC-Drone\01_training\scripts\export_model.py"; Type = "file" },
	@{ Name = "android_dir"; Path = "C:\SARC-Drone\03_android_app\SARCApp"; Type = "dir" }
)

$failed = @()

foreach ($c in $checks) {
	$ok = Test-Path $c.Path
	if ($ok) {
		Write-Host "[OK]   $($c.Name): $($c.Path)" -ForegroundColor Green
	} else {
		Write-Host "[FAIL] $($c.Name): $($c.Path)" -ForegroundColor Red
		$failed += $c
	}
}

if ($failed.Count -gt 0) {
	Write-Host "\nFaltan rutas/archivos requeridos. Corrige antes de continuar." -ForegroundColor Red
} else {
	Write-Host "\nVerificacion completada. Estructura minima OK." -ForegroundColor Cyan
}
```

Chequeo rapido de GPU y paquetes principales:

```powershell
cd C:\SARC-Drone
venv_sarc\Scripts\activate
python -c "import torch; import yaml; import ultralytics; print('CUDA:', torch.cuda.is_available()); print('Torch OK'); print('PyYAML OK'); print('Ultralytics OK')"
```

Chequeo rapido de artefactos tras entrenamiento/exportacion:

```powershell
cd C:\SARC-Drone

if (Test-Path .\02_models\weights\best_detection.pt) { "OK best_detection.pt" } else { "FALTA best_detection.pt" }
if (Test-Path .\02_models\weights\best_pose.pt) { "OK best_pose.pt" } else { "FALTA best_pose.pt" }
if (Test-Path .\02_models\exported\tflite) { "OK carpeta tflite" } else { "FALTA carpeta tflite" }
if (Test-Path .\02_models\exported\onnx) { "OK carpeta onnx" } else { "FALTA carpeta onnx" }
if (Test-Path .\02_models\exported\openvino) { "OK carpeta openvino" } else { "FALTA carpeta openvino" }
```

## 1) Entorno Python

```powershell
cd C:\SARC-Drone
python -m venv venv_sarc
venv_sarc\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Verificacion rapida

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

## 3) Preparar raw datasets

```powershell
cd C:\SARC-Drone\00_datasets\raw
mkdir visdrone, okutama, nitc_rescue
```

## 4) Descargar VisDrone y convertir

```powershell
cd C:\SARC-Drone\01_training\scripts
powershell -ExecutionPolicy Bypass -File .\download_visdrone.ps1
python .\convert_visdrone.py
```

## 5) Agregar OKUTAMA y NITC manualmente

```powershell
cd C:\SARC-Drone\00_datasets\raw\nitc_rescue
git clone https://github.com/shubhasreeav/NITC-Person-Rescue.git

cd C:\SARC-Drone\01_training\scripts
python .\convert_okutama.py --split train
```

## 6) Mezclar train/val/test

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\merge_datasets.py --sources visdrone okutama nitc_rescue --val-ratio 0.15 --test-ratio 0.10
```

## 7) Entrenamiento

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\train.py
```

## 8) Exportacion

```powershell
cd C:\SARC-Drone\01_training\scripts
python .\export_model.py
python .\extreme_optimization.py
```

## 9) Build Android

```powershell
cd C:\SARC-Drone\03_android_app\SARCApp
.\gradlew assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

## 10) Operacion diaria corta

```powershell
cd C:\SARC-Drone
venv_sarc\Scripts\activate
python .\01_training\scripts\train.py
python .\01_training\scripts\export_model.py
```

## 11) Fase 5 - IoT/MQTT y control remoto

Verifica que config.yaml tenga secciones iot y autonomy.

```powershell
cd C:\SARC-Drone
python -c "import yaml; c=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); print('iot' in c, 'autonomy' in c)"
```

Compilar Android con soporte MQTT:

```powershell
cd C:\SARC-Drone\03_android_app\SARCApp
.\gradlew assembleDebug
```

En ejecución, la app usa estos módulos:

- MqttManager.kt
- CommandProcessor.kt
- DroneSimulator.kt

## 12) Fase 5 - Backend VMware Debian y almacenamiento

```powershell
cd C:\SARC-Drone\04_docs\vmware_debian
docker compose up -d --build
```

Verificaciones rapidas:

- http://IP_DE_LA_VM:8000/health
- http://IP_DE_LA_VM:8000/console

## 13) Fase 6 - Autonomia avanzada

Módulos Android incluidos para autonomia:

- SearchPattern.kt (lawnmower, spiral, waypoint)
- ObstacleAvoidance.kt
- BatteryManager.kt

Validación rápida de batería y retorno:

```powershell
cd C:\SARC-Drone
python -c "import yaml; c=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); print('battery=', c['autonomy']['battery'])"
```

## 14) Pruebas sin dron real (simulacion)

La simulación queda habilitada por Config.SIMULATION_ENABLED.

```powershell
cd C:\SARC-Drone\03_android_app\SARCApp
.\gradlew installDebug
adb shell am start -n com.sarc.drone.vision/.VisionActivity
```

## Nota

Si necesitas cambiar rutas, hiperparametros o umbrales, edita solo C:\SARC-Drone\config.yaml.
