#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAR YOLO26 - TRAIN BASELINE V1

Primer entrenamiento baseline reproducible del dataset
VisDrone_SAR_2CLASS_V1.

IMPORTANTE:
- No modifica el dataset.
- No elimina imágenes ni labels.
- Registra la configuración completa del experimento.
- Guarda pesos best.pt y last.pt.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
import ultralytics
import torch


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar"
    r"\cleaned\VisDrone_SAR_2CLASS_V1"
)

DATA_YAML = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\preparation"
    r"\prepare_yolo_dataset_v1\data.yaml"
)

TRAINING_ROOT = PROJECT_ROOT / "training"
RUNS_ROOT = TRAINING_ROOT / "runs"

RUN_NAME = "baseline_v1"

RUN_DIR = RUNS_ROOT / RUN_NAME
SUMMARY_FILE = RUN_DIR / "TRAINING_SUMMARY.txt"
CONFIG_FILE = RUN_DIR / "TRAINING_CONFIG.json"

# ------------------------------------------------------------
# Modelo
# ------------------------------------------------------------

# Modelo pequeño para establecer una primera referencia.
MODEL_NAME = "yolo26s.pt"

# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------

EPOCHS = 100
IMGSZ = 640

BATCH = 16

DEVICE = 0

WORKERS = 8

PATIENCE = 20

SEED = 42

PROJECT = str(RUNS_ROOT)

NAME = RUN_NAME

# ------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------

# Para el baseline dejamos los valores estándar de Ultralytics.
# No introducimos todavía una estrategia específica para SAR.

AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
}


# ============================================================
# UTILIDADES
# ============================================================

def print_header() -> None:
    print()
    print("=" * 70)
    print("# SAR YOLO26 - TRAIN BASELINE V1")
    print("=" * 70)
    print()


def validate_paths() -> None:

    print("Verificando rutas...")
    print()

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"No existe PROJECT_ROOT:\n{PROJECT_ROOT}"
        )

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"No existe DATASET_ROOT:\n{DATASET_ROOT}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"No existe DATA_YAML:\n{DATA_YAML}"
        )

    print("[OK] PROJECT_ROOT")
    print(PROJECT_ROOT)
    print()

    print("[OK] DATASET_ROOT")
    print(DATASET_ROOT)
    print()

    print("[OK] DATA_YAML")
    print(DATA_YAML)
    print()


def get_environment_info() -> dict:

    cuda_available = torch.cuda.is_available()

    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_executable": sys.executable,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count()
        if cuda_available else 0,
    }

    if cuda_available:

        info["gpu"] = torch.cuda.get_device_name(0)

        info["gpu_capability"] = (
            torch.cuda.get_device_capability(0)
        )

    else:

        info["gpu"] = None
        info["gpu_capability"] = None

    return info


def read_data_yaml() -> str:

    try:

        return DATA_YAML.read_text(
            encoding="utf-8"
        )

    except Exception as exc:

        return (
            f"[ERROR leyendo data.yaml: {exc}]"
        )


def create_run_directory() -> None:

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_configuration(environment: dict) -> None:

    config = {

        "experiment": {
            "name": RUN_NAME,
            "type": "baseline",
            "version": "v1",
        },

        "paths": {
            "project_root": str(PROJECT_ROOT),
            "dataset_root": str(DATASET_ROOT),
            "data_yaml": str(DATA_YAML),
            "run_dir": str(RUN_DIR),
        },

        "model": {
            "model_name": MODEL_NAME,
        },

        "training": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "workers": WORKERS,
            "patience": PATIENCE,
            "seed": SEED,
        },

        "augmentation": AUGMENTATION,

        "environment": environment,

        "data_yaml_content": read_data_yaml(),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print(f"[OK] Configuración guardada:")
    print(CONFIG_FILE)
    print()


def write_initial_summary(environment: dict) -> None:

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - TRAIN BASELINE V1\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            "EXPERIMENTO\n"
        )

        f.write(
            f"Nombre: {RUN_NAME}\n"
        )

        f.write(
            "Tipo: BASELINE\n"
        )

        f.write(
            f"Fecha inicio: "
            f"{datetime.now().isoformat()}\n\n"
        )

        f.write(
            "RUTAS\n"
        )

        f.write(
            f"Dataset:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"data.yaml:\n{DATA_YAML}\n\n"
        )

        f.write(
            f"Run:\n{RUN_DIR}\n\n"
        )

        f.write(
            "MODELO\n"
        )

        f.write(
            f"{MODEL_NAME}\n\n"
        )

        f.write(
            "CONFIGURACIÓN\n"
        )

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"Image size: {IMGSZ}\n"
        )

        f.write(
            f"Batch: {BATCH}\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Workers: {WORKERS}\n"
        )

        f.write(
            f"Patience: {PATIENCE}\n"
        )

        f.write(
            f"Seed: {SEED}\n\n"
        )

        f.write(
            "ENTORNO\n"
        )

        for key, value in environment.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        f.write(
            "DATA.YAML\n"
        )

        f.write("-" * 70 + "\n")

        f.write(
            read_data_yaml()
        )

        f.write("\n\n")

        f.write(
            "ESTADO\n"
        )

        f.write(
            "Entrenamiento iniciado.\n"
        )


def update_summary_after_training(
    results
) -> None:

    with open(
        SUMMARY_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("RESULTADO DEL ENTRENAMIENTO\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Fecha final: "
            f"{datetime.now().isoformat()}\n\n"
        )

        f.write(
            "RESULTADOS\n"
        )

        f.write(
            f"{results}\n\n"
        )

        f.write(
            "PESOS\n"
        )

        best_path = RUN_DIR / "weights" / "best.pt"
        last_path = RUN_DIR / "weights" / "last.pt"

        f.write(
            f"best.pt: {best_path}\n"
        )

        f.write(
            f"last.pt: {last_path}\n\n"
        )

        f.write(
            "IMPORTANTE\n"
        )

        f.write(
            "El dataset original y el dataset limpio "
            "no han sido modificados por este script.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header()

    validate_paths()

    create_run_directory()

    environment = get_environment_info()

    print("ENTORNO")
    print("-" * 70)

    print(
        f"Python       : "
        f"{platform.python_version()}"
    )

    print(
        f"Ultralytics  : "
        f"{ultralytics.__version__}"
    )

    print(
        f"PyTorch      : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA         : "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU          : "
        f"{environment.get('gpu')}"
    )

    print()

    print("CONFIGURACIÓN")
    print("-" * 70)

    print(
        f"Modelo       : {MODEL_NAME}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Image size   : {IMGSZ}"
    )

    print(
        f"Batch        : {BATCH}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Workers      : {WORKERS}"
    )

    print(
        f"Patience     : {PATIENCE}"
    )

    print(
        f"Seed         : {SEED}"
    )

    print()

    save_configuration(
        environment
    )

    write_initial_summary(
        environment
    )

    print(
        f"[OK] Resumen inicial:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print("=" * 70)
    print("CARGANDO MODELO")
    print("=" * 70)

    model = YOLO(
        MODEL_NAME
    )

    print()
    print(
        f"[OK] Modelo cargado: {MODEL_NAME}"
    )

    print()

    print("=" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 70)

    print()

    print(
        "Dataset:"
    )

    print(
        DATA_YAML
    )

    print()

    print(
        "Run:"
    )

    print(
        RUN_DIR
    )

    print()

    print(
        "IMPORTANTE: durante el entrenamiento "
        "no se modifica el dataset."
    )

    print()

    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,

        imgsz=IMGSZ,

        batch=BATCH,

        device=DEVICE,

        workers=WORKERS,

        patience=PATIENCE,

        seed=SEED,

        project=PROJECT,

        name=NAME,

        exist_ok=True,

        pretrained=True,

        verbose=True,

        # Augmentation baseline
        hsv_h=AUGMENTATION["hsv_h"],
        hsv_s=AUGMENTATION["hsv_s"],
        hsv_v=AUGMENTATION["hsv_v"],

        degrees=AUGMENTATION["degrees"],
        translate=AUGMENTATION["translate"],
        scale=AUGMENTATION["scale"],
        shear=AUGMENTATION["shear"],
        perspective=AUGMENTATION["perspective"],

        flipud=AUGMENTATION["flipud"],
        fliplr=AUGMENTATION["fliplr"],

        bgr=AUGMENTATION["bgr"],

        mosaic=AUGMENTATION["mosaic"],
        mixup=AUGMENTATION["mixup"],
        cutmix=AUGMENTATION["cutmix"],

        plots=True,

        save=True,

        save_period=10,
    )

    update_summary_after_training(
        results
    )

    print()
    print("=" * 70)
    print("# ENTRENAMIENTO BASELINE V1 FINALIZADO")
    print("=" * 70)

    print()

    print(
        "Run:"
    )

    print(
        RUN_DIR
    )

    print()

    print(
        "Best:"
    )

    print(
        RUN_DIR / "weights" / "best.pt"
    )

    print()

    print(
        "Last:"
    )

    print(
        RUN_DIR / "weights" / "last.pt"
    )

    print()

    print(
        "Resumen:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    print()


if __name__ == "__main__":

    main()