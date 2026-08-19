from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO


# ============================================================================
# SAR YOLO26
# EXP07 - TARGETED EXTREME SMALL + DENSE SCENE + CLOSE NEIGHBORS V1
# ============================================================================
#
# HIPOTESIS
# ---------
# EXP04 demostró que los dense-scene targeted crops mejoran el recall de
# PERSON pequeñas.
#
# El análisis posterior mostró que la mejora se concentra especialmente en:
#
#   EXTREME_SMALL + DENSE_SCENE
#   EXTREME_SMALL + CLOSE_NEIGHBORS
#   EXTREME_SMALL + DENSE_SCENE + CLOSE_NEIGHBORS
#
# EXP07 añade al pipeline de EXP04 un subconjunto controlado de 3.500 crops
# del conjunto TRIPLE:
#
#   EXTREME_SMALL
#        +
#   DENSE_SCENE
#        +
#   CLOSE_NEIGHBORS
#
# CONTROLES
# ----------
# - máximo 8 crops por imagen fuente
# - 3.500 crops objetivo
# - seed 42
# - TRAIN original + EXP04 dense crops + EXP07 triple crops
#
# SEGURIDAD / REPRODUCIBILIDAD
# ----------------------------
# - No modifica dataset original.
# - No modifica labels originales.
# - No modifica YAML oficial.
# - Usa únicamente el YAML experimental de EXP07.
# - save_period=1 para guardar checkpoints por época.
# - Si existe last.pt, puede reanudar el entrenamiento.
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

EXPERIMENT_NAME = (
    "exp07_targeted_extreme_small_dense_neighbor"
)

EPOCHS = 100

IMAGE_SIZE = 960

BATCH_SIZE = 8

WORKERS = 8

DEVICE = 0

SEED = 42

PROJECT_NAME = (
    "exp07_targeted_extreme_small_dense_neighbor_v1"
)


# ============================================================================
# LOCALIZACIÓN DEL PROYECTO
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_baseline_dir() -> Path:

    for parent in [
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
    ]:

        if parent.name.lower() == "baseline":
            return parent

    raise RuntimeError(
        "No se pudo localizar el directorio baseline."
    )


BASELINE_DIR = find_baseline_dir()

PROJECT_ROOT = BASELINE_DIR.parents[4]


# ============================================================================
# EXP07 TRAINING DIRECTORY
# ============================================================================

EXP07_TRAINING_DIR = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp07_targeted_extreme_small_dense_neighbor_v1"
)


# ============================================================================
# DATASET EXPERIMENTAL
# ============================================================================

DATA_YAML = (
    EXP07_TRAINING_DIR
    / "exp07_dataset.yaml"
)


# ============================================================================
# MODELO PREENTRENADO
# ============================================================================

PRETRAINED_MODEL = (
    BASELINE_DIR
    / "yolo26s.pt"
)


# ============================================================================
# RUN DIRECTORY
# ============================================================================

RUNS_DIR = (
    EXP07_TRAINING_DIR
    / "runs"
)


RUN_NAME = (
    EXPERIMENT_NAME
)


# ============================================================================
# CHECKPOINTS
# ============================================================================

RUN_DIR = (
    RUNS_DIR
    / RUN_NAME
)


LAST_PT = (
    RUN_DIR
    / "weights"
    / "last.pt"
)


BEST_PT = (
    RUN_DIR
    / "weights"
    / "best.pt"
)


# ============================================================================
# REPORT
# ============================================================================

REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / PROJECT_NAME
    / "reports"
)


SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP07_TARGETED_EXTREME_SMALL_DENSE_NEIGHBOR_V1_SUMMARY.txt"
)


# ============================================================================
# VALIDACIÓN
# ============================================================================

def validate_structure() -> None:

    print()
    print("=" * 72)
    print("VALIDANDO ESTRUCTURA EXP07")
    print("=" * 72)

    required = {
        "PROJECT_ROOT":
            PROJECT_ROOT,

        "BASELINE_DIR":
            BASELINE_DIR,

        "EXP07_TRAINING_DIR":
            EXP07_TRAINING_DIR,

        "DATA_YAML":
            DATA_YAML,

        "PRETRAINED_MODEL":
            PRETRAINED_MODEL,
    }

    for name, path in required.items():

        if not path.exists():

            raise FileNotFoundError(
                f"No se encontró {name}:\n{path}"
            )

        print(
            f"[OK] {name}"
        )

        print(
            f"     {path}"
        )


# ============================================================================
# ESTADO
# ============================================================================

def print_experiment_configuration() -> None:

    print()
    print("=" * 72)
    print("CONFIGURACIÓN EXP07")
    print("=" * 72)

    print()
    print(
        f"MODEL       : {PRETRAINED_MODEL}"
    )

    print(
        f"DATA        : {DATA_YAML}"
    )

    print(
        f"PROJECT     : {RUNS_DIR}"
    )

    print(
        f"NAME        : {RUN_NAME}"
    )

    print(
        f"IMGSZ       : {IMAGE_SIZE}"
    )

    print(
        f"BATCH       : {BATCH_SIZE}"
    )

    print(
        f"EPOCHS      : {EPOCHS}"
    )

    print(
        f"WORKERS     : {WORKERS}"
    )

    print(
        f"DEVICE      : {DEVICE}"
    )

    print(
        f"SEED        : {SEED}"
    )

    print(
        f"LAST.PT     : {LAST_PT}"
    )

    print(
        f"BEST.PT     : {BEST_PT}"
    )


# ============================================================================
# RESUME AUTOMÁTICO
# ============================================================================

def find_resume_checkpoint() -> Path | None:

    if LAST_PT.exists():
        return LAST_PT

    return None


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_from_scratch() -> None:

    print()
    print("=" * 72)
    print("INICIANDO ENTRENAMIENTO EXP07 DESDE YOLO26S")
    print("=" * 72)

    model = YOLO(
        str(PRETRAINED_MODEL)
    )

    print()
    print(
        "[OK] Modelo YOLO26s cargado."
    )

    print()
    print(
        "Entrenamiento:"
    )

    print(
        "  TRAIN = original + EXP04 dense + EXP07 triple"
    )

    print(
        f"  imgsz  = {IMAGE_SIZE}"
    )

    print(
        f"  batch  = {BATCH_SIZE}"
    )

    print(
        f"  epochs = {EPOCHS}"
    )

    print(
        f"  workers = {WORKERS}"
    )

    print(
        f"  device = {DEVICE}"
    )

    print()

    results = model.train(

        data=str(
            DATA_YAML
        ),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        device=DEVICE,

        seed=SEED,

        project=str(
            RUNS_DIR
        ),

        name=RUN_NAME,

        exist_ok=True,

        pretrained=True,

        optimizer="auto",

        amp=True,

        cache=False,

        save=True,

        save_period=1,

        verbose=True,

        plots=True,

        val=True,

    )

    return results


def resume_training(
    checkpoint: Path,
) -> None:

    print()
    print("=" * 72)
    print("REANUDANDO ENTRENAMIENTO EXP07")
    print("=" * 72)

    print()
    print(
        f"[OK] Checkpoint encontrado:"
    )

    print(
        f"     {checkpoint}"
    )

    model = YOLO(
        str(checkpoint)
    )

    print()
    print(
        "[OK] Checkpoint cargado."
    )

    print()
    print(
        "Continuando hasta:"
        f" {EPOCHS} épocas."
    )

    model.train(
        resume=True,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def write_summary(
    resumed: bool,
) -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "=" * 72,
        "SAR YOLO26 - EXP07 TARGETED EXTREME SMALL DENSE NEIGHBOR V1",
        "=" * 72,
        "",
        "HIPÓTESIS",
        (
            "Mejorar la detección de PERSON extremadamente pequeñas "
            "cuando coinciden DENSE_SCENE y CLOSE_NEIGHBORS."
        ),
        "",
        "DATASET EXPERIMENTAL",
        str(DATA_YAML),
        "",
        "COMPOSICIÓN",
        "TRAIN original",
        "+ EXP04 dense crops",
        "+ 3.500 EXP07 triple crops",
        "",
        "CONFIGURACIÓN",
        f"epochs  = {EPOCHS}",
        f"imgsz   = {IMAGE_SIZE}",
        f"batch   = {BATCH_SIZE}",
        f"workers = {WORKERS}",
        f"device  = {DEVICE}",
        f"seed    = {SEED}",
        f"AMP     = True",
        f"cache   = False",
        f"save_period = 1",
        "",
        f"TRAINING RESUMED: {resumed}",
        "",
        "MODELO PREENTRENADO",
        str(PRETRAINED_MODEL),
        "",
        "OUTPUT",
        str(RUN_DIR),
        "",
        "IMPORTANTE",
        "Dataset original NO modificado.",
        "Labels originales NO modificados.",
        "YAML oficial NO modificado.",
        "YAML utilizado exclusivamente:",
        str(DATA_YAML),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - EXP07 TARGETED EXTREME SMALL"
    )
    print(
        "# DENSE SCENE + CLOSE NEIGHBORS V1"
    )
    print("=" * 72)

    validate_structure()

    print_experiment_configuration()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    resume_checkpoint = (
        find_resume_checkpoint()
    )

    # ------------------------------------------------------------------------
    # Resume automático
    # ------------------------------------------------------------------------

    if resume_checkpoint is not None:

        print()
        print(
            "[INFO] Existe last.pt."
        )

        print(
            "[INFO] EXP07 se reanudará automáticamente."
        )

        write_summary(
            resumed=True
        )

        resume_training(
            resume_checkpoint
        )

        return

    # ------------------------------------------------------------------------
    # Nuevo entrenamiento
    # ------------------------------------------------------------------------

    print()
    print(
        "[INFO] No existe last.pt."
    )

    print(
        "[INFO] Se iniciará EXP07 desde YOLO26s."
    )

    write_summary(
        resumed=False
    )

    train_from_scratch()

    print()
    print("=" * 72)
    print(
        "# ENTRENAMIENTO EXP07 FINALIZADO"
    )
    print("=" * 72)

    print()
    print(
        f"RUN: {RUN_DIR}"
    )

    print()
    print(
        f"BEST.PT esperado:"
    )

    print(
        f"{BEST_PT}"
    )

    print()
    print(
        f"LAST.PT esperado:"
    )

    print(
        f"{LAST_PT}"
    )

    print()
    print(
        f"SUMMARY:"
    )

    print(
        f"{SUMMARY_TXT}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "[CANCELADO] "
            "Entrenamiento interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print(
            "[ERROR EXP07]"
        )
        print("=" * 72)
        print()
        print(
            str(exc)
        )
        print()

        sys.exit(1)