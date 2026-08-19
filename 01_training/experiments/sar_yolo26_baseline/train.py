from __future__ import annotations

import gc
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

PROJECT_ROOT = Path(r"C:\SARC-Drone")

# Dataset
DATASET_YAML = (
    PROJECT_ROOT
    / "00_datasets"
    / "SAR_DATASET_STUDIO"
    / "configs"
    / "sar_visdrone_2class.yaml"
)

# Experimento actual
EXP_DIR = (
    PROJECT_ROOT
    / "01_training"
    / "experiments"
    / "sar_yolo26_baseline"
)

# Modelo preentrenado
MODEL_PATH = (
    PROJECT_ROOT
    / "01_training"
    / "models"
    / "pretrained"
    / "yolo26n.pt"
)


# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================

MODEL_NAME = "yolo26n"

EPOCHS = 100
IMAGE_SIZE = 640

# MUY IMPORTANTE:
# Primera ejecución deliberadamente conservadora.
BATCH_SIZE = 8

# Workers bajos para evitar presión innecesaria de RAM/CPU en Windows.
WORKERS = 2

DEVICE = 0

# AMP reduce consumo de VRAM y acelera entrenamiento en GPU moderna.
AMP = True

# Guardar checkpoint cada N épocas.
SAVE_PERIOD = 10

# Early stopping.
PATIENCE = 25

# Optimización.
OPTIMIZER = "auto"

# Learning rate.
LR0 = 0.01
LRF = 0.01

# Seed para reproducibilidad.
SEED = 42

# Protección de memoria.
#
# La RTX 5070 tiene ~12 GB.
# Dejamos deliberadamente un margen porque Windows ya consume VRAM.
MAX_GPU_MEMORY_GB = 7.0

# No permitir batch superior a este valor.
MAX_BATCH_GPU = 8


# =============================================================================
# DIRECTORIOS
# =============================================================================

CONFIG_DIR = EXP_DIR / "configs"
LOG_DIR = EXP_DIR / "logs"
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR = EXP_DIR / "plots"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"

ALL_DIRS = [
    CONFIG_DIR,
    LOG_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
]


# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = LOG_DIR / "train.log"

logger = logging.getLogger("sarc_drone_training")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a",
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# =============================================================================
# UTILIDADES
# =============================================================================

def create_directories() -> None:
    """Crea la estructura del experimento."""

    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def validate_paths() -> None:
    """Comprueba que las rutas críticas existen."""

    logger.info("=" * 80)
    logger.info("VALIDACIÓN DE RUTAS")
    logger.info("=" * 80)

    logger.info("Proyecto : %s", PROJECT_ROOT)
    logger.info("Dataset  : %s", DATASET_YAML)
    logger.info("Modelo   : %s", MODEL_PATH)
    logger.info("Experim. : %s", EXP_DIR)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"No existe el proyecto: {PROJECT_ROOT}"
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"No existe el YAML: {DATASET_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {MODEL_PATH}"
        )

    if MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(
            f"El modelo existe pero está vacío: {MODEL_PATH}"
        )

    logger.info("OK: YAML encontrado")
    logger.info("OK: modelo encontrado")
    logger.info(
        "Tamaño modelo: %.2f MB",
        MODEL_PATH.stat().st_size / (1024 * 1024),
    )


def configure_gpu() -> tuple[bool, int]:
    """
    Configura GPU y protección de VRAM.

    Devuelve:
        (gpu_disponible, batch_efectivo)
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("CONFIGURACIÓN GPU")
    logger.info("=" * 80)

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible.")
        logger.warning("El entrenamiento se ejecutará en CPU.")

        return False, min(BATCH_SIZE, 2)

    device_index = DEVICE

    gpu_name = torch.cuda.get_device_name(device_index)

    total_vram_bytes = torch.cuda.get_device_properties(
        device_index
    ).total_memory

    total_vram_gb = total_vram_bytes / (1024 ** 3)

    logger.info("GPU       : %s", gpu_name)
    logger.info("VRAM total: %.2f GB", total_vram_gb)

    # -------------------------------------------------------------------------
    # Límite de VRAM
    # -------------------------------------------------------------------------

    target_gb = min(
        MAX_GPU_MEMORY_GB,
        total_vram_gb * 0.80,
    )

    fraction = target_gb / total_vram_gb

    # Mantener el valor dentro de límites razonables.
    fraction = max(0.05, min(0.90, fraction))

    try:
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device_index,
        )

        logger.info(
            "Límite VRAM proceso: %.2f GB (%.1f%%)",
            target_gb,
            fraction * 100.0,
        )

    except Exception as exc:
        logger.warning(
            "No se pudo establecer límite VRAM: %s",
            exc,
        )

    # -------------------------------------------------------------------------
    # Backends
    # -------------------------------------------------------------------------

    torch.backends.cudnn.benchmark = True

    # TF32 mejora rendimiento en GPUs modernas.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------------------------------------------------------------
    # Batch
    # -------------------------------------------------------------------------

    effective_batch = min(
        BATCH_SIZE,
        MAX_BATCH_GPU,
    )

    logger.info(
        "Batch configurado : %d",
        effective_batch,
    )

    logger.info(
        "Workers configurados: %d",
        WORKERS,
    )

    logger.info(
        "AMP: %s",
        AMP,
    )

    return True, effective_batch


def cleanup_gpu() -> None:
    """Libera memoria CUDA después de operaciones."""

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def log_gpu_memory(prefix: str = "") -> None:
    """Registra consumo actual de VRAM."""

    if not torch.cuda.is_available():
        return

    try:
        allocated = torch.cuda.memory_allocated(DEVICE)
        reserved = torch.cuda.memory_reserved(DEVICE)

        allocated_gb = allocated / (1024 ** 3)
        reserved_gb = reserved / (1024 ** 3)

        logger.info(
            "%sVRAM allocated: %.2f GB | reserved: %.2f GB",
            prefix,
            allocated_gb,
            reserved_gb,
        )

    except Exception:
        pass


def save_experiment_config(batch_effective: int) -> None:
    """
    Guarda una copia de la configuración utilizada.
    """

    config_file = CONFIG_DIR / "training_config.txt"

    content = f"""
SARC-Drone
YOLO26 baseline training

PROJECT_ROOT:
{PROJECT_ROOT}

DATASET_YAML:
{DATASET_YAML}

MODEL:
{MODEL_PATH}

MODEL_NAME:
{MODEL_NAME}

EPOCHS:
{EPOCHS}

IMAGE_SIZE:
{IMAGE_SIZE}

BATCH_REQUESTED:
{BATCH_SIZE}

BATCH_EFFECTIVE:
{batch_effective}

WORKERS:
{WORKERS}

DEVICE:
{DEVICE}

AMP:
{AMP}

SAVE_PERIOD:
{SAVE_PERIOD}

PATIENCE:
{PATIENCE}

OPTIMIZER:
{OPTIMIZER}

LR0:
{LR0}

LRF:
{LRF}

SEED:
{SEED}

MAX_GPU_MEMORY_GB:
{MAX_GPU_MEMORY_GB}

MAX_BATCH_GPU:
{MAX_BATCH_GPU}
""".strip()

    config_file.write_text(
        content,
        encoding="utf-8",
    )

    logger.info(
        "Configuración guardada: %s",
        config_file,
    )


def copy_results(run_dir: Path) -> None:
    """
    Copia resultados importantes desde el directorio de Ultralytics
    al directorio estructurado del experimento.
    """

    if not run_dir.exists():
        logger.warning(
            "No existe directorio de resultados: %s",
            run_dir,
        )
        return

    logger.info("")
    logger.info("COPIANDO RESULTADOS")

    # results.csv
    results_csv = run_dir / "results.csv"

    if results_csv.exists():
        destination = RESULTS_DIR / "results.csv"

        shutil.copy2(
            results_csv,
            destination,
        )

        logger.info(
            "OK: results.csv -> %s",
            destination,
        )

    # Pesos
    weights_dir = run_dir / "weights"

    if weights_dir.exists():

        for filename in [
            "best.pt",
            "last.pt",
        ]:

            source = weights_dir / filename

            if source.exists():

                destination = CHECKPOINTS_DIR / filename

                shutil.copy2(
                    source,
                    destination,
                )

                logger.info(
                    "OK: %s -> %s",
                    filename,
                    destination,
                )

    # Gráficas generadas por Ultralytics
    for pattern in [
        "*.png",
        "*.jpg",
        "*.jpeg",
    ]:

        for source in run_dir.glob(pattern):

            destination = PLOTS_DIR / source.name

            shutil.copy2(
                source,
                destination,
            )

    # args.yaml
    args_yaml = run_dir / "args.yaml"

    if args_yaml.exists():

        destination = CONFIG_DIR / "ultralytics_args.yaml"

        shutil.copy2(
            args_yaml,
            destination,
        )

    # Confusion matrix, PR curves, etc.
    logger.info(
        "Resultados organizados en: %s",
        EXP_DIR,
    )


def write_final_report(
    run_dir: Path,
    elapsed_seconds: float,
    batch_effective: int,
) -> None:
    """Genera informe textual final."""

    report_file = REPORTS_DIR / "training_summary.txt"

    elapsed_hours = elapsed_seconds / 3600

    lines = [
        "SARC-DRONE - YOLO26 BASELINE",
        "=" * 70,
        "",
        f"Modelo: {MODEL_PATH}",
        f"Dataset YAML: {DATASET_YAML}",
        "",
        f"Epochs configuradas: {EPOCHS}",
        f"Image size: {IMAGE_SIZE}",
        f"Batch efectivo: {batch_effective}",
        f"Workers: {WORKERS}",
        f"AMP: {AMP}",
        "",
        f"Tiempo entrenamiento: {elapsed_hours:.2f} horas",
        "",
        f"Directorio Ultralytics: {run_dir}",
        f"Resultados: {RESULTS_DIR}",
        f"Plots: {PLOTS_DIR}",
        f"Checkpoints: {CHECKPOINTS_DIR}",
        "",
    ]

    report_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    logger.info(
        "Informe final: %s",
        report_file,
    )


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train() -> int:

    start_time = time.time()

    create_directories()

    logger.info("")
    logger.info("=" * 80)
    logger.info("SARC-DRONE - ENTRENAMIENTO YOLO26")
    logger.info("=" * 80)

    try:

        # ---------------------------------------------------------------------
        # 1. Rutas
        # ---------------------------------------------------------------------

        validate_paths()

        # ---------------------------------------------------------------------
        # 2. GPU
        # ---------------------------------------------------------------------

        has_gpu, batch_effective = configure_gpu()

        save_experiment_config(
            batch_effective=batch_effective,
        )

        # ---------------------------------------------------------------------
        # 3. Cargar modelo
        # ---------------------------------------------------------------------

        logger.info("")
        logger.info("=" * 80)
        logger.info("CARGANDO MODELO")
        logger.info("=" * 80)

        logger.info(
            "Modelo: %s",
            MODEL_PATH,
        )

        model = YOLO(
            str(MODEL_PATH)
        )

        logger.info(
            "OK: modelo YOLO26 cargado"
        )

        log_gpu_memory(
            prefix="Después de cargar modelo: "
        )

        # ---------------------------------------------------------------------
        # 4. Directorio Ultralytics
        # ---------------------------------------------------------------------

        ultralytics_project = RESULTS_DIR / "ultralytics"

        run_name = "yolo26n_visdrone_2class"

        logger.info("")
        logger.info("=" * 80)
        logger.info("CONFIGURACIÓN ENTRENAMIENTO")
        logger.info("=" * 80)

        logger.info(
            "Dataset YAML : %s",
            DATASET_YAML,
        )

        logger.info(
            "Modelo       : %s",
            MODEL_PATH,
        )

        logger.info(
            "Epochs       : %d",
            EPOCHS,
        )

        logger.info(
            "Image size   : %d",
            IMAGE_SIZE,
        )

        logger.info(
            "Batch        : %d",
            batch_effective,
        )

        logger.info(
            "Workers      : %d",
            WORKERS,
        )

        logger.info(
            "Device       : %s",
            "GPU 0" if has_gpu else "CPU",
        )

        logger.info(
            "AMP          : %s",
            AMP,
        )

        logger.info(
            "VRAM límite  : %.2f GB",
            MAX_GPU_MEMORY_GB,
        )

        # ---------------------------------------------------------------------
        # 5. Entrenamiento
        # ---------------------------------------------------------------------

        logger.info("")
        logger.info("=" * 80)
        logger.info("INICIANDO ENTRENAMIENTO")
        logger.info("=" * 80)

        training_start = time.time()

        results = model.train(
            data=str(DATASET_YAML),

            epochs=EPOCHS,

            imgsz=IMAGE_SIZE,

            batch=batch_effective,

            workers=WORKERS,

            device=DEVICE if has_gpu else "cpu",

            amp=AMP,

            optimizer=OPTIMIZER,

            lr0=LR0,

            lrf=LRF,

            patience=PATIENCE,

            save=True,

            save_period=SAVE_PERIOD,

            seed=SEED,

            project=str(ultralytics_project),

            name=run_name,

            exist_ok=True,

            pretrained=True,

            verbose=True,

        )

        elapsed_seconds = time.time() - training_start

        logger.info("")
        logger.info("=" * 80)
        logger.info("ENTRENAMIENTO FINALIZADO")
        logger.info("=" * 80)

        logger.info(
            "Tiempo: %.2f minutos",
            elapsed_seconds / 60,
        )

        log_gpu_memory(
            prefix="Final: "
        )

        # ---------------------------------------------------------------------
        # 6. Localizar resultados
        # ---------------------------------------------------------------------

        run_dir = ultralytics_project / run_name

        logger.info(
            "Directorio entrenamiento: %s",
            run_dir,
        )

        # ---------------------------------------------------------------------
        # 7. Copiar resultados a estructura del experimento
        # ---------------------------------------------------------------------

        copy_results(
            run_dir=run_dir,
        )

        # ---------------------------------------------------------------------
        # 8. Métricas
        # ---------------------------------------------------------------------

        try:

            metrics = results.results_dict

            logger.info("")
            logger.info("=" * 80)
            logger.info("MÉTRICAS FINALES")
            logger.info("=" * 80)

            for key, value in metrics.items():

                try:
                    logger.info(
                        "%-35s : %.6f",
                        key,
                        float(value),
                    )

                except (TypeError, ValueError):
                    logger.info(
                        "%-35s : %s",
                        key,
                        value,
                    )

        except Exception as exc:

            logger.warning(
                "No se pudieron extraer métricas finales: %s",
                exc,
            )

        # ---------------------------------------------------------------------
        # 9. Informe
        # ---------------------------------------------------------------------

        write_final_report(
            run_dir=run_dir,
            elapsed_seconds=elapsed_seconds,
            batch_effective=batch_effective,
        )

        # ---------------------------------------------------------------------
        # 10. Limpieza
        # ---------------------------------------------------------------------

        cleanup_gpu()

        total_time = time.time() - start_time

        logger.info("")
        logger.info("=" * 80)
        logger.info("PROCESO COMPLETADO CORRECTAMENTE")
        logger.info("=" * 80)

        logger.info(
            "Tiempo total: %.2f minutos",
            total_time / 60,
        )

        logger.info(
            "Experimento: %s",
            EXP_DIR,
        )

        logger.info(
            "Best model: %s",
            CHECKPOINTS_DIR / "best.pt",
        )

        logger.info(
            "Results CSV: %s",
            RESULTS_DIR / "results.csv",
        )

        return 0

    except KeyboardInterrupt:

        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("ENTRENAMIENTO INTERRUMPIDO POR EL USUARIO")
        logger.warning("=" * 80)

        cleanup_gpu()

        return 130

    except RuntimeError as exc:

        logger.exception(
            "ERROR DE RUNTIME DURANTE EL ENTRENAMIENTO"
        )

        if "out of memory" in str(exc).lower():

            logger.error("")
            logger.error(
                "POSIBLE OOM DE GPU."
            )

            logger.error(
                "Para la siguiente ejecución reducir BATCH_SIZE de %d a 4.",
                BATCH_SIZE,
            )

        cleanup_gpu()

        return 2

    except Exception as exc:

        logger.exception(
            "ERROR DURANTE EL ENTRENAMIENTO: %s",
            exc,
        )

        cleanup_gpu()

        return 1


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        train()
    )