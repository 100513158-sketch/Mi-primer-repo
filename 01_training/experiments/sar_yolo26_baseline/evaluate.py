from pathlib import Path
import json
import csv
import logging
import time

from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(r"C:\SARC-Drone")

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "01_training"
    / "experiments"
    / "sar_yolo26_baseline"
)

MODEL_PATH = EXPERIMENT_DIR / "checkpoints" / "best.pt"

DATASET_YAML = (
    PROJECT_ROOT
    / "00_datasets"
    / "SAR_DATASET_STUDIO"
    / "configs"
    / "sar_visdrone_2class.yaml"
)

TEST_IMAGES = (
    PROJECT_ROOT
    / "00_datasets"
    / "SAR_DATASET_STUDIO"
    / "processed"
    / "sar"
    / "VisDrone_SAR_2CLASS"
    / "test_dev"
    / "images"
)

EVALUATION_DIR = EXPERIMENT_DIR / "evaluation"

METRICS_DIR = EVALUATION_DIR / "metrics"
PREDICTIONS_DIR = EVALUATION_DIR / "predictions"
SAMPLES_DIR = EVALUATION_DIR / "samples"
REPORTS_DIR = EVALUATION_DIR / "reports"

REPORT_FILE = REPORTS_DIR / "evaluation_summary.txt"
METRICS_JSON = METRICS_DIR / "metrics.json"
PREDICTIONS_CSV = PREDICTIONS_DIR / "predictions.csv"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("SARC-DRONE-EVALUATION")


# ============================================================
# UTILIDADES
# ============================================================

def ensure_directories():
    for directory in [
        METRICS_DIR,
        PREDICTIONS_DIR,
        SAMPLES_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_paths():

    logger.info("=" * 80)
    logger.info("SARC-DRONE - EVALUACIÓN YOLO26")
    logger.info("=" * 80)

    logger.info("Modelo : %s", MODEL_PATH)
    logger.info("Dataset: %s", DATASET_YAML)
    logger.info("Test   : %s", TEST_IMAGES)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {MODEL_PATH}"
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"No existe el YAML: {DATASET_YAML}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"No existe el directorio de test: {TEST_IMAGES}"
        )

    logger.info("OK: modelo encontrado")
    logger.info("OK: YAML encontrado")
    logger.info("OK: imágenes de test encontradas")


def save_metrics(metrics):

    data = {}

    # Métricas globales
    if hasattr(metrics, "results_dict"):
        data["results"] = {
            str(k): float(v)
            for k, v in metrics.results_dict.items()
            if isinstance(v, (int, float))
        }

    # Métricas por clase
    if hasattr(metrics, "names"):
        data["names"] = metrics.names

    if hasattr(metrics, "box"):

        box = metrics.box

        for attribute in [
            "mp",
            "mr",
            "map50",
            "map",
        ]:
            if hasattr(box, attribute):
                value = getattr(box, attribute)

                try:
                    data[f"box_{attribute}"] = float(value)
                except (TypeError, ValueError):
                    pass

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "OK: métricas guardadas en %s",
        METRICS_JSON,
    )


def save_predictions(results):

    rows = []

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i in range(len(boxes)):

            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            xyxy = boxes.xyxy[i].tolist()

            rows.append(
                {
                    "image": str(result.path),
                    "class_id": cls,
                    "class_name": result.names.get(cls, str(cls)),
                    "confidence": conf,
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                }
            )

    if not rows:
        logger.warning("No se encontraron predicciones.")
        return

    with open(
        PREDICTIONS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "OK: predicciones guardadas: %s",
        PREDICTIONS_CSV,
    )

    logger.info(
        "Predicciones registradas: %d",
        len(rows),
    )


def create_report(metrics, elapsed):

    lines = []

    lines.append("=" * 80)
    lines.append("SARC-DRONE - INFORME DE EVALUACIÓN")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Modelo: {MODEL_PATH}")
    lines.append(f"Dataset YAML: {DATASET_YAML}")
    lines.append(f"Test: {TEST_IMAGES}")
    lines.append("")

    lines.append("TIEMPO")
    lines.append("-" * 80)
    lines.append(f"Evaluación: {elapsed:.2f} segundos")
    lines.append("")

    lines.append("MÉTRICAS")
    lines.append("-" * 80)

    if hasattr(metrics, "results_dict"):

        for key, value in metrics.results_dict.items():

            try:
                lines.append(
                    f"{key}: {float(value):.6f}"
                )
            except (TypeError, ValueError):
                lines.append(
                    f"{key}: {value}"
                )

    lines.append("")
    lines.append("CLASES")
    lines.append("-" * 80)

    if hasattr(metrics, "names"):

        for class_id, name in metrics.names.items():

            lines.append(
                f"{class_id}: {name}"
            )

    lines.append("")
    lines.append("=" * 80)

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("\n".join(lines))

    logger.info(
        "OK: informe generado: %s",
        REPORT_FILE,
    )


# ============================================================
# EVALUACIÓN
# ============================================================

def main():

    ensure_directories()

    validate_paths()

    logger.info("")
    logger.info("=" * 80)
    logger.info("CARGANDO MODELO")
    logger.info("=" * 80)

    model = YOLO(str(MODEL_PATH))

    logger.info("OK: modelo cargado")

    # --------------------------------------------------------
    # VALIDACIÓN FORMAL SOBRE TEST
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUACIÓN DEL DATASET TEST")
    logger.info("=" * 80)

    start = time.time()

    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        amp=True,
        plots=True,
        project=str(METRICS_DIR),
        name="test",
        exist_ok=True,
    )

    elapsed = time.time() - start

    save_metrics(metrics)

    create_report(
        metrics,
        elapsed,
    )

    # --------------------------------------------------------
    # INFERENCIA SOBRE TEST
    # --------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("INFERENCIA SOBRE TEST")
    logger.info("=" * 80)

    results = model.predict(
        source=str(TEST_IMAGES),
        imgsz=640,
        batch=8,
        device=0,
        conf=0.25,
        iou=0.7,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(PREDICTIONS_DIR),
        name="test_predictions",
        exist_ok=True,
        verbose=True,
    )

    save_predictions(results)

    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUACIÓN COMPLETADA")
    logger.info("=" * 80)

    logger.info(
        "Resultados: %s",
        EVALUATION_DIR,
    )


if __name__ == "__main__":
    main()