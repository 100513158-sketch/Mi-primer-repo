from pathlib import Path
import csv
import json
import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

EXP_DIR = Path(__file__).resolve().parent

EVAL_DIR = EXP_DIR / "evaluation"

PREDICTIONS_CSV = EVAL_DIR / "predictions" / "predictions.csv"

METRICS_JSON = EVAL_DIR / "metrics" / "metrics.json"

OUTPUT_DIR = EVAL_DIR / "analysis"

REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"
DATA_DIR = OUTPUT_DIR / "data"

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def class_name(class_id):
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


def ensure_directories():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LECTURA DE PREDICCIONES
# ============================================================

def load_predictions():
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"No existe predictions.csv:\n{PREDICTIONS_CSV}"
        )

    with PREDICTIONS_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    return rows


# ============================================================
# DETECCIÓN DE COLUMNAS
# ============================================================

def find_column(fieldnames, candidates):
    """
    Busca una columna aceptando distintas variantes de nombre.
    """

    normalized = {
        str(name).strip().lower(): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    return None


def detect_columns(rows):

    if not rows:
        raise RuntimeError("El CSV de predicciones está vacío.")

    fields = list(rows[0].keys())

    columns = {}

    columns["image"] = find_column(
        fields,
        [
            "image",
            "image_path",
            "filename",
            "file",
            "path",
        ],
    )

    columns["class_id"] = find_column(
        fields,
        [
            "class",
            "class_id",
            "cls",
            "category",
        ],
    )

    columns["confidence"] = find_column(
        fields,
        [
            "confidence",
            "conf",
            "score",
        ],
    )

    columns["x1"] = find_column(fields, ["x1", "xmin"])
    columns["y1"] = find_column(fields, ["y1", "ymin"])
    columns["x2"] = find_column(fields, ["x2", "xmax"])
    columns["y2"] = find_column(fields, ["y2", "ymax"])

    print("\nCOLUMNAS DETECTADAS")
    print("-" * 60)

    for key, value in columns.items():
        print(f"{key:15}: {value}")

    required = [
        "image",
        "class_id",
        "confidence",
    ]

    missing = [
        x for x in required
        if columns[x] is None
    ]

    if missing:
        raise RuntimeError(
            "No se pudieron identificar las columnas necesarias: "
            + ", ".join(missing)
        )

    return columns


# ============================================================
# ESTADÍSTICAS GENERALES
# ============================================================

def calculate_statistics(rows, columns):

    class_counter = Counter()
    image_counter = Counter()

    confidence_by_class = defaultdict(list)
    area_by_class = defaultdict(list)

    image_detections = Counter()
    image_person_detections = Counter()

    widths = []
    heights = []

    for row in rows:

        image = str(row[columns["image"]])

        class_id = safe_int(
            row[columns["class_id"]]
        )

        name = class_name(class_id)

        conf = safe_float(
            row[columns["confidence"]]
        )

        class_counter[name] += 1
        image_counter[image] += 1

        confidence_by_class[name].append(conf)

        image_detections[image] += 1

        if name == "person":
            image_person_detections[image] += 1

        # ----------------------------------------------------
        # Tamaño de bounding box
        # ----------------------------------------------------

        if all(
            columns[x] is not None
            for x in ["x1", "y1", "x2", "y2"]
        ):

            x1 = safe_float(row[columns["x1"]])
            y1 = safe_float(row[columns["y1"]])
            x2 = safe_float(row[columns["x2"]])
            y2 = safe_float(row[columns["y2"]])

            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            area = width * height

            widths.append(width)
            heights.append(height)

            area_by_class[name].append(area)

    return {
        "class_counter": class_counter,
        "image_counter": image_counter,
        "confidence_by_class": confidence_by_class,
        "area_by_class": area_by_class,
        "image_detections": image_detections,
        "image_person_detections": image_person_detections,
        "widths": widths,
        "heights": heights,
    }


# ============================================================
# RESUMEN ESTADÍSTICO
# ============================================================

def mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return (
        values[f] * (c - k)
        + values[c] * (k - f)
    )


def build_summary(stats):

    summary = {}

    summary["total_predictions"] = sum(
        stats["class_counter"].values()
    )

    summary["classes"] = {}

    for cls, count in stats["class_counter"].items():

        confs = stats["confidence_by_class"][cls]
        areas = stats["area_by_class"].get(cls, [])

        summary["classes"][cls] = {
            "predictions": count,
            "mean_confidence": mean(confs),
            "p10_confidence": percentile(confs, 10),
            "p50_confidence": percentile(confs, 50),
            "p90_confidence": percentile(confs, 90),
            "mean_bbox_area": mean(areas),
            "p10_bbox_area": percentile(areas, 10),
            "p50_bbox_area": percentile(areas, 50),
            "p90_bbox_area": percentile(areas, 90),
        }

    return summary


# ============================================================
# GUARDAR JSON
# ============================================================

def save_json(data, path):

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# INFORME
# ============================================================

def generate_report(summary, stats):

    report = []

    report.append("=" * 80)
    report.append("SARC-DRONE - ERROR ANALYSIS")
    report.append("=" * 80)
    report.append("")

    report.append(
        f"Predicciones analizadas: "
        f"{summary['total_predictions']:,}"
    )

    report.append("")

    report.append("ESTADÍSTICAS POR CLASE")
    report.append("-" * 80)

    for cls, values in summary["classes"].items():

        report.append("")
        report.append(f"CLASE: {cls}")

        report.append(
            f"  Predicciones       : "
            f"{values['predictions']:,}"
        )

        report.append(
            f"  Confianza media    : "
            f"{values['mean_confidence']:.4f}"
        )

        report.append(
            f"  Confianza P10      : "
            f"{values['p10_confidence']:.4f}"
        )

        report.append(
            f"  Confianza P50      : "
            f"{values['p50_confidence']:.4f}"
        )

        report.append(
            f"  Confianza P90      : "
            f"{values['p90_confidence']:.4f}"
        )

        report.append(
            f"  Área bbox media    : "
            f"{values['mean_bbox_area']:.2f}"
        )

        report.append(
            f"  Área bbox P10      : "
            f"{values['p10_bbox_area']:.2f}"
        )

        report.append(
            f"  Área bbox P50      : "
            f"{values['p50_bbox_area']:.2f}"
        )

        report.append(
            f"  Área bbox P90      : "
            f"{values['p90_bbox_area']:.2f}"
        )

    report.append("")
    report.append("IMÁGENES CON MAYOR NÚMERO DE DETECCIONES")
    report.append("-" * 80)

    for image, count in (
        stats["image_detections"]
        .most_common(20)
    ):

        report.append(
            f"{count:4d} detecciones | {image}"
        )

    report.append("")
    report.append(
        "IMÁGENES CON MAYOR NÚMERO DE PERSONAS DETECTADAS"
    )

    report.append("-" * 80)

    for image, count in (
        stats["image_person_detections"]
        .most_common(20)
    ):

        report.append(
            f"{count:4d} personas | {image}"
        )

    report.append("")
    report.append("=" * 80)

    report_path = (
        REPORTS_DIR
        / "error_analysis_summary.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# GRÁFICAS
# ============================================================

def plot_confidence_distribution(stats):

    classes = list(
        stats["confidence_by_class"].keys()
    )

    if not classes:
        return

    plt.figure(figsize=(10, 6))

    for cls in classes:

        values = (
            stats["confidence_by_class"][cls]
        )

        plt.hist(
            values,
            bins=20,
            alpha=0.5,
            label=cls,
        )

    plt.xlabel("Confidence")
    plt.ylabel("Number of detections")
    plt.title("Distribution of detection confidence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "confidence_distribution.png",
        dpi=150,
    )

    plt.close()


def plot_bbox_area_distribution(stats):

    classes = list(
        stats["area_by_class"].keys()
    )

    if not classes:
        return

    plt.figure(figsize=(10, 6))

    data = [
        stats["area_by_class"][cls]
        for cls in classes
        if stats["area_by_class"][cls]
    ]

    labels = [
        cls
        for cls in classes
        if stats["area_by_class"][cls]
    ]

    plt.boxplot(
        data,
        tick_labels=labels,
    )

    plt.yscale("log")

    plt.ylabel("Bounding box area (log scale)")
    plt.xlabel("Class")
    plt.title("Bounding box area by class")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "bbox_area_by_class.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SARC-DRONE - ERROR ANALYSIS")
    print("=" * 80)

    ensure_directories()

    print("\nCargando predicciones...")
    print(PREDICTIONS_CSV)

    rows = load_predictions()

    print(
        f"Predicciones cargadas: {len(rows):,}"
    )

    columns = detect_columns(rows)

    print("\nCalculando estadísticas...")

    stats = calculate_statistics(
        rows,
        columns,
    )

    summary = build_summary(stats)

    json_path = (
        DATA_DIR
        / "error_analysis.json"
    )

    save_json(
        summary,
        json_path,
    )

    report_path = generate_report(
        summary,
        stats,
    )

    plot_confidence_distribution(
        stats
    )

    plot_bbox_area_distribution(
        stats
    )

    print("")
    print("=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)

    print(
        f"\nInforme:"
        f"\n{report_path}"
    )

    print(
        f"\nJSON:"
        f"\n{json_path}"
    )

    print(
        f"\nGráficas:"
        f"\n{PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()