from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26
# EXP03 - HIGH RESOLUTION SMALL OBJECT EVALUATION V1
# ============================================================================
#
# IMPORTANTE:
# Este script NO entrena.
#
# Utiliza directamente el best.pt que ya fue generado por EXP03.
#
# Objetivo:
# comparar SMALL PERSON Recall de EXP03 contra:
#
#   EXP01 = 29.68%
#   EXP02 = 29.62%
#
# Protocolo:
#   PERSON class = 0
#   SMALL = area < 256 px²
#   test_dev
#   inference imgsz = 1536
#   confidence = 0.25
#   matching IoU = 0.50
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PERSON_CLASS_ID = 0

SMALL_AREA_THRESHOLD = 256.0

EVAL_IMAGE_SIZE = 1536

CONF_THRESHOLD = 0.25

MATCH_IOU_THRESHOLD = 0.50

DEVICE = 0


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================================
# PROJECT
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()

SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root() -> Path:

    for directory in [
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
    ]:

        if directory.name.lower() == "sarc-drone":

            return directory

    raise RuntimeError(
        "No se pudo localizar C:\\SARC-Drone.\n"
        f"Script:\n{SCRIPT_PATH}"
    )


PROJECT_ROOT = find_project_root()


BASELINE_DIR = (
    PROJECT_ROOT
    / "01_training"
    / "experiments"
    / "sar_yolo26"
    / "baseline"
)


# ============================================================================
# DATASET
# ============================================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "00_datasets"
    / "SAR_DATASET_STUDIO"
    / "processed"
    / "sar"
    / "cleaned"
    / "VisDrone_SAR_2CLASS_V1"
)


TEST_IMAGES_DIR = (
    DATASET_ROOT
    / "test_dev"
    / "images"
)


TEST_LABELS_DIR = (
    DATASET_ROOT
    / "test_dev"
    / "labels"
)


# ============================================================================
# EXISTING EXP03 MODEL
# ============================================================================

EXP03_MODEL = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp03_high_resolution_small_object_v1"
    / "runs"
    / "exp03_high_resolution_small_object"
    / "weights"
    / "best.pt"
)


# ============================================================================
# REPORTS
# ============================================================================

REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp03_high_resolution_small_object_v1"
    / "reports"
)


OBJECTS_CSV = (
    REPORTS_DIR
    / "exp03_small_person_objects_v1.csv"
)


SUMMARY_CSV = (
    REPORTS_DIR
    / "exp03_small_person_recall_v1.csv"
)


SIZE_CSV = (
    REPORTS_DIR
    / "exp03_small_person_recall_by_size_v1.csv"
)


COMPARISON_CSV = (
    REPORTS_DIR
    / "exp03_vs_exp01_exp02_small_person_recall_v1.csv"
)


SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP03_HIGH_RESOLUTION_SMALL_OBJECT_EVALUATION_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_div(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:

        return 0.0

    return numerator / denominator


def percentage(
    numerator: float,
    denominator: float,
) -> float:

    return (
        safe_div(
            numerator,
            denominator,
        )
        * 100.0
    )


def iou_xyxy(
    box_a: List[float],
    box_b: List[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection
        / union
    )


def normalized_xywh_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> List[float]:

    cx = xc * image_width
    cy = yc * image_height

    bw = w * image_width
    bh = h * image_height

    x1 = max(
        0.0,
        cx - bw / 2.0,
    )

    y1 = max(
        0.0,
        cy - bh / 2.0,
    )

    x2 = min(
        float(image_width),
        cx + bw / 2.0,
    )

    y2 = min(
        float(image_height),
        cy + bh / 2.0,
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ============================================================================
# VALIDACIÓN
# ============================================================================

def validate_paths() -> None:

    print()
    print("=" * 72)
    print("VALIDANDO EXP03 EVALUATION")
    print("=" * 72)
    print()

    required = {

        "PROJECT_ROOT":
            PROJECT_ROOT,

        "BASELINE_DIR":
            BASELINE_DIR,

        "DATASET_ROOT":
            DATASET_ROOT,

        "TEST_IMAGES_DIR":
            TEST_IMAGES_DIR,

        "TEST_LABELS_DIR":
            TEST_LABELS_DIR,

        "EXP03_MODEL":
            EXP03_MODEL,
    }

    for name, path in required.items():

        if not path.exists():

            raise FileNotFoundError(
                f"No se encontró {name}:\n"
                f"{path}"
            )

        print(
            f"[OK] {name}"
        )

        print(
            f"     {path}"
        )


# ============================================================================
# GROUND TRUTH
# ============================================================================

def load_small_person_gt(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Dict]:

    objects = []

    if not label_path.exists():

        return objects

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except UnicodeDecodeError:

        lines = label_path.read_text(
            encoding="latin-1"
        ).splitlines()

    for gt_index, line in enumerate(
        lines
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:

            class_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:

            continue

        if class_id != PERSON_CLASS_ID:
            continue

        if w <= 0 or h <= 0:
            continue

        area = (
            w
            * h
            * image_width
            * image_height
        )

        if area >= SMALL_AREA_THRESHOLD:
            continue

        box = normalized_xywh_to_xyxy(
            xc,
            yc,
            w,
            h,
            image_width,
            image_height,
        )

        objects.append(
            {
                "gt_index": gt_index,
                "box": box,
                "area": area,
                "size_sqrt": math.sqrt(
                    max(
                        area,
                        0.0,
                    )
                ),
            }
        )

    return objects


# ============================================================================
# MATCHING
# ============================================================================

def match_predictions(
    gt_objects: List[Dict],
    prediction_boxes: List[List[float]],
    prediction_confidences: List[float],
) -> List[Dict]:

    results = []

    used_predictions = set()

    for gt in gt_objects:

        best_iou = 0.0

        best_index = None

        for prediction_index, prediction_box in enumerate(
            prediction_boxes
        ):

            if prediction_index in used_predictions:
                continue

            current_iou = iou_xyxy(
                gt["box"],
                prediction_box,
            )

            if current_iou > best_iou:

                best_iou = current_iou

                best_index = prediction_index

        matched = (
            best_index is not None
            and
            best_iou >= MATCH_IOU_THRESHOLD
        )

        confidence = 0.0

        if matched:

            used_predictions.add(
                best_index
            )

            confidence = (
                prediction_confidences[
                    best_index
                ]
            )

        results.append(
            {
                "matched":
                    matched,

                "iou":
                    best_iou,

                "confidence":
                    confidence,
            }
        )

    return results


# ============================================================================
# SIZE BUCKET
# ============================================================================

def get_size_bucket(
    size_sqrt: float,
) -> str:

    if size_sqrt < 16:
        return "<16"

    if size_sqrt < 32:
        return "16-32"

    if size_sqrt < 64:
        return "32-64"

    if size_sqrt < 128:
        return "64-128"

    if size_sqrt < 256:
        return "128-256"

    return ">=256"


# ============================================================================
# PROCESS IMAGE
# ============================================================================

def process_image(
    model: YOLO,
    image_path: Path,
) -> List[Dict]:

    label_path = (
        TEST_LABELS_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():

        return []

    try:

        with Image.open(
            image_path
        ) as image:

            image_width = image.width
            image_height = image.height

    except Exception as exc:

        print(
            f"[WARNING] No se pudo abrir "
            f"{image_path.name}: {exc}"
        )

        return []

    gt_objects = load_small_person_gt(
        label_path,
        image_width,
        image_height,
    )

    if not gt_objects:

        return []

    try:

        results = model.predict(
            source=str(image_path),
            imgsz=EVAL_IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
            save=False,
        )

    except Exception as exc:

        print(
            f"[ERROR] Inferencia fallida en "
            f"{image_path.name}: {exc}"
        )

        return []

    if not results:

        return []

    result = results[0]

    prediction_boxes = []

    prediction_confidences = []

    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .tolist()
        )

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            classes,
        ):

            if int(class_id) != PERSON_CLASS_ID:
                continue

            prediction_boxes.append(box)

            prediction_confidences.append(
                float(confidence)
            )

    matches = match_predictions(
        gt_objects,
        prediction_boxes,
        prediction_confidences,
    )

    rows = []

    for gt, match in zip(
        gt_objects,
        matches,
    ):

        rows.append(
            {
                "image":
                    image_path.name,

                "gt_index":
                    gt["gt_index"],

                "area":
                    round(
                        gt["area"],
                        6,
                    ),

                "size_sqrt":
                    round(
                        gt["size_sqrt"],
                        6,
                    ),

                "size_bucket":
                    get_size_bucket(
                        gt["size_sqrt"]
                    ),

                "status":
                    (
                        "TP"
                        if match["matched"]
                        else "FN"
                    ),

                "iou":
                    round(
                        match["iou"],
                        6,
                    ),

                "confidence":
                    round(
                        match["confidence"],
                        6,
                    ),
            }
        )

    return rows


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    rows: List[Dict],
) -> Dict:

    gt = len(rows)

    tp = sum(
        1
        for row in rows
        if row["status"] == "TP"
    )

    fn = gt - tp

    return {
        "gt": gt,
        "tp": tp,
        "fn": fn,
        "recall": safe_div(
            tp,
            gt,
        ),
        "recall_percentage": percentage(
            tp,
            gt,
        ),
    }


# ============================================================================
# SIZE
# ============================================================================

def calculate_size_metrics(
    rows: List[Dict],
) -> List[Dict]:

    groups = defaultdict(list)

    for row in rows:

        groups[
            row["size_bucket"]
        ].append(row)

    buckets = [
        "<16",
        "16-32",
        "32-64",
        "64-128",
        "128-256",
        ">=256",
    ]

    output = []

    for bucket in buckets:

        metrics = calculate_metrics(
            groups.get(
                bucket,
                [],
            )
        )

        output.append(
            {
                "size_bucket":
                    bucket,

                **metrics,
            }
        )

    return output


# ============================================================================
# WRITE CSV
# ============================================================================

def write_csv(
    path: Path,
    rows: List[Dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:

        path.write_text(
            "",
            encoding="utf-8",
        )

        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================================
# COMPARISON
# ============================================================================

def write_comparison(
    exp03: Dict,
) -> None:

    exp01_recall = 29.68
    exp02_recall = 29.62

    rows = [

        {
            "experiment":
                "EXP01",

            "train_imgsz":
                640,

            "small_person_gt":
                17879,

            "small_person_tp":
                5306,

            "small_person_fn":
                12573,

            "small_person_recall_percentage":
                exp01_recall,

            "delta_vs_exp01_pp":
                0.0,
        },

        {
            "experiment":
                "EXP02",

            "train_imgsz":
                640,

            "small_person_gt":
                17879,

            "small_person_tp":
                5295,

            "small_person_fn":
                12584,

            "small_person_recall_percentage":
                exp02_recall,

            "delta_vs_exp01_pp":
                exp02_recall - exp01_recall,
        },

        {
            "experiment":
                "EXP03",

            "train_imgsz":
                960,

            "small_person_gt":
                exp03["gt"],

            "small_person_tp":
                exp03["tp"],

            "small_person_fn":
                exp03["fn"],

            "small_person_recall_percentage":
                exp03[
                    "recall_percentage"
                ],

            "delta_vs_exp01_pp":
                exp03[
                    "recall_percentage"
                ]
                -
                exp01_recall,
        },
    ]

    write_csv(
        COMPARISON_CSV,
        rows,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def write_summary(
    metrics: Dict,
    size_metrics: List[Dict],
) -> None:

    exp01_recall = 29.68
    exp02_recall = 29.62

    delta_exp01 = (
        metrics["recall_percentage"]
        -
        exp01_recall
    )

    delta_exp02 = (
        metrics["recall_percentage"]
        -
        exp02_recall
    )

    lines = []

    lines.append("=" * 72)

    lines.append(
        "SAR YOLO26 - EXP03 HIGH RESOLUTION SMALL OBJECT EVALUATION V1"
    )

    lines.append("=" * 72)

    lines.append("")

    lines.append(
        "MODELO"
    )

    lines.append(
        str(EXP03_MODEL)
    )

    lines.append("")

    lines.append(
        "DATASET"
    )

    lines.append(
        str(DATASET_ROOT)
    )

    lines.append("")

    lines.append(
        "PROTOCOLO"
    )

    lines.append(
        "Split: test_dev"
    )

    lines.append(
        "PERSON class: 0"
    )

    lines.append(
        "Small: area < 256 px²"
    )

    lines.append(
        "Inference imgsz: 1536"
    )

    lines.append(
        "Confidence: 0.25"
    )

    lines.append(
        "Match IoU: 0.50"
    )

    lines.append("")

    lines.append(
        "RESULTADO EXP03"
    )

    lines.append("-" * 72)

    lines.append(
        f"SMALL PERSON GT:      "
        f"{metrics['gt']:,}"
    )

    lines.append(
        f"SMALL PERSON TP:      "
        f"{metrics['tp']:,}"
    )

    lines.append(
        f"SMALL PERSON FN:      "
        f"{metrics['fn']:,}"
    )

    lines.append(
        f"SMALL PERSON Recall:  "
        f"{metrics['recall_percentage']:.2f}%"
    )

    lines.append("")

    lines.append(
        "COMPARACIÓN"
    )

    lines.append("-" * 72)

    lines.append(
        f"EXP01 (640):          "
        f"{exp01_recall:.2f}%"
    )

    lines.append(
        f"EXP02 (640 + 2x):     "
        f"{exp02_recall:.2f}%"
    )

    lines.append(
        f"EXP03 (960):          "
        f"{metrics['recall_percentage']:.2f}%"
    )

    lines.append(
        f"EXP03 - EXP01:        "
        f"{delta_exp01:+.2f} pp"
    )

    lines.append(
        f"EXP03 - EXP02:        "
        f"{delta_exp02:+.2f} pp"
    )

    lines.append("")

    if delta_exp01 > 2.0:

        interpretation = (
            "MEJORA FUERTE: aumentar la resolución "
            "muestra una mejora clara en SMALL PERSON Recall."
        )

    elif delta_exp01 > 0.5:

        interpretation = (
            "MEJORA MODERADA: aumentar la resolución "
            "muestra una señal positiva."
        )

    elif delta_exp01 >= -0.5:

        interpretation = (
            "SIN CAMBIO RELEVANTE: el aumento "
            "de 640 a 960 no cambia apreciablemente "
            "el SMALL PERSON Recall."
        )

    else:

        interpretation = (
            "EMPEORAMIENTO: la configuración de 960 "
            "reduce el SMALL PERSON Recall."
        )

    lines.append(
        "INTERPRETACIÓN"
    )

    lines.append("-" * 72)

    lines.append(
        interpretation
    )

    lines.append("")

    lines.append(
        "DESGLOSE POR TAMAÑO"
    )

    lines.append("-" * 72)

    for row in size_metrics:

        lines.append(
            f"{row['size_bucket']:>8} "
            f"GT={row['gt']:>6,} "
            f"TP={row['tp']:>6,} "
            f"FN={row['fn']:>6,} "
            f"Recall={row['recall_percentage']:>7.2f}%"
        )

    lines.append("")

    lines.append(
        "DECISIÓN EXPERIMENTAL"
    )

    lines.append("-" * 72)

    if delta_exp01 > 0.5:

        lines.append(
            "EXP03 aporta evidencia a favor "
            "de aumentar la resolución de entrenamiento."
        )

        lines.append(
            "Debe considerarse como intervención "
            "válida para la siguiente fase."
        )

    else:

        lines.append(
            "EXP03 no demuestra una mejora suficiente "
            "por sí sola."
        )

        lines.append(
            "Debe analizarse el failure residual "
            "y pasar a la siguiente hipótesis."
        )

    lines.append("")

    lines.append(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    lines.append(
        "IMPORTANTE: el YAML oficial NO ha sido modificado."
    )

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
        "# SAR YOLO26 - EXP03 HIGH RESOLUTION SMALL OBJECT EVALUATION V1"
    )

    print("=" * 72)

    print()
    print(
        "SCRIPT:"
    )

    print(
        f"  {SCRIPT_PATH}"
    )

    print()
    print(
        "EXP03 MODEL:"
    )

    print(
        f"  {EXP03_MODEL}"
    )

    print()
    print(
        "DATASET:"
    )

    print(
        f"  {DATASET_ROOT}"
    )

    # ------------------------------------------------------------------------
    # VALIDACIÓN
    # ------------------------------------------------------------------------

    validate_paths()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("CARGANDO BEST.PT EXP03")
    print("=" * 72)
    print()

    model = YOLO(
        str(EXP03_MODEL)
    )

    print(
        "[OK] Modelo cargado."
    )

    print(
        f"[OK] Clases: {model.names}"
    )

    # ------------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------------

    image_files = sorted(
        [
            path
            for path in TEST_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:

        raise RuntimeError(
            "No se encontraron imágenes TEST_DEV."
        )

    print()
    print(
        f"[OK] Imágenes TEST_DEV: "
        f"{len(image_files):,}"
    )

    # ------------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("ANALIZANDO SMALL PERSON EXP03")
    print("=" * 72)
    print()

    all_rows = []

    images_with_small_person = 0

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        rows = process_image(
            model,
            image_path,
        )

        if rows:

            images_with_small_person += 1

            all_rows.extend(
                rows
            )

        if (
            index % 100 == 0
            or index == len(image_files)
        ):

            print(
                f"Analizadas: "
                f"{index:,}/{len(image_files):,} "
                f"| Small GT: "
                f"{len(all_rows):,}"
            )

    if not all_rows:

        raise RuntimeError(
            "No se encontraron SMALL PERSON."
        )

    # ------------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------------

    metrics = calculate_metrics(
        all_rows
    )

    size_metrics = (
        calculate_size_metrics(
            all_rows
        )
    )

    # ------------------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------------------

    write_csv(
        OBJECTS_CSV,
        all_rows,
    )

    write_csv(
        SIZE_CSV,
        size_metrics,
    )

    write_csv(
        SUMMARY_CSV,
        [
            {
                "metric":
                    "SMALL_PERSON_GT",

                "value":
                    metrics["gt"],
            },

            {
                "metric":
                    "SMALL_PERSON_TP",

                "value":
                    metrics["tp"],
            },

            {
                "metric":
                    "SMALL_PERSON_FN",

                "value":
                    metrics["fn"],
            },

            {
                "metric":
                    "SMALL_PERSON_RECALL",

                "value":
                    metrics["recall"],
            },

            {
                "metric":
                    "SMALL_PERSON_RECALL_PERCENTAGE",

                "value":
                    metrics[
                        "recall_percentage"
                    ],
            },

            {
                "metric":
                    "TRAIN_IMAGE_SIZE",

                "value":
                    960,
            },

            {
                "metric":
                    "EVAL_IMAGE_SIZE",

                "value":
                    EVAL_IMAGE_SIZE,
            },

            {
                "metric":
                    "CONF_THRESHOLD",

                "value":
                    CONF_THRESHOLD,
            },

            {
                "metric":
                    "MATCH_IOU_THRESHOLD",

                "value":
                    MATCH_IOU_THRESHOLD,
            },
        ],
    )

    write_comparison(
        metrics
    )

    write_summary(
        metrics,
        size_metrics,
    )

    # ------------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------------

    delta_exp01 = (
        metrics["recall_percentage"]
        - 29.68
    )

    delta_exp02 = (
        metrics["recall_percentage"]
        - 29.62
    )

    print()
    print("=" * 72)

    print(
        "# RESULTADO EXP03 SMALL PERSON RECALL"
    )

    print("=" * 72)

    print()

    print(
        f"Imágenes con small person: "
        f"{images_with_small_person:,}"
    )

    print(
        f"SMALL PERSON GT: "
        f"{metrics['gt']:,}"
    )

    print(
        f"SMALL PERSON TP: "
        f"{metrics['tp']:,}"
    )

    print(
        f"SMALL PERSON FN: "
        f"{metrics['fn']:,}"
    )

    print(
        f"SMALL PERSON Recall: "
        f"{metrics['recall_percentage']:.2f}%"
    )

    print()

    print(
        "COMPARACIÓN"
    )

    print(
        f"EXP01: 29.68%"
    )

    print(
        f"EXP02: 29.62%"
    )

    print(
        f"EXP03: "
        f"{metrics['recall_percentage']:.2f}%"
    )

    print(
        f"EXP03 - EXP01: "
        f"{delta_exp01:+.2f} pp"
    )

    print(
        f"EXP03 - EXP02: "
        f"{delta_exp02:+.2f} pp"
    )

    print()

    print(
        "REPORTS"
    )

    print(
        f"[OK] {OBJECTS_CSV}"
    )

    print(
        f"[OK] {SUMMARY_CSV}"
    )

    print(
        f"[OK] {SIZE_CSV}"
    )

    print(
        f"[OK] {COMPARISON_CSV}"
    )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    print()

    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    print(
        "IMPORTANTE: el YAML oficial NO ha sido modificado."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "[CANCELADO] Evaluación EXP03 interrumpida."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)

        print(
            "[ERROR EXP03 EVALUATION]"
        )

        print("=" * 72)

        print()

        print(
            str(exc)
        )

        print()

        sys.exit(1)