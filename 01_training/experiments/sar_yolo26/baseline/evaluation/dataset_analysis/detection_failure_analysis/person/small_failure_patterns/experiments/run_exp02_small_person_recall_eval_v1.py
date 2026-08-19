from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26
# EXP02 - SMALL PERSON RECALL EVALUATION V1
# ============================================================================
#
# OBJETIVO
# --------
# Evaluar el best.pt producido por EXP02 utilizando EXACTAMENTE el mismo
# protocolo de EXP01 para SMALL PERSON.
#
# COMPARACIÓN:
#
# EXP01:
#   baseline_v1/weights/best.pt
#
# EXP02:
#   exp02_targeted_small_person_oversampling_v1/
#   runs/exp02_small_person_oversampling/weights/best.pt
#
# PROTOCOLO:
#   Dataset  : VisDrone_SAR_2CLASS_V1
#   Split    : test_dev
#   Class    : 0 = person
#   Small    : area < 256 px²
#   ImageSize: 1536
#   Conf     : 0.25
#   Match IoU: 0.50
#
# IMPORTANTE
# ----------
# - NO modifica el dataset.
# - NO modifica labels.
# - NO modifica imágenes.
# - NO modifica YAML oficial.
# - Solo realiza inferencia y genera reports.
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PERSON_CLASS_ID = 0

SMALL_AREA_THRESHOLD = 256.0

IMAGE_SIZE = 1536

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
# PROYECTO
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root() -> Path:
    """
    Localiza C:\\SARC-Drone.
    """

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
# MODELOS
# ============================================================================

EXP01_MODEL = (
    BASELINE_DIR
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)


EXP02_MODEL = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp02_targeted_small_person_oversampling_v1"
    / "runs"
    / "exp02_small_person_oversampling"
    / "weights"
    / "best.pt"
)


# ============================================================================
# OUTPUT EXP02
# ============================================================================

OUTPUT_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp02_targeted_small_person_oversampling_v1"
    / "reports"
)


OBJECTS_CSV = (
    OUTPUT_DIR
    / "exp02_small_person_recall_objects_v1.csv"
)


SUMMARY_CSV = (
    OUTPUT_DIR
    / "exp02_small_person_recall_summary_v1.csv"
)


SIZE_CSV = (
    OUTPUT_DIR
    / "exp02_small_person_recall_by_size_v1.csv"
)


IMAGE_CSV = (
    OUTPUT_DIR
    / "exp02_small_person_recall_by_image_v1.csv"
)


COMPARISON_CSV = (
    OUTPUT_DIR
    / "exp02_vs_exp01_small_person_recall_v1.csv"
)


SUMMARY_TXT = (
    OUTPUT_DIR
    / "EXP02_SMALL_PERSON_RECALL_EVALUATION_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (TypeError, ValueError):

        return default


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

    return intersection / union


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
    print("VALIDANDO ESTRUCTURA EXP02 EVALUATION")
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

        "EXP01_MODEL":
            EXP01_MODEL,

        "EXP02_MODEL":
            EXP02_MODEL,
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

    for gt_index, line in enumerate(lines):

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

            xc = float(
                parts[1]
            )

            yc = float(
                parts[2]
            )

            w = float(
                parts[3]
            )

            h = float(
                parts[4]
            )

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
# MATCHING GT -> PREDICTIONS
# ============================================================================

def match_gt_to_predictions(
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
                best_index = (
                    prediction_index
                )

        matched = (
            best_index is not None
            and best_iou >= MATCH_IOU_THRESHOLD
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
#
# IMPORTANTE:
# Aquí mantenemos el mismo criterio de EXP01:
# se trata de la raíz cuadrada del área.
#
# Para esta comparación, el dato principal será siempre el recall global
# SMALL PERSON. Los buckets solo sirven como desglose auxiliar.
#
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
# PROCESAR IMAGEN
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

            source=str(
                image_path
            ),

            imgsz=IMAGE_SIZE,

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

            prediction_boxes.append(
                box
            )

            prediction_confidences.append(
                float(confidence)
            )

    matches = match_gt_to_predictions(
        gt_objects,
        prediction_boxes,
        prediction_confidences,
    )

    output = []

    for gt, match in zip(
        gt_objects,
        matches,
    ):

        output.append(
            {
                "image":
                    image_path.name,

                "image_width":
                    image_width,

                "image_height":
                    image_height,

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

                "matched":
                    int(
                        match["matched"]
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

    return output


# ============================================================================
# MÉTRICAS
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

    fn = (
        gt
        - tp
    )

    return {
        "gt":
            gt,

        "tp":
            tp,

        "fn":
            fn,

        "recall":
            safe_div(
                tp,
                gt,
            ),

        "recall_percentage":
            percentage(
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
        ].append(
            row
        )

    ordered_buckets = [
        "<16",
        "16-32",
        "32-64",
        "64-128",
        "128-256",
        ">=256",
    ]

    output = []

    for bucket in ordered_buckets:

        group = groups.get(
            bucket,
            [],
        )

        metrics = calculate_metrics(
            group
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
# IMAGE
# ============================================================================

def calculate_image_metrics(
    rows: List[Dict],
) -> List[Dict]:

    groups = defaultdict(list)

    for row in rows:

        groups[
            row["image"]
        ].append(
            row
        )

    output = []

    for image, group in sorted(
        groups.items()
    ):

        metrics = calculate_metrics(
            group
        )

        output.append(
            {
                "image":
                    image,

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

        writer.writerows(
            rows
        )


# ============================================================================
# COMPARACIÓN EXP01 / EXP02
# ============================================================================

def write_comparison(
    exp01_metrics: Dict,
    exp02_metrics: Dict,
) -> None:

    delta_tp = (
        exp02_metrics["tp"]
        - exp01_metrics["tp"]
    )

    delta_fn = (
        exp02_metrics["fn"]
        - exp01_metrics["fn"]
    )

    delta_recall_pp = (
        exp02_metrics[
            "recall_percentage"
        ]
        -
        exp01_metrics[
            "recall_percentage"
        ]
    )

    rows = [

        {
            "metric":
                "SMALL_PERSON_GT",

            "EXP01":
                exp01_metrics["gt"],

            "EXP02":
                exp02_metrics["gt"],

            "delta":
                exp02_metrics["gt"]
                -
                exp01_metrics["gt"],
        },

        {
            "metric":
                "SMALL_PERSON_TP",

            "EXP01":
                exp01_metrics["tp"],

            "EXP02":
                exp02_metrics["tp"],

            "delta":
                delta_tp,
        },

        {
            "metric":
                "SMALL_PERSON_FN",

            "EXP01":
                exp01_metrics["fn"],

            "EXP02":
                exp02_metrics["fn"],

            "delta":
                delta_fn,
        },

        {
            "metric":
                "SMALL_PERSON_RECALL",

            "EXP01":
                exp01_metrics["recall"],

            "EXP02":
                exp02_metrics["recall"],

            "delta":
                (
                    exp02_metrics["recall"]
                    -
                    exp01_metrics["recall"]
                ),
        },

        {
            "metric":
                "SMALL_PERSON_RECALL_PERCENTAGE",

            "EXP01":
                exp01_metrics[
                    "recall_percentage"
                ],

            "EXP02":
                exp02_metrics[
                    "recall_percentage"
                ],

            "delta":
                delta_recall_pp,
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
    exp02_metrics: Dict,
    exp02_size_metrics: List[Dict],
    exp01_reference: Dict,
) -> None:

    delta_tp = (
        exp02_metrics["tp"]
        -
        exp01_reference["tp"]
    )

    delta_fn = (
        exp02_metrics["fn"]
        -
        exp01_reference["fn"]
    )

    delta_recall_pp = (
        exp02_metrics[
            "recall_percentage"
        ]
        -
        exp01_reference[
            "recall_percentage"
        ]
    )

    lines = []

    lines.append("=" * 72)

    lines.append(
        "SAR YOLO26 - EXP02 SMALL PERSON RECALL EVALUATION V1"
    )

    lines.append("=" * 72)

    lines.append("")

    lines.append(
        "MODELO EXP02"
    )

    lines.append(
        str(EXP02_MODEL)
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
        f"Split: test_dev"
    )

    lines.append(
        f"PERSON class: {PERSON_CLASS_ID}"
    )

    lines.append(
        f"Small area: < {SMALL_AREA_THRESHOLD} px²"
    )

    lines.append(
        f"Input size: {IMAGE_SIZE}"
    )

    lines.append(
        f"Confidence: {CONF_THRESHOLD}"
    )

    lines.append(
        f"Match IoU: {MATCH_IOU_THRESHOLD}"
    )

    lines.append("")

    lines.append(
        "RESULTADO EXP02"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"SMALL PERSON GT:     "
        f"{exp02_metrics['gt']:,}"
    )

    lines.append(
        f"SMALL PERSON TP:     "
        f"{exp02_metrics['tp']:,}"
    )

    lines.append(
        f"SMALL PERSON FN:     "
        f"{exp02_metrics['fn']:,}"
    )

    lines.append(
        f"SMALL PERSON Recall: "
        f"{exp02_metrics['recall_percentage']:.2f}%"
    )

    lines.append("")

    lines.append(
        "COMPARACIÓN CON EXP01"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"EXP01 TP:             "
        f"{exp01_reference['tp']:,}"
    )

    lines.append(
        f"EXP02 TP:             "
        f"{exp02_metrics['tp']:,}"
    )

    lines.append(
        f"Delta TP:             "
        f"{delta_tp:+,}"
    )

    lines.append("")

    lines.append(
        f"EXP01 FN:             "
        f"{exp01_reference['fn']:,}"
    )

    lines.append(
        f"EXP02 FN:             "
        f"{exp02_metrics['fn']:,}"
    )

    lines.append(
        f"Delta FN:             "
        f"{delta_fn:+,}"
    )

    lines.append("")

    lines.append(
        f"EXP01 Recall:         "
        f"{exp01_reference['recall_percentage']:.2f}%"
    )

    lines.append(
        f"EXP02 Recall:         "
        f"{exp02_metrics['recall_percentage']:.2f}%"
    )

    lines.append(
        f"Delta Recall:         "
        f"{delta_recall_pp:+.2f} pp"
    )

    lines.append("")

    if delta_recall_pp > 1.0:

        interpretation = (
            "MEJORA CLARA: el oversampling parece "
            "mejorar el recall de SMALL PERSON."
        )

    elif delta_recall_pp > 0.25:

        interpretation = (
            "MEJORA LEVE: existe una mejora, "
            "pero debe comprobarse si es suficiente."
        )

    elif delta_recall_pp >= -0.25:

        interpretation = (
            "SIN CAMBIO RELEVANTE: el oversampling "
            "no produce una variación apreciable."
        )

    else:

        interpretation = (
            "EMPEORAMIENTO: el oversampling reduce "
            "el recall de SMALL PERSON."
        )

    lines.append(
        "INTERPRETACIÓN"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        interpretation
    )

    lines.append("")

    lines.append(
        "DESGLOSE POR TAMAÑO"
    )

    lines.append(
        "-" * 72
    )

    for row in exp02_size_metrics:

        lines.append(
            f"{row['size_bucket']:>8} "
            f"GT={row['gt']:>6,} "
            f"TP={row['tp']:>6,} "
            f"FN={row['fn']:>6,} "
            f"Recall={row['recall_percentage']:>7.2f}%"
        )

    lines.append("")

    lines.append(
        "SIGUIENTE DECISIÓN"
    )

    lines.append(
        "-" * 72
    )

    if delta_recall_pp > 0.25:

        lines.append(
            "EXP02 demuestra potencial."
        )

        lines.append(
            "Debe compararse también por tamaño y "
            "métricas globales antes de adoptar "
            "oversampling como intervención."
        )

    else:

        lines.append(
            "EXP02 no demuestra una mejora suficiente "
            "para justificar por sí solo la intervención."
        )

        lines.append(
            "El siguiente experimento debe centrarse "
            "en otro mecanismo del fallo."
        )

    lines.append("")

    lines.append(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    lines.append(
        "IMPORTANTE: el YAML oficial NO ha sido modificado."
    )

    lines.append("")

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
        "# SAR YOLO26 - EXP02 SMALL PERSON RECALL EVALUATION V1"
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
        "EXP02 MODEL:"
    )

    print(
        f"  {EXP02_MODEL}"
    )

    print()
    print(
        "DATASET:"
    )

    print(
        f"  {DATASET_ROOT}"
    )

    print()
    print(
        "TEST:"
    )

    print(
        f"  {TEST_IMAGES_DIR}"
    )

    print(
        f"  {TEST_LABELS_DIR}"
    )

    # ------------------------------------------------------------------------
    # VALIDACIÓN
    # ------------------------------------------------------------------------

    validate_paths()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # MODELO
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("CARGANDO MODELO EXP02")
    print("=" * 72)

    model = YOLO(
        str(EXP02_MODEL)
    )

    print(
        "[OK] Modelo EXP02 cargado."
    )

    if isinstance(
        model.names,
        dict,
    ):

        print(
            f"[OK] Clases: {model.names}"
        )

    # ------------------------------------------------------------------------
    # IMÁGENES
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
            "No se encontraron imágenes en TEST_DEV."
        )

    print()
    print(
        f"[OK] Imágenes TEST_DEV: "
        f"{len(image_files):,}"
    )

    # ------------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("ANALIZANDO SMALL PERSON CON EXP02")
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
            "No se encontraron SMALL PERSON "
            "en TEST_DEV."
        )

    # ------------------------------------------------------------------------
    # MÉTRICAS
    # ------------------------------------------------------------------------

    exp02_metrics = calculate_metrics(
        all_rows
    )

    exp02_size_metrics = (
        calculate_size_metrics(
            all_rows
        )
    )

    exp02_image_metrics = (
        calculate_image_metrics(
            all_rows
        )
    )

    # ------------------------------------------------------------------------
    # EXP01 REFERENCIA
    # ------------------------------------------------------------------------
    #
    # Resultado real de EXP01 obtenido con el mismo protocolo:
    #
    # GT = 17,879
    # TP = 5,306
    # FN = 12,573
    # Recall = 29.68%
    #
    # ------------------------------------------------------------------------

    exp01_reference = {

        "gt":
            17_879,

        "tp":
            5_306,

        "fn":
            12_573,

        "recall":
            5_306 / 17_879,

        "recall_percentage":
            (
                5_306
                /
                17_879
                *
                100.0
            ),
    }

    # ------------------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------------------

    write_csv(
        OBJECTS_CSV,
        all_rows,
    )

    write_csv(
        SIZE_CSV,
        exp02_size_metrics,
    )

    write_csv(
        IMAGE_CSV,
        exp02_image_metrics,
    )

    summary_rows = [

        {
            "metric":
                "SMALL_PERSON_GT",

            "value":
                exp02_metrics[
                    "gt"
                ],
        },

        {
            "metric":
                "SMALL_PERSON_TP",

            "value":
                exp02_metrics[
                    "tp"
                ],
        },

        {
            "metric":
                "SMALL_PERSON_FN",

            "value":
                exp02_metrics[
                    "fn"
                ],
        },

        {
            "metric":
                "SMALL_PERSON_RECALL",

            "value":
                exp02_metrics[
                    "recall"
                ],
        },

        {
            "metric":
                "SMALL_PERSON_RECALL_PERCENTAGE",

            "value":
                exp02_metrics[
                    "recall_percentage"
                ],
        },

        {
            "metric":
                "IMAGE_SIZE",

            "value":
                IMAGE_SIZE,
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
    ]

    write_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    write_comparison(
        exp01_reference,
        exp02_metrics,
    )

    write_summary(
        exp02_metrics,
        exp02_size_metrics,
        exp01_reference,
    )

    # ------------------------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------------------------

    delta_recall_pp = (
        exp02_metrics[
            "recall_percentage"
        ]
        -
        exp01_reference[
            "recall_percentage"
        ]
    )

    print()
    print("=" * 72)
    print(
        "# RESULTADO EXP02 SMALL PERSON RECALL EVALUATION V1"
    )
    print("=" * 72)

    print()

    print(
        f"Imágenes con small person: "
        f"{images_with_small_person:,}"
    )

    print(
        f"SMALL PERSON GT: "
        f"{exp02_metrics['gt']:,}"
    )

    print(
        f"SMALL PERSON TP: "
        f"{exp02_metrics['tp']:,}"
    )

    print(
        f"SMALL PERSON FN: "
        f"{exp02_metrics['fn']:,}"
    )

    print(
        f"SMALL PERSON Recall: "
        f"{exp02_metrics['recall_percentage']:.2f}%"
    )

    print()
    print(
        "COMPARACIÓN EXP01 → EXP02"
    )

    print(
        f"EXP01 Recall: "
        f"{exp01_reference['recall_percentage']:.2f}%"
    )

    print(
        f"EXP02 Recall: "
        f"{exp02_metrics['recall_percentage']:.2f}%"
    )

    print(
        f"Delta: "
        f"{delta_recall_pp:+.2f} pp"
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
        f"[OK] {IMAGE_CSV}"
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
            "[CANCELADO] Evaluación EXP02 interrumpida."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print("[ERROR EXP02 EVALUATION]")
        print("=" * 72)
        print()
        print(
            str(exc)
        )
        print()

        sys.exit(1)