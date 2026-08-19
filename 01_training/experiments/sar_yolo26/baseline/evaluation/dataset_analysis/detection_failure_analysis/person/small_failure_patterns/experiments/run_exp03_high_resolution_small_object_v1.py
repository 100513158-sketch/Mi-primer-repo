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
# EXP03 - HIGH RESOLUTION SMALL OBJECT V1
# ============================================================================
#
# OBJETIVO
# --------
# Evaluar si aumentar la resolución de entrenamiento de 640 a 960 mejora
# específicamente la detección de PERSON pequeñas.
#
# HIPÓTESIS
# ---------
# La resolución de 640 limita la información espacial disponible para personas
# pequeñas. Aumentar imgsz a 960 debería mejorar especialmente SMALL PERSON
# RECALL.
#
# CAMBIO EXPERIMENTAL
# -------------------
# EXP01 / EXP02:
#     imgsz entrenamiento = 640
#
# EXP03:
#     imgsz entrenamiento = 960
#
# RESTO DE LA CONFIGURACIÓN
# --------------------------
#     YOLO26s
#     epochs  = 100
#     batch   = 8
#     workers = 8
#     seed    = 42
#     device  = 0
#
# EVALUACIÓN
# ----------
#     test_dev
#     imgsz = 1536
#     conf  = 0.25
#     matching IoU = 0.50
#     PERSON = class 0
#     SMALL PERSON = area < 256 px²
#
# IMPORTANTE
# ----------
# - No modifica el dataset original.
# - No modifica labels.
# - No modifica imágenes.
# - No modifica el YAML oficial.
# - Genera únicamente artefactos de EXP03.
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

EXPERIMENT_NAME = (
    "exp03_high_resolution_small_object_v1"
)

PERSON_CLASS_ID = 0

SMALL_AREA_THRESHOLD = 256.0

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

TRAIN_IMAGE_SIZE = 960

EPOCHS = 100

BATCH = 8

WORKERS = 8

DEVICE = 0

SEED = 42

AMP = True

PATIENCE = 20

CACHE = False

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

EVAL_IMAGE_SIZE = 1536

EVAL_CONF_THRESHOLD = 0.25

EVAL_MATCH_IOU = 0.50

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


TRAIN_IMAGES_DIR = (
    DATASET_ROOT
    / "train"
    / "images"
)


VAL_IMAGES_DIR = (
    DATASET_ROOT
    / "val"
    / "images"
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
# PRETRAINED MODEL
# ============================================================================

MODEL_CANDIDATES = [

    BASELINE_DIR
    / "yolo26s.pt",

    BASELINE_DIR
    / "training"
    / "models"
    / "pretrained"
    / "yolo26s.pt",

    PROJECT_ROOT
    / "01_training"
    / "models"
    / "pretrained"
    / "yolo26s.pt",

    PROJECT_ROOT
    / "yolo26s.pt",
]


def find_pretrained_model() -> Path:

    for candidate in MODEL_CANDIDATES:

        if candidate.is_file():

            return candidate

    candidates = []

    for root in [
        PROJECT_ROOT / "01_training",
        PROJECT_ROOT,
    ]:

        if not root.exists():
            continue

        try:

            for candidate in root.rglob(
                "yolo26s.pt"
            ):

                if candidate.is_file():

                    candidates.append(
                        candidate
                    )

        except PermissionError:

            continue

    candidates = sorted(
        set(candidates),
        key=lambda p: str(p).lower(),
    )

    if len(candidates) == 1:

        return candidates[0]

    if len(candidates) > 1:

        text = "\n".join(
            f"  - {path}"
            for path in candidates
        )

        raise RuntimeError(
            "Se encontraron varias copias de yolo26s.pt:\n"
            f"{text}\n\n"
            "No se seleccionará ninguna automáticamente."
        )

    raise FileNotFoundError(
        "No se encontró yolo26s.pt."
    )


# ============================================================================
# EXPERIMENT DIRECTORY
# ============================================================================

EXPERIMENT_ROOT = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / EXPERIMENT_NAME
)


RUNS_DIR = (
    EXPERIMENT_ROOT
    / "runs"
)


TEMP_DATA_YAML = (
    EXPERIMENT_ROOT
    / "exp03_dataset.yaml"
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
    / EXPERIMENT_NAME
    / "reports"
)


TRAINING_CSV = (
    REPORTS_DIR
    / "exp03_training_configuration_v1.csv"
)


EVAL_CSV = (
    REPORTS_DIR
    / "exp03_small_person_recall_v1.csv"
)


SIZE_CSV = (
    REPORTS_DIR
    / "exp03_small_person_recall_by_size_v1.csv"
)


OBJECTS_CSV = (
    REPORTS_DIR
    / "exp03_small_person_objects_v1.csv"
)


COMPARISON_CSV = (
    REPORTS_DIR
    / "exp03_vs_exp01_exp02_v1.csv"
)


SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP03_HIGH_RESOLUTION_SMALL_OBJECT_V1_SUMMARY.txt"
)


# ============================================================================
# UTILITIES
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

    intersection = iw * ih

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
# VALIDATE STRUCTURE
# ============================================================================

def validate_structure() -> None:

    print()
    print("=" * 72)
    print("VALIDANDO ESTRUCTURA EXP03")
    print("=" * 72)
    print()

    required = {

        "PROJECT_ROOT":
            PROJECT_ROOT,

        "BASELINE_DIR":
            BASELINE_DIR,

        "DATASET_ROOT":
            DATASET_ROOT,

        "TRAIN_IMAGES_DIR":
            TRAIN_IMAGES_DIR,

        "VAL_IMAGES_DIR":
            VAL_IMAGES_DIR,

        "TEST_IMAGES_DIR":
            TEST_IMAGES_DIR,

        "TEST_LABELS_DIR":
            TEST_LABELS_DIR,
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

    model_path = find_pretrained_model()

    print()
    print(
        "[OK] PRETRAINED MODEL"
    )

    print(
        f"     {model_path}"
    )


# ============================================================================
# EXPERIMENTAL YAML
# ============================================================================

def create_experiment_yaml() -> None:

    """
    YAML exclusivo de EXP03.
    No modifica sar_visdrone_2class.yaml.
    """

    content = f"""path: {DATASET_ROOT.as_posix()}

train: {TRAIN_IMAGES_DIR.as_posix()}
val: {VAL_IMAGES_DIR.as_posix()}
test: {TEST_IMAGES_DIR.as_posix()}

names:
  0: person
  1: vehicle
"""

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DATA_YAML.write_text(
        content,
        encoding="utf-8",
    )

    print()
    print(
        "[OK] YAML experimental EXP03:"
    )

    print(
        f"     {TEMP_DATA_YAML}"
    )

    print(
        "[INFO] YAML oficial NO modificado."
    )


# ============================================================================
# TRAINING CONFIGURATION REPORT
# ============================================================================

def write_training_configuration(
    model_path: Path,
) -> None:

    rows = [

        {
            "parameter":
                "model",

            "value":
                "YOLO26s",
        },

        {
            "parameter":
                "pretrained_model",

            "value":
                str(model_path),
        },

        {
            "parameter":
                "train_imgsz",

            "value":
                TRAIN_IMAGE_SIZE,
        },

        {
            "parameter":
                "exp01_train_imgsz",

            "value":
                640,
        },

        {
            "parameter":
                "epochs",

            "value":
                EPOCHS,
        },

        {
            "parameter":
                "batch",

            "value":
                BATCH,
        },

        {
            "parameter":
                "workers",

            "value":
                WORKERS,
        },

        {
            "parameter":
                "seed",

            "value":
                SEED,
        },

        {
            "parameter":
                "device",

            "value":
                DEVICE,
        },

        {
            "parameter":
                "eval_imgsz",

            "value":
                EVAL_IMAGE_SIZE,
        },

        {
            "parameter":
                "small_area_threshold",

            "value":
                SMALL_AREA_THRESHOLD,
        },
    ]

    with TRAINING_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "value",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_exp03(
    model_path: Path,
) -> Path:

    print()
    print("=" * 72)
    print("ENTRENAMIENTO EXP03")
    print("=" * 72)
    print()

    print(
        f"Modelo inicial: {model_path}"
    )

    print(
        f"Train imgsz:    {TRAIN_IMAGE_SIZE}"
    )

    print(
        f"Epochs:         {EPOCHS}"
    )

    print(
        f"Batch:          {BATCH}"
    )

    print(
        f"Workers:        {WORKERS}"
    )

    print(
        f"Seed:           {SEED}"
    )

    print()

    model = YOLO(
        str(model_path)
    )

    try:

        results = model.train(

            data=str(
                TEMP_DATA_YAML
            ),

            epochs=EPOCHS,

            imgsz=TRAIN_IMAGE_SIZE,

            batch=BATCH,

            workers=WORKERS,

            device=DEVICE,

            seed=SEED,

            amp=AMP,

            cache=CACHE,

            patience=PATIENCE,

            project=str(
                RUNS_DIR
            ),

            name="exp03_high_resolution_small_object",

            pretrained=True,

            save=True,

            plots=True,

            verbose=True,
        )

    except RuntimeError as exc:

        error_text = str(
            exc
        ).lower()

        if (
            "out of memory"
            in error_text
            or "cuda out of memory"
            in error_text
        ):

            raise RuntimeError(
                "\nEXP03 no pudo arrancar por falta de VRAM "
                "con la configuración controlada:\n\n"
                f"imgsz={TRAIN_IMAGE_SIZE}\n"
                f"batch={BATCH}\n"
                f"device={DEVICE}\n\n"
                "No se ha cambiado automáticamente el batch "
                "porque eso alteraría el diseño experimental.\n"
                "Si ocurre este error, debemos decidir "
                "explícitamente cómo adaptar el experimento."
            ) from exc

        raise

    save_dir = Path(
        results.save_dir
    )

    best_path = (
        save_dir
        / "weights"
        / "best.pt"
    )

    if not best_path.exists():

        raise FileNotFoundError(
            "El entrenamiento terminó pero "
            "no se encontró best.pt:\n"
            f"{best_path}"
        )

    return best_path


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
                float(
                    parts[0]
                )
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
                "gt_index":
                    gt_index,

                "box":
                    box,

                "area":
                    area,

                "size_sqrt":
                    math.sqrt(
                        max(
                            area,
                            0.0,
                        )
                    ),
            }
        )

    return objects


# ============================================================================
# MATCH
# ============================================================================

def match_gt_predictions(
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

                best_iou = (
                    current_iou
                )

                best_index = (
                    prediction_index
                )

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

            image_width = (
                image.width
            )

            image_height = (
                image.height
            )

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

            imgsz=EVAL_IMAGE_SIZE,

            conf=EVAL_CONF_THRESHOLD,

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

            if (
                int(class_id)
                != PERSON_CLASS_ID
            ):

                continue

            prediction_boxes.append(
                box
            )

            prediction_confidences.append(
                float(
                    confidence
                )
            )

    matches = match_gt_predictions(
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

    return output


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    rows: List[Dict],
) -> Dict:

    gt = len(
        rows
    )

    tp = sum(
        1
        for row in rows
        if row["status"] == "TP"
    )

    fn = (
        gt
        -
        tp
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
# SIZE BUCKETS
# ============================================================================

def size_bucket(
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


def calculate_size_metrics(
    rows: List[Dict],
) -> List[Dict]:

    groups = defaultdict(
        list
    )

    for row in rows:

        groups[
            size_bucket(
                float(
                    row["size_sqrt"]
                )
            )
        ].append(
            row
        )

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

        writer.writerows(
            rows
        )


# ============================================================================
# COMPARISON
# ============================================================================

def write_comparison(
    exp03_metrics: Dict,
) -> None:

    # Valores obtenidos experimentalmente con EXP01 y EXP02
    # usando el mismo protocolo de SMALL PERSON.
    exp01 = {
        "gt": 17879,
        "tp": 5306,
        "fn": 12573,
        "recall_percentage": 29.68,
    }

    exp02 = {
        "gt": 17879,
        "tp": 5295,
        "fn": 12584,
        "recall_percentage": 29.62,
    }

    rows = []

    for name, reference in [
        ("EXP01", exp01),
        ("EXP02", exp02),
    ]:

        rows.append(
            {
                "comparison":
                    name,

                "gt":
                    reference["gt"],

                "tp":
                    reference["tp"],

                "fn":
                    reference["fn"],

                "recall_percentage":
                    reference[
                        "recall_percentage"
                    ],

                "delta_vs_exp03_pp":
                    (
                        exp03_metrics[
                            "recall_percentage"
                        ]
                        -
                        reference[
                            "recall_percentage"
                        ]
                    ),
            }
        )

    rows.append(
        {
            "comparison":
                "EXP03",

            "gt":
                exp03_metrics["gt"],

            "tp":
                exp03_metrics["tp"],

            "fn":
                exp03_metrics["fn"],

            "recall_percentage":
                exp03_metrics[
                    "recall_percentage"
                ],

            "delta_vs_exp03_pp":
                0.0,
        }
    )

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
    best_path: Path,
) -> None:

    exp01_recall = 29.68

    exp02_recall = 29.62

    delta_exp01 = (
        metrics[
            "recall_percentage"
        ]
        -
        exp01_recall
    )

    delta_exp02 = (
        metrics[
            "recall_percentage"
        ]
        -
        exp02_recall
    )

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "SAR YOLO26 - EXP03 HIGH RESOLUTION SMALL OBJECT V1"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "HIPÓTESIS"
    )

    lines.append(
        "Aumentar imgsz durante entrenamiento "
        "de 640 a 960 puede mejorar la detección "
        "de PERSON pequeñas."
    )

    lines.append("")

    lines.append(
        "CONFIGURACIÓN"
    )

    lines.append(
        f"Train imgsz:       {TRAIN_IMAGE_SIZE}"
    )

    lines.append(
        f"Baseline imgsz:    640"
    )

    lines.append(
        f"Epochs:             {EPOCHS}"
    )

    lines.append(
        f"Batch:              {BATCH}"
    )

    lines.append(
        f"Workers:            {WORKERS}"
    )

    lines.append(
        f"Seed:               {SEED}"
    )

    lines.append(
        f"Eval imgsz:         {EVAL_IMAGE_SIZE}"
    )

    lines.append(
        f"Confidence:         {EVAL_CONF_THRESHOLD}"
    )

    lines.append(
        f"Match IoU:          {MATCH_IOU_THRESHOLD}"
    )

    lines.append(
        f"Small threshold:    < {SMALL_AREA_THRESHOLD} px²"
    )

    lines.append("")

    lines.append(
        "MODELO"
    )

    lines.append(
        str(
            best_path
        )
    )

    lines.append("")

    lines.append(
        "RESULTADO EXP03"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"SMALL PERSON GT:    "
        f"{metrics['gt']:,}"
    )

    lines.append(
        f"SMALL PERSON TP:    "
        f"{metrics['tp']:,}"
    )

    lines.append(
        f"SMALL PERSON FN:    "
        f"{metrics['fn']:,}"
    )

    lines.append(
        f"SMALL PERSON Recall:"
        f" {metrics['recall_percentage']:.2f}%"
    )

    lines.append("")

    lines.append(
        "COMPARACIÓN"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"EXP01 Recall:       {exp01_recall:.2f}%"
    )

    lines.append(
        f"EXP02 Recall:       {exp02_recall:.2f}%"
    )

    lines.append(
        f"EXP03 Recall:       "
        f"{metrics['recall_percentage']:.2f}%"
    )

    lines.append(
        f"EXP03 - EXP01:      "
        f"{delta_exp01:+.2f} pp"
    )

    lines.append(
        f"EXP03 - EXP02:      "
        f"{delta_exp02:+.2f} pp"
    )

    lines.append("")

    if delta_exp01 > 2.0:

        interpretation = (
            "MEJORA FUERTE: la mayor resolución "
            "parece tener un efecto relevante "
            "sobre SMALL PERSON."
        )

    elif delta_exp01 > 0.5:

        interpretation = (
            "MEJORA MODERADA: la resolución "
            "muestra una señal positiva."
        )

    elif delta_exp01 > -0.5:

        interpretation = (
            "SIN CAMBIO RELEVANTE: aumentar "
            "640 -> 960 no cambia apreciablemente "
            "el SMALL PERSON Recall."
        )

    else:

        interpretation = (
            "EMPEORAMIENTO: la resolución mayor "
            "no mejora el objetivo y reduce el recall."
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

    lines.append(
        "-" * 72
    )

    if delta_exp01 > 0.5:

        lines.append(
            "La resolución muestra evidencia positiva."
        )

        lines.append(
            "Debe considerarse para la siguiente combinación "
            "experimental."
        )

    else:

        lines.append(
            "La resolución 960 por sí sola no demuestra "
            "una mejora suficiente."
        )

        lines.append(
            "El foco debe desplazarse hacia el siguiente "
            "mecanismo del diagnóstico."
        )

    lines.append("")

    lines.append(
        "IMPORTANTE: el dataset original NO ha sido modificado."
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
        "# SAR YOLO26 - EXP03 HIGH RESOLUTION SMALL OBJECT V1"
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
        "PROJECT ROOT:"
    )

    print(
        f"  {PROJECT_ROOT}"
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
        "TRAIN IMG SIZE:"
    )

    print(
        f"  {TRAIN_IMAGE_SIZE}"
    )

    print()
    print(
        "EVALUATION IMG SIZE:"
    )

    print(
        f"  {EVAL_IMAGE_SIZE}"
    )

    # ------------------------------------------------------------------------
    # VALIDACIÓN
    # ------------------------------------------------------------------------

    validate_structure()

    # ------------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------------

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    model_path = (
        find_pretrained_model()
    )

    print()
    print(
        "[OK] YOLO26s pretrained:"
    )

    print(
        f"     {model_path}"
    )

    # ------------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------------

    create_experiment_yaml()

    write_training_configuration(
        model_path
    )

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    best_path = train_exp03(
        model_path
    )

    print()
    print(
        "[OK] BEST MODEL:"
    )

    print(
        f"     {best_path}"
    )

    # ------------------------------------------------------------------------
    # LOAD BEST
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "EVALUACIÓN SMALL PERSON EXP03"
    )

    print(
        "=" * 72
    )

    model = YOLO(
        str(best_path)
    )

    # ------------------------------------------------------------------------
    # TEST IMAGES
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
        f"[OK] TEST images: "
        f"{len(image_files):,}"
    )

    # ------------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------------

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

    summary_rows = [

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
                TRAIN_IMAGE_SIZE,
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
                EVAL_CONF_THRESHOLD,
        },

        {
            "metric":
                "MATCH_IOU",

            "value":
                EVAL_MATCH_IOU,
        },
    ]

    write_csv(
        EVAL_CSV,
        summary_rows,
    )

    write_comparison(
        metrics
    )

    write_summary(
        metrics,
        size_metrics,
        best_path,
    )

    # ------------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "# RESULTADO EXP03 HIGH RESOLUTION SMALL OBJECT V1"
    )

    print("=" * 72)

    print()

    print(
        f"SMALL PERSON GT:       "
        f"{metrics['gt']:,}"
    )

    print(
        f"SMALL PERSON TP:       "
        f"{metrics['tp']:,}"
    )

    print(
        f"SMALL PERSON FN:       "
        f"{metrics['fn']:,}"
    )

    print(
        f"SMALL PERSON Recall:   "
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

    print()

    print(
        "REPORTS:"
    )

    print(
        f"[OK] {TRAINING_CSV}"
    )

    print(
        f"[OK] {OBJECTS_CSV}"
    )

    print(
        f"[OK] {EVAL_CSV}"
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
            "[CANCELADO] EXP03 interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)

        print(
            "[ERROR EXP03]"
        )

        print("=" * 72)

        print()

        print(
            str(exc)
        )

        print()

        sys.exit(1)