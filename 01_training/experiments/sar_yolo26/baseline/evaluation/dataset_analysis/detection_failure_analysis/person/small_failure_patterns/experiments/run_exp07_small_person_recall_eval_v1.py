from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26
# EXP07 - SMALL PERSON RECALL EVALUATION V1
# ============================================================================
#
# OBJETIVO
# --------
# Evaluar EXP07 sobre TEST_DEV exactamente con el mismo criterio usado en
# EXP01-EXP06.
#
# Criterios:
#
#   PERSON class = 0
#   SMALL area < 256 px²
#   IoU >= 0.50
#   confidence >= 0.25
#   imgsz = 1536
#
# Además:
#
#   - compara EXP01 vs EXP04 vs EXP07
#   - analiza factores de EXP07
#   - analiza las tres poblaciones objetivo:
#
#       EXTREME_SMALL + DENSE_SCENE
#       EXTREME_SMALL + CLOSE_NEIGHBORS
#       EXTREME_SMALL + DENSE_SCENE + CLOSE_NEIGHBORS
#
# IMPORTANTE
# ----------
# TEST_DEV solo se utiliza para EVALUACIÓN.
#
# No se modifica:
#   - dataset
#   - labels
#   - YAML oficial
#
# ============================================================================


# ============================================================================
# CONFIG
# ============================================================================

PERSON_CLASS_ID = 0

SMALL_AREA_THRESHOLD = 256.0

MATCH_IOU_THRESHOLD = 0.50

CONF_THRESHOLD = 0.25

IMAGE_SIZE = 1536

DENSE_PERSON_COUNT = 25

EXTREME_SMALL_THRESHOLD = 16.0

NEIGHBOR_DISTANCE_FACTOR = 2.0

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
# LOCALIZACIÓN
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
        "No se pudo localizar baseline."
    )


BASELINE_DIR = find_baseline_dir()


def find_project_root() -> Path:

    for parent in [
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
    ]:

        if parent.name.lower() == "sarc-drone":
            return parent

    raise RuntimeError(
        "No se pudo localizar la raíz del proyecto SARC-Drone."
    )


PROJECT_ROOT = find_project_root()


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

EXP04_MODEL = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp04_dense_scene_targeted_crops_v1"
    / "runs"
    / "exp04_dense_scene_targeted_crops"
    / "weights"
    / "best.pt"
)

EXP07_MODEL = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp07_targeted_extreme_small_dense_neighbor_v1"
    / "runs"
    / "exp07_targeted_extreme_small_dense_neighbor"
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
    / "exp07_small_person_recall_eval_v1"
    / "reports"
)

OBJECTS_CSV = (
    REPORTS_DIR
    / "exp07_small_person_objects_v1.csv"
)

SUMMARY_CSV = (
    REPORTS_DIR
    / "exp07_small_person_recall_summary_v1.csv"
)

FACTORS_CSV = (
    REPORTS_DIR
    / "exp07_small_person_recall_by_factor_v1.csv"
)

INTERACTIONS_CSV = (
    REPORTS_DIR
    / "exp07_small_person_recall_by_interaction_v1.csv"
)

SIZE_CSV = (
    REPORTS_DIR
    / "exp07_small_person_recall_by_size_v1.csv"
)

COMPARISON_CSV = (
    REPORTS_DIR
    / "exp07_vs_exp01_exp04_small_person_recall_v1.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP07_SMALL_PERSON_RECALL_EVALUATION_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_div(
    a: float,
    b: float,
) -> float:

    if b == 0:
        return 0.0

    return a / b


def iou_xyxy(
    box_a: list[float],
    box_b: list[float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw
        *
        ih
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
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def xywhn_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> list[float]:

    cx = (
        xc
        *
        image_width
    )

    cy = (
        yc
        *
        image_height
    )

    bw = (
        w
        *
        image_width
    )

    bh = (
        h
        *
        image_height
    )

    return [
        max(
            0.0,
            cx - bw / 2,
        ),

        max(
            0.0,
            cy - bh / 2,
        ),

        min(
            float(image_width),
            cx + bw / 2,
        ),

        min(
            float(image_height),
            cy + bh / 2,
        ),
    ]


def center_distance(
    box_a: list[float],
    box_b: list[float],
) -> float:

    ax = (
        box_a[0]
        +
        box_a[2]
    ) / 2.0

    ay = (
        box_a[1]
        +
        box_a[3]
    ) / 2.0

    bx = (
        box_b[0]
        +
        box_b[2]
    ) / 2.0

    by = (
        box_b[1]
        +
        box_b[3]
    ) / 2.0

    return math.hypot(
        ax - bx,
        ay - by,
    )


def write_csv(
    path: Path,
    rows: list[dict],
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
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# GT SMALL PERSON
# ============================================================================

def load_small_person_gt(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[dict]:

    persons = []

    if not label_path.exists():
        return persons

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

        parts = (
            line
            .strip()
            .split()
        )

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
            *
            h
            *
            image_width
            *
            image_height
        )

        if area >= SMALL_AREA_THRESHOLD:
            continue

        box = (
            xywhn_to_xyxy(
                xc,
                yc,
                w,
                h,
                image_width,
                image_height,
            )
        )

        size_sqrt = math.sqrt(
            area
        )

        persons.append(
            {
                "gt_index":
                    gt_index,

                "box":
                    box,

                "area":
                    area,

                "size_sqrt":
                    size_sqrt,
            }
        )

    return persons


# ============================================================================
# FACTORES
# ============================================================================

def calculate_factors(
    persons: list[dict],
    target: dict,
) -> dict:

    dense = (
        len(persons)
        >=
        DENSE_PERSON_COUNT
    )

    distances = []

    for other in persons:

        if (
            other["gt_index"]
            ==
            target["gt_index"]
        ):
            continue

        distances.append(
            center_distance(
                target["box"],
                other["box"],
            )
        )

    nearest_distance = (
        min(distances)
        if distances
        else math.inf
    )

    extreme_small = (
        target["size_sqrt"]
        <
        EXTREME_SMALL_THRESHOLD
    )

    close_neighbors = (
        nearest_distance
        <=
        NEIGHBOR_DISTANCE_FACTOR
        *
        max(
            target["size_sqrt"],
            1.0,
        )
    )

    return {
        "EXTREME_SMALL":
            int(
                extreme_small
            ),

        "DENSE_SCENE":
            int(
                dense
            ),

        "CLOSE_NEIGHBORS":
            int(
                close_neighbors
            ),

        "nearest_distance":
            nearest_distance,
    }


# ============================================================================
# PREDICCIONES
# ============================================================================

def predict_person(
    model: YOLO,
    image_path: Path,
) -> tuple[
    list[list[float]],
    list[float],
]:

    results = model.predict(
        source=str(
            image_path
        ),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        verbose=False,
        save=False,
        save_txt=False,
        save_conf=False,
    )

    if not results:

        return (
            [],
            [],
        )

    result = results[0]

    if result.boxes is None:

        return (
            [],
            [],
        )

    boxes = (
        result
        .boxes
        .xyxy
        .cpu()
        .tolist()
    )

    confidences = (
        result
        .boxes
        .conf
        .cpu()
        .tolist()
    )

    classes = (
        result
        .boxes
        .cls
        .cpu()
        .tolist()
    )

    person_boxes = []
    person_confidences = []

    for box, confidence, class_id in zip(
        boxes,
        confidences,
        classes,
    ):

        if int(class_id) != PERSON_CLASS_ID:
            continue

        person_boxes.append(
            box
        )

        person_confidences.append(
            float(
                confidence
            )
        )

    return (
        person_boxes,
        person_confidences,
    )


# ============================================================================
# MATCHING
# ============================================================================

def match_gt(
    gt_objects: list[dict],
    prediction_boxes: list[list[float]],
    prediction_confidences: list[float],
) -> list[dict]:

    used_predictions = set()

    matches = []

    for gt in gt_objects:

        best_iou = 0.0
        best_index = None

        for index, prediction in enumerate(
            prediction_boxes
        ):

            if index in used_predictions:
                continue

            current_iou = (
                iou_xyxy(
                    gt["box"],
                    prediction,
                )
            )

            if current_iou > best_iou:

                best_iou = (
                    current_iou
                )

                best_index = (
                    index
                )

        matched = (
            best_index is not None
            and
            best_iou >= MATCH_IOU_THRESHOLD
        )

        confidence = (
            prediction_confidences[
                best_index
            ]
            if matched
            else 0.0
        )

        if matched:
            used_predictions.add(
                best_index
            )

        matches.append(
            {
                "tp":
                    int(
                        matched
                    ),

                "iou":
                    best_iou,

                "confidence":
                    confidence,
            }
        )

    return matches


# ============================================================================
# PROCESAR TEST
# ============================================================================

def evaluate_models() -> list[dict]:

    print()
    print("=" * 72)
    print(
        "CARGANDO MODELOS"
    )
    print("=" * 72)

    exp01 = YOLO(
        str(
            EXP01_MODEL
        )
    )

    exp04 = YOLO(
        str(
            EXP04_MODEL
        )
    )

    exp07 = YOLO(
        str(
            EXP07_MODEL
        )
    )

    print(
        "[OK] EXP01 cargado."
    )

    print(
        "[OK] EXP04 cargado."
    )

    print(
        "[OK] EXP07 cargado."
    )

    image_files = sorted(
        [
            path
            for path
            in TEST_IMAGES_DIR.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    print()
    print(
        f"[OK] Imágenes TEST_DEV: "
        f"{len(image_files):,}"
    )

    if not image_files:

        raise RuntimeError(
            "No existen imágenes TEST_DEV."
        )

    rows = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        label_path = (
            TEST_LABELS_DIR
            /
            (
                image_path.stem
                +
                ".txt"
            )
        )

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
                f"[WARNING] "
                f"{image_path.name}: "
                f"{exc}"
            )

            continue

        gt_objects = (
            load_small_person_gt(
                label_path,
                image_width,
                image_height,
            )
        )

        if not gt_objects:
            continue

        factors = [
            calculate_factors(
                gt_objects,
                gt,
            )
            for gt in gt_objects
        ]

        # --------------------------------------------------------------
        # Inferencia
        # --------------------------------------------------------------

        pred01, conf01 = (
            predict_person(
                exp01,
                image_path,
            )
        )

        pred04, conf04 = (
            predict_person(
                exp04,
                image_path,
            )
        )

        pred07, conf07 = (
            predict_person(
                exp07,
                image_path,
            )
        )

        # --------------------------------------------------------------
        # Matching
        # --------------------------------------------------------------

        match01 = match_gt(
            gt_objects,
            pred01,
            conf01,
        )

        match04 = match_gt(
            gt_objects,
            pred04,
            conf04,
        )

        match07 = match_gt(
            gt_objects,
            pred07,
            conf07,
        )

        # --------------------------------------------------------------
        # Rows
        # --------------------------------------------------------------

        for gt, factor, m01, m04, m07 in zip(
            gt_objects,
            factors,
            match01,
            match04,
            match07,
        ):

            if (
                factor["EXTREME_SMALL"]
                and
                factor["DENSE_SCENE"]
                and
                factor["CLOSE_NEIGHBORS"]
            ):

                target_population = (
                    "EXTREME_SMALL_DENSE_NEIGHBOR"
                )

            elif (
                factor["EXTREME_SMALL"]
                and
                factor["DENSE_SCENE"]
            ):

                target_population = (
                    "EXTREME_SMALL_DENSE"
                )

            elif (
                factor["EXTREME_SMALL"]
                and
                factor["CLOSE_NEIGHBORS"]
            ):

                target_population = (
                    "EXTREME_SMALL_NEIGHBOR"
                )

            elif factor["EXTREME_SMALL"]:

                target_population = (
                    "EXTREME_SMALL_ONLY"
                )

            else:

                target_population = (
                    "SMALL_OTHER"
                )

            rows.append(
                {
                    "image":
                        image_path.name,

                    "gt_index":
                        gt["gt_index"],

                    "area":
                        gt["area"],

                    "size_sqrt":
                        gt["size_sqrt"],

                    "EXTREME_SMALL":
                        factor[
                            "EXTREME_SMALL"
                        ],

                    "DENSE_SCENE":
                        factor[
                            "DENSE_SCENE"
                        ],

                    "CLOSE_NEIGHBORS":
                        factor[
                            "CLOSE_NEIGHBORS"
                        ],

                    "nearest_distance":
                        (
                            factor[
                                "nearest_distance"
                            ]
                            if math.isfinite(
                                factor[
                                    "nearest_distance"
                                ]
                            )
                            else ""
                        ),

                    "target_population":
                        target_population,

                    "EXP01_TP":
                        m01["tp"],

                    "EXP01_iou":
                        m01["iou"],

                    "EXP01_conf":
                        m01["confidence"],

                    "EXP04_TP":
                        m04["tp"],

                    "EXP04_iou":
                        m04["iou"],

                    "EXP04_conf":
                        m04["confidence"],

                    "EXP07_TP":
                        m07["tp"],

                    "EXP07_iou":
                        m07["iou"],

                    "EXP07_conf":
                        m07["confidence"],

                    "EXP07_vs_EXP04":
                        (
                            m07["tp"]
                            -
                            m04["tp"]
                        ),

                    "EXP07_vs_EXP01":
                        (
                            m07["tp"]
                            -
                            m01["tp"]
                        ),
                }
            )

        if (
            index % 100 == 0
            or
            index == len(image_files)
        ):

            current_gt = len(
                rows
            )

            print(
                f"Analizadas: "
                f"{index:,}/"
                f"{len(image_files):,} "
                f"| Small GT: "
                f"{current_gt:,}"
            )

    if not rows:

        raise RuntimeError(
            "No se encontraron SMALL PERSON."
        )

    return rows


# ============================================================================
# MÉTRICAS
# ============================================================================

def calculate_metrics(
    rows: list[dict],
) -> dict:

    gt = len(
        rows
    )

    tp01 = sum(
        int(
            row["EXP01_TP"]
        )
        for row in rows
    )

    tp04 = sum(
        int(
            row["EXP04_TP"]
        )
        for row in rows
    )

    tp07 = sum(
        int(
            row["EXP07_TP"]
        )
        for row in rows
    )

    recall01 = (
        safe_div(
            tp01,
            gt,
        )
        *
        100
    )

    recall04 = (
        safe_div(
            tp04,
            gt,
        )
        *
        100
    )

    recall07 = (
        safe_div(
            tp07,
            gt,
        )
        *
        100
    )

    return {
        "gt":
            gt,

        "EXP01_tp":
            tp01,

        "EXP01_fn":
            gt - tp01,

        "EXP01_recall_pct":
            recall01,

        "EXP04_tp":
            tp04,

        "EXP04_fn":
            gt - tp04,

        "EXP04_recall_pct":
            recall04,

        "EXP07_tp":
            tp07,

        "EXP07_fn":
            gt - tp07,

        "EXP07_recall_pct":
            recall07,

        "EXP04_vs_EXP01_delta_pp":
            recall04 - recall01,

        "EXP07_vs_EXP01_delta_pp":
            recall07 - recall01,

        "EXP07_vs_EXP04_delta_pp":
            recall07 - recall04,

        "EXP07_tp_gain_vs_EXP01":
            tp07 - tp01,

        "EXP07_tp_gain_vs_EXP04":
            tp07 - tp04,
    }


# ============================================================================
# FACTORES / INTERACCIONES
# ============================================================================

def build_factor_report(
    rows: list[dict],
) -> list[dict]:

    factors = [
        "EXTREME_SMALL",
        "DENSE_SCENE",
        "CLOSE_NEIGHBORS",
    ]

    output = []

    for factor in factors:

        subset = [
            row
            for row in rows
            if int(
                row[factor]
            ) == 1
        ]

        if not subset:
            continue

        metrics = calculate_metrics(
            subset
        )

        output.append(
            {
                "factor":
                    factor,

                **metrics,

                "share_pct":
                    safe_div(
                        len(
                            subset
                        ),
                        len(
                            rows
                        ),
                    )
                    *
                    100,
            }
        )

    return output


def build_interaction_report(
    rows: list[dict],
) -> list[dict]:

    factor_names = [
        "EXTREME_SMALL",
        "DENSE_SCENE",
        "CLOSE_NEIGHBORS",
    ]

    output = []

    for i, factor_a in enumerate(
        factor_names
    ):

        for factor_b in factor_names[
            i + 1:
        ]:

            subset = [
                row
                for row in rows
                if (
                    int(
                        row[factor_a]
                    ) == 1
                    and
                    int(
                        row[factor_b]
                    ) == 1
                )
            ]

            if not subset:
                continue

            metrics = calculate_metrics(
                subset
            )

            output.append(
                {
                    "interaction":
                        f"{factor_a} + {factor_b}",

                    **metrics,

                    "share_pct":
                        safe_div(
                            len(
                                subset
                            ),
                            len(
                                rows
                            ),
                        )
                        *
                        100,
                }
            )

    # Triple
    triple = [
        row
        for row in rows
        if (
            int(
                row["EXTREME_SMALL"]
            )
            ==
            1
            and
            int(
                row["DENSE_SCENE"]
            )
            ==
            1
            and
            int(
                row["CLOSE_NEIGHBORS"]
            )
            ==
            1
        )
    ]

    if triple:

        metrics = calculate_metrics(
            triple
        )

        output.append(
            {
                "interaction":
                    (
                        "EXTREME_SMALL + "
                        "DENSE_SCENE + "
                        "CLOSE_NEIGHBORS"
                    ),

                **metrics,

                "share_pct":
                    safe_div(
                        len(
                            triple
                        ),
                        len(
                            rows
                        ),
                    )
                    *
                    100,
            }
        )

    output.sort(
        key=lambda row:
            row[
                "EXP07_vs_EXP04_delta_pp"
            ],
        reverse=True,
    )

    return output


# ============================================================================
# SIZE
# ============================================================================

def build_size_report(
    rows: list[dict],
) -> list[dict]:

    buckets = [
        (
            "<16",
            lambda s: s < 16,
        ),

        (
            "16-32",
            lambda s: (
                16 <= s < 32
            ),
        ),

        (
            "32-64",
            lambda s: (
                32 <= s < 64
            ),
        ),

        (
            "64-128",
            lambda s: (
                64 <= s < 128
            ),
        ),

        (
            "128-256",
            lambda s: (
                128 <= s < 256
            ),
        ),
    ]

    output = []

    for bucket_name, selector in buckets:

        subset = [
            row
            for row in rows
            if selector(
                float(
                    row["size_sqrt"]
                )
            )
        ]

        if not subset:
            continue

        metrics = calculate_metrics(
            subset
        )

        output.append(
            {
                "size_bucket":
                    bucket_name,

                **metrics,
            }
        )

    return output


# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

def build_comparison(
    rows: list[dict],
) -> list[dict]:

    metrics = calculate_metrics(
        rows
    )

    output = [
        {
            "model":
                "EXP01",

            "tp":
                metrics[
                    "EXP01_tp"
                ],

            "fn":
                metrics[
                    "EXP01_fn"
                ],

            "recall_pct":
                metrics[
                    "EXP01_recall_pct"
                ],

            "delta_vs_EXP01_pp":
                0.0,
        },

        {
            "model":
                "EXP04",

            "tp":
                metrics[
                    "EXP04_tp"
                ],

            "fn":
                metrics[
                    "EXP04_fn"
                ],

            "recall_pct":
                metrics[
                    "EXP04_recall_pct"
                ],

            "delta_vs_EXP01_pp":
                metrics[
                    "EXP04_vs_EXP01_delta_pp"
                ],
        },

        {
            "model":
                "EXP07",

            "tp":
                metrics[
                    "EXP07_tp"
                ],

            "fn":
                metrics[
                    "EXP07_fn"
                ],

            "recall_pct":
                metrics[
                    "EXP07_recall_pct"
                ],

            "delta_vs_EXP01_pp":
                metrics[
                    "EXP07_vs_EXP01_delta_pp"
                ],
        },
    ]

    return output


# ============================================================================
# SUMMARY TXT
# ============================================================================

def write_summary(
    rows: list[dict],
    factors: list[dict],
    interactions: list[dict],
    sizes: list[dict],
) -> None:

    metrics = calculate_metrics(
        rows
    )

    lines = [
        "=" * 72,
        "SAR YOLO26 - EXP07 SMALL PERSON RECALL EVALUATION V1",
        "=" * 72,
        "",
        "TEST_DEV",
        f"Images: {len(set(row['image'] for row in rows)):,}",
        "",
        "CRITERIOS",
        f"SMALL_AREA_THRESHOLD = {SMALL_AREA_THRESHOLD}",
        f"IOU_THRESHOLD        = {MATCH_IOU_THRESHOLD}",
        f"CONF_THRESHOLD       = {CONF_THRESHOLD}",
        f"IMAGE_SIZE           = {IMAGE_SIZE}",
        "",
        "RESULTADO GLOBAL",
        "-" * 72,
        (
            f"EXP01 Recall: "
            f"{metrics['EXP01_recall_pct']:.2f}%"
        ),
        (
            f"EXP04 Recall: "
            f"{metrics['EXP04_recall_pct']:.2f}%"
        ),
        (
            f"EXP07 Recall: "
            f"{metrics['EXP07_recall_pct']:.2f}%"
        ),
        (
            f"EXP04 vs EXP01: "
            f"{metrics['EXP04_vs_EXP01_delta_pp']:+.2f} pp"
        ),
        (
            f"EXP07 vs EXP01: "
            f"{metrics['EXP07_vs_EXP01_delta_pp']:+.2f} pp"
        ),
        (
            f"EXP07 vs EXP04: "
            f"{metrics['EXP07_vs_EXP04_delta_pp']:+.2f} pp"
        ),
        (
            f"EXP07 TP gain vs EXP04: "
            f"{metrics['EXP07_tp_gain_vs_EXP04']:+d}"
        ),
        "",
        "FACTORES",
        "-" * 72,
    ]

    for row in factors:

        lines.append(
            (
                f"{row['factor']:<20} "
                f"GT={row['gt']:>7,} "
                f"EXP04={row['EXP04_recall_pct']:>7.2f}% "
                f"EXP07={row['EXP07_recall_pct']:>7.2f}% "
                f"Delta={row['EXP07_vs_EXP04_delta_pp']:+7.2f} pp "
                f"TPgain={row['EXP07_tp_gain_vs_EXP04']:+d}"
            )
        )

    lines.extend(
        [
            "",
            "INTERACCIONES",
            "-" * 72,
        ]
    )

    for row in interactions:

        lines.append(
            (
                f"{row['interaction']:<50} "
                f"GT={row['gt']:>6,} "
                f"EXP04={row['EXP04_recall_pct']:>7.2f}% "
                f"EXP07={row['EXP07_recall_pct']:>7.2f}% "
                f"Delta={row['EXP07_vs_EXP04_delta_pp']:+7.2f} pp "
                f"TPgain={row['EXP07_tp_gain_vs_EXP04']:+d}"
            )
        )

    lines.extend(
        [
            "",
            "TAMAÑO",
            "-" * 72,
        ]
    )

    for row in sizes:

        lines.append(
            (
                f"{row['size_bucket']:>8} "
                f"GT={row['gt']:>7,} "
                f"EXP04={row['EXP04_recall_pct']:>7.2f}% "
                f"EXP07={row['EXP07_recall_pct']:>7.2f}% "
                f"Delta={row['EXP07_vs_EXP04_delta_pp']:+7.2f} pp"
            )
        )

    lines.extend(
        [
            "",
            "DECISION",
            "-" * 72,
        ]
    )

    if (
        metrics[
            "EXP07_vs_EXP04_delta_pp"
        ]
        >
        1.0
    ):

        lines.append(
            (
                "EXP07 mejora claramente a EXP04."
            )
        )

    elif (
        metrics[
            "EXP07_vs_EXP04_delta_pp"
        ]
        >
        0.0
    ):

        lines.append(
            (
                "EXP07 mejora ligeramente a EXP04."
            )
        )

    else:

        lines.append(
            (
                "EXP07 no supera a EXP04 globalmente."
            )
        )

    lines.append(
        "IMPORTANTE: TEST_DEV solo se utilizó para evaluación."
    )

    lines.append(
        "Dataset, labels y YAML original NO modificados."
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
        "# SAR YOLO26 - EXP07 SMALL PERSON RECALL EVALUATION V1"
    )
    print("=" * 72)

    print()
    print(
        "EXP07 MODEL:"
    )

    print(
        f"  {EXP07_MODEL}"
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
    # Validación
    # ------------------------------------------------------------------------

    required = {
        "EXP01_MODEL":
            EXP01_MODEL,

        "EXP04_MODEL":
            EXP04_MODEL,

        "EXP07_MODEL":
            EXP07_MODEL,

        "DATASET_ROOT":
            DATASET_ROOT,

        "TEST_IMAGES_DIR":
            TEST_IMAGES_DIR,

        "TEST_LABELS_DIR":
            TEST_LABELS_DIR,
    }

    print()
    print("=" * 72)
    print(
        "VALIDANDO ESTRUCTURA EXP07 EVALUATION"
    )
    print("=" * 72)

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

    # ------------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------------

    rows = evaluate_models()

    metrics = calculate_metrics(
        rows
    )

    factors = build_factor_report(
        rows
    )

    interactions = build_interaction_report(
        rows
    )

    sizes = build_size_report(
        rows
    )

    comparison = build_comparison(
        rows
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OBJECTS_CSV,
        rows,
    )

    write_csv(
        SUMMARY_CSV,
        [
            metrics
        ],
    )

    write_csv(
        FACTORS_CSV,
        factors,
    )

    write_csv(
        INTERACTIONS_CSV,
        interactions,
    )

    write_csv(
        SIZE_CSV,
        sizes,
    )

    write_csv(
        COMPARISON_CSV,
        comparison,
    )

    write_summary(
        rows,
        factors,
        interactions,
        sizes,
    )

    # ------------------------------------------------------------------------
    # Resultado
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "# RESULTADO EXP07 SMALL PERSON RECALL"
    )
    print("=" * 72)

    print()

    print(
        f"SMALL PERSON GT:   "
        f"{metrics['gt']:,}"
    )

    print()

    print(
        f"EXP01 Recall:      "
        f"{metrics['EXP01_recall_pct']:.2f}%"
    )

    print(
        f"EXP04 Recall:      "
        f"{metrics['EXP04_recall_pct']:.2f}%"
    )

    print(
        f"EXP07 Recall:      "
        f"{metrics['EXP07_recall_pct']:.2f}%"
    )

    print()

    print(
        f"EXP07 - EXP01:     "
        f"{metrics['EXP07_vs_EXP01_delta_pp']:+.2f} pp"
    )

    print(
        f"EXP07 - EXP04:     "
        f"{metrics['EXP07_vs_EXP04_delta_pp']:+.2f} pp"
    )

    print()

    print(
        f"EXP07 TP gain vs EXP01: "
        f"{metrics['EXP07_tp_gain_vs_EXP01']:+d}"
    )

    print(
        f"EXP07 TP gain vs EXP04: "
        f"{metrics['EXP07_tp_gain_vs_EXP04']:+d}"
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
        f"[OK] {FACTORS_CSV}"
    )

    print(
        f"[OK] {INTERACTIONS_CSV}"
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
        "IMPORTANTE: dataset original NO modificado."
    )

    print(
        "IMPORTANTE: YAML oficial NO modificado."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\n[CANCELADO]"
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "=" * 72
        )
        print(
            "[ERROR EXP07 EVALUATION]"
        )
        print("=" * 72)
        print()
        print(
            str(exc)
        )
        print()

        sys.exit(1)