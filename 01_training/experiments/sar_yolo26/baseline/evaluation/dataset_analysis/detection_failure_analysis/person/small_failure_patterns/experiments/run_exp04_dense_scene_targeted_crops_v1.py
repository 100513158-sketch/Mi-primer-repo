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
# EXP04 - DENSE SCENE TARGETED CROPS V1
# ============================================================================
#
# OBJETIVO
# --------
# Mejorar la detección de PERSON pequeñas en escenas densas mediante crops
# dirigidos a regiones donde existen muchas personas.
#
# BASE EXPERIMENTAL
# -----------------
# EXP03 ha demostrado una mejora con:
#
#     train imgsz = 960
#
# EXP04 mantiene esa configuración y añade UNA intervención:
#
#     targeted dense-scene crops
#
# HIPÓTESIS
# ---------
# La pérdida de PERSON pequeñas en escenas densas está relacionada con la
# reducción de resolución efectiva de cada persona y con la interferencia
# entre personas próximas.
#
# INTERVENCIÓN
# ------------
# 1. Identificar imágenes TRAIN con >= 25 PERSON.
# 2. Crear un crop dirigido alrededor del centro de la población de personas.
# 3. Transformar las anotaciones al sistema de coordenadas del crop.
# 4. Mantener las imágenes originales.
# 5. Añadir los crops como ejemplos adicionales de entrenamiento.
#
# NO SE MODIFICA:
#     - dataset original
#     - train original
#     - val original
#     - test_dev
#     - labels originales
#     - YAML oficial
#
# ENTRENAMIENTO
# -------------
#     YOLO26s
#     imgsz = 960
#     epochs = 100
#     batch = 8
#     workers = 8
#     seed = 42
#
# EVALUACIÓN
# ----------
#     test_dev
#     imgsz = 1536
#     confidence = 0.25
#     IoU = 0.50
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

EXPERIMENT_NAME = (
    "exp04_dense_scene_targeted_crops_v1"
)

PERSON_CLASS_ID = 0

SMALL_AREA_THRESHOLD = 256.0

# Una imagen se considera DENSE_SCENE cuando contiene >= 25 personas.
DENSE_PERSON_THRESHOLD = 25

# Crop dirigido.
#
# 0.50 significa un crop de aproximadamente el 50 % del ancho y el 50 %
# de la altura de la imagen original.
CROP_WIDTH_RATIO = 0.50
CROP_HEIGHT_RATIO = 0.50

# Margen para que una caja centrada en el borde del crop no sea eliminada.
CROP_CONTEXT_MARGIN = 0.10

# Solo conservar anotaciones cuyo centro esté dentro del crop.
MIN_BOX_VISIBILITY = 0.35

# Entrenamiento: heredado de EXP03.
TRAIN_IMAGE_SIZE = 960
EPOCHS = 100
BATCH = 8
WORKERS = 8
DEVICE = 0
SEED = 42
AMP = True
PATIENCE = 20
CACHE = False

# Evaluación.
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
# PROYECTO
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
# DATASET ORIGINAL
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

TRAIN_LABELS_DIR = (
    DATASET_ROOT
    / "train"
    / "labels"
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
# MODELO
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
            for candidate in root.rglob("yolo26s.pt"):
                if candidate.is_file():
                    candidates.append(candidate)

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
# EXP04
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

CROPS_IMAGES_DIR = (
    EXPERIMENT_ROOT
    / "dense_crops"
    / "images"
)

CROPS_LABELS_DIR = (
    EXPERIMENT_ROOT
    / "dense_crops"
    / "labels"
)

TRAIN_MANIFEST = (
    EXPERIMENT_ROOT
    / "train_with_dense_crops.txt"
)

TEMP_DATA_YAML = (
    EXPERIMENT_ROOT
    / "exp04_dataset.yaml"
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

CROP_STATS_CSV = (
    REPORTS_DIR
    / "exp04_dense_crop_statistics_v1.csv"
)

TRAIN_CONFIG_CSV = (
    REPORTS_DIR
    / "exp04_training_configuration_v1.csv"
)

EVAL_CSV = (
    REPORTS_DIR
    / "exp04_small_person_recall_v1.csv"
)

SIZE_CSV = (
    REPORTS_DIR
    / "exp04_small_person_recall_by_size_v1.csv"
)

OBJECTS_CSV = (
    REPORTS_DIR
    / "exp04_small_person_objects_v1.csv"
)

COMPARISON_CSV = (
    REPORTS_DIR
    / "exp04_vs_exp01_exp02_exp03_v1.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP04_DENSE_SCENE_TARGETED_CROPS_V1_SUMMARY.txt"
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


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
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
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
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

def validate_structure() -> None:

    print()
    print("=" * 72)
    print("VALIDANDO ESTRUCTURA EXP04")
    print("=" * 72)
    print()

    required = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "BASELINE_DIR": BASELINE_DIR,
        "DATASET_ROOT": DATASET_ROOT,
        "TRAIN_IMAGES_DIR": TRAIN_IMAGES_DIR,
        "TRAIN_LABELS_DIR": TRAIN_LABELS_DIR,
        "VAL_IMAGES_DIR": VAL_IMAGES_DIR,
        "TEST_IMAGES_DIR": TEST_IMAGES_DIR,
        "TEST_LABELS_DIR": TEST_LABELS_DIR,
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

    model_path = find_pretrained_model()

    print()
    print(
        "[OK] PRETRAINED MODEL"
    )

    print(
        f"     {model_path}"
    )


# ============================================================================
# LEER LABELS ORIGINALES
# ============================================================================

def load_all_person_boxes(
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
                "class_id": class_id,
                "box": box,
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
            }
        )

    return objects


# ============================================================================
# DENSE SCENE
# ============================================================================

def is_dense_scene(
    person_count: int,
) -> bool:

    return (
        person_count
        >= DENSE_PERSON_THRESHOLD
    )


# ============================================================================
# CROP GEOMETRY
# ============================================================================

def compute_targeted_crop(
    person_boxes: List[Dict],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:

    crop_width = int(
        round(
            image_width
            * CROP_WIDTH_RATIO
        )
    )

    crop_height = int(
        round(
            image_height
            * CROP_HEIGHT_RATIO
        )
    )

    # Centro de la población de personas.
    centers_x = [
        (
            box["box"][0]
            +
            box["box"][2]
        )
        / 2.0

        for box in person_boxes
    ]

    centers_y = [
        (
            box["box"][1]
            +
            box["box"][3]
        )
        / 2.0

        for box in person_boxes
    ]

    center_x = (
        sum(centers_x)
        /
        max(
            1,
            len(centers_x),
        )
    )

    center_y = (
        sum(centers_y)
        /
        max(
            1,
            len(centers_y),
        )
    )

    crop_x1 = int(
        round(
            center_x
            -
            crop_width / 2.0
        )
    )

    crop_y1 = int(
        round(
            center_y
            -
            crop_height / 2.0
        )
    )

    crop_x1 = clamp(
        crop_x1,
        0,
        image_width - crop_width,
    )

    crop_y1 = clamp(
        crop_y1,
        0,
        image_height - crop_height,
    )

    crop_x2 = (
        crop_x1
        +
        crop_width
    )

    crop_y2 = (
        crop_y1
        +
        crop_height
    )

    return (
        int(crop_x1),
        int(crop_y1),
        int(crop_x2),
        int(crop_y2),
    )


# ============================================================================
# BOX INTERSECTION / VISIBILITY
# ============================================================================

def clip_box_to_crop(
    box: List[float],
    crop: Tuple[int, int, int, int],
) -> Tuple[List[float], float]:

    bx1, by1, bx2, by2 = box

    cx1, cy1, cx2, cy2 = crop

    clipped_x1 = max(
        bx1,
        cx1,
    )

    clipped_y1 = max(
        by1,
        cy1,
    )

    clipped_x2 = min(
        bx2,
        cx2,
    )

    clipped_y2 = min(
        by2,
        cy2,
    )

    original_area = (
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

    clipped_area = (
        max(
            0.0,
            clipped_x2 - clipped_x1,
        )
        *
        max(
            0.0,
            clipped_y2 - clipped_y1,
        )
    )

    visibility = safe_div(
        clipped_area,
        original_area,
    )

    return (
        [
            clipped_x1,
            clipped_y1,
            clipped_x2,
            clipped_y2,
        ],
        visibility,
    )


# ============================================================================
# WRITE CROPPED LABEL
# ============================================================================

def box_to_crop_yolo(
    clipped_box: List[float],
    crop: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    x1, y1, x2, y2 = clipped_box

    crop_width = (
        crop_x2
        -
        crop_x1
    )

    crop_height = (
        crop_y2
        -
        crop_y1
    )

    local_x1 = x1 - crop_x1
    local_y1 = y1 - crop_y1

    local_x2 = x2 - crop_x1
    local_y2 = y2 - crop_y1

    local_x1 = clamp(
        local_x1,
        0.0,
        float(crop_width),
    )

    local_y1 = clamp(
        local_y1,
        0.0,
        float(crop_height),
    )

    local_x2 = clamp(
        local_x2,
        0.0,
        float(crop_width),
    )

    local_y2 = clamp(
        local_y2,
        0.0,
        float(crop_height),
    )

    width = (
        local_x2
        -
        local_x1
    )

    height = (
        local_y2
        -
        local_y1
    )

    if width <= 0 or height <= 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    xc = (
        local_x1
        +
        width / 2.0
    ) / crop_width

    yc = (
        local_y1
        +
        height / 2.0
    ) / crop_height

    w = width / crop_width
    h = height / crop_height

    return (
        xc,
        yc,
        w,
        h,
    )


# ============================================================================
# GENERATE DENSE CROPS
# ============================================================================

def generate_dense_crops() -> Dict:

    print()
    print("=" * 72)
    print("GENERANDO CROPS DIRIGIDOS DE ESCENAS DENSAS")
    print("=" * 72)
    print()

    CROPS_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROPS_LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        [
            path
            for path in TRAIN_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:

        raise RuntimeError(
            "No se encontraron imágenes TRAIN."
        )

    dense_images = 0
    generated_crops = 0
    generated_small_person_boxes = 0
    total_person_boxes_dense = 0

    crop_rows = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        label_path = (
            TRAIN_LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        try:

            with Image.open(
                image_path
            ) as image:

                image_width = image.width
                image_height = image.height

                person_boxes = load_all_person_boxes(
                    label_path,
                    image_width,
                    image_height,
                )

                person_count = len(
                    person_boxes
                )

                if not is_dense_scene(
                    person_count
                ):
                    continue

                dense_images += 1
                total_person_boxes_dense += (
                    person_count
                )

                crop = compute_targeted_crop(
                    person_boxes,
                    image_width,
                    image_height,
                )

                crop_image = image.crop(
                    crop
                )

                crop_name = (
                    f"{image_path.stem}"
                    f"__dense_crop.jpg"
                )

                crop_image_path = (
                    CROPS_IMAGES_DIR
                    / crop_name
                )

                crop_image.save(
                    crop_image_path,
                    quality=95,
                )

        except Exception as exc:

            print(
                f"[WARNING] No se pudo procesar "
                f"{image_path.name}: {exc}"
            )

            continue

        crop_x1, crop_y1, crop_x2, crop_y2 = (
            crop
        )

        crop_label_lines = []

        small_boxes_in_crop = 0

        for person in person_boxes:

            clipped_box, visibility = (
                clip_box_to_crop(
                    person["box"],
                    crop,
                )
            )

            if visibility < MIN_BOX_VISIBILITY:

                continue

            xc, yc, w, h = (
                box_to_crop_yolo(
                    clipped_box,
                    crop,
                )
            )

            if w <= 0 or h <= 0:

                continue

            crop_area = (
                w
                * h
                * (
                    crop_x2
                    -
                    crop_x1
                )
                * (
                    crop_y2
                    -
                    crop_y1
                )
            )

            if (
                crop_area
                <
                SMALL_AREA_THRESHOLD
            ):

                small_boxes_in_crop += 1

            crop_label_lines.append(
                f"{PERSON_CLASS_ID} "
                f"{xc:.8f} "
                f"{yc:.8f} "
                f"{w:.8f} "
                f"{h:.8f}"
            )

        if not crop_label_lines:

            try:

                crop_image_path.unlink(
                    missing_ok=True
                )

            except Exception:
                pass

            continue

        crop_label_path = (
            CROPS_LABELS_DIR
            / (
                f"{image_path.stem}"
                f"__dense_crop.txt"
            )
        )

        crop_label_path.write_text(
            "\n".join(
                crop_label_lines
            )
            +
            "\n",
            encoding="utf-8",
        )

        generated_crops += 1
        generated_small_person_boxes += (
            small_boxes_in_crop
        )

        crop_rows.append(
            {
                "source_image":
                    image_path.name,

                "source_person_count":
                    person_count,

                "crop_image":
                    crop_name,

                "crop_width":
                    crop_x2 - crop_x1,

                "crop_height":
                    crop_y2 - crop_y1,

                "crop_x1":
                    crop_x1,

                "crop_y1":
                    crop_y1,

                "crop_x2":
                    crop_x2,

                "crop_y2":
                    crop_y2,

                "persons_in_crop":
                    len(
                        crop_label_lines
                    ),

                "small_persons_in_crop":
                    small_boxes_in_crop,
            }
        )

        if (
            index % 500 == 0
            or index == len(image_files)
        ):

            print(
                f"Procesadas: "
                f"{index:,}/{len(image_files):,} "
                f"| Densas: "
                f"{dense_images:,} "
                f"| Crops: "
                f"{generated_crops:,}"
            )

    print()
    print(
        f"TRAIN images:             {len(image_files):,}"
    )

    print(
        f"DENSE images:              {dense_images:,}"
    )

    print(
        f"Generated crops:           {generated_crops:,}"
    )

    print(
        f"Persons in dense scenes:   "
        f"{total_person_boxes_dense:,}"
    )

    print(
        f"Small persons in crops:    "
        f"{generated_small_person_boxes:,}"
    )

    if not crop_rows:

        raise RuntimeError(
            "No se generó ningún crop denso."
        )

    write_csv(
        CROP_STATS_CSV,
        crop_rows,
    )

    return {
        "train_images":
            len(image_files),

        "dense_images":
            dense_images,

        "generated_crops":
            generated_crops,

        "persons_dense":
            total_person_boxes_dense,

        "small_persons_cropped":
            generated_small_person_boxes,
    }


# ============================================================================
# MANIFEST
# ============================================================================

def create_train_manifest() -> int:

    print()
    print(
        "CONSTRUYENDO MANIFEST EXP04"
    )

    original_images = sorted(
        [
            path
            for path in TRAIN_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    crop_images = sorted(
        [
            path
            for path in CROPS_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    lines = [
        str(
            path.resolve()
        )
        for path in original_images
    ]

    lines.extend(
        str(
            path.resolve()
        )
        for path in crop_images
    )

    if not lines:

        raise RuntimeError(
            "El manifest EXP04 quedó vacío."
        )

    TRAIN_MANIFEST.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[OK] Original images: "
        f"{len(original_images):,}"
    )

    print(
        f"[OK] Dense crops: "
        f"{len(crop_images):,}"
    )

    print(
        f"[OK] Manifest total: "
        f"{len(lines):,}"
    )

    print(
        f"[OK] Manifest:\n"
        f"     {TRAIN_MANIFEST}"
    )

    return len(lines)


# ============================================================================
# YAML EXP04
# ============================================================================

def create_experiment_yaml() -> None:

    content = f"""path: {DATASET_ROOT.as_posix()}

train: {TRAIN_MANIFEST.as_posix()}
val: {VAL_IMAGES_DIR.as_posix()}
test: {TEST_IMAGES_DIR.as_posix()}

names:
  0: person
  1: vehicle
"""

    TEMP_DATA_YAML.write_text(
        content,
        encoding="utf-8",
    )

    print()
    print(
        "[OK] YAML EXP04 generado:"
    )

    print(
        f"     {TEMP_DATA_YAML}"
    )

    print(
        "[INFO] El YAML oficial NO ha sido modificado."
    )


# ============================================================================
# TRAINING CONFIG
# ============================================================================

def write_training_configuration(
    model_path: Path,
    manifest_count: int,
    crop_statistics: Dict,
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
                "dense_person_threshold",

            "value":
                DENSE_PERSON_THRESHOLD,
        },

        {
            "parameter":
                "crop_width_ratio",

            "value":
                CROP_WIDTH_RATIO,
        },

        {
            "parameter":
                "crop_height_ratio",

            "value":
                CROP_HEIGHT_RATIO,
        },

        {
            "parameter":
                "minimum_box_visibility",

            "value":
                MIN_BOX_VISIBILITY,
        },

        {
            "parameter":
                "manifest_images",

            "value":
                manifest_count,
        },

        {
            "parameter":
                "dense_images",

            "value":
                crop_statistics[
                    "dense_images"
                ],
        },

        {
            "parameter":
                "generated_crops",

            "value":
                crop_statistics[
                    "generated_crops"
                ],
        },

        {
            "parameter":
                "small_persons_in_crops",

            "value":
                crop_statistics[
                    "small_persons_cropped"
                ],
        },
    ]

    write_csv(
        TRAIN_CONFIG_CSV,
        rows,
    )


# ============================================================================
# TRAIN
# ============================================================================

def train_exp04(
    model_path: Path,
) -> Path:

    print()
    print("=" * 72)
    print("ENTRENAMIENTO EXP04")
    print("=" * 72)
    print()

    print(
        f"Modelo:          {model_path}"
    )

    print(
        f"Train imgsz:     {TRAIN_IMAGE_SIZE}"
    )

    print(
        f"Epochs:          {EPOCHS}"
    )

    print(
        f"Batch:           {BATCH}"
    )

    print(
        f"Workers:         {WORKERS}"
    )

    print(
        f"Seed:            {SEED}"
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

            name="exp04_dense_scene_targeted_crops",

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
            or
            "cuda out of memory"
            in error_text
        ):

            raise RuntimeError(
                "\nEXP04 no pudo entrenar por falta de VRAM.\n\n"
                "La configuración no se modificó automáticamente "
                "para mantener el experimento controlado.\n"
                f"imgsz={TRAIN_IMAGE_SIZE}\n"
                f"batch={BATCH}\n"
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
            "No se encontró best.pt después de EXP04:\n"
            f"{best_path}"
        )

    return best_path


# ============================================================================
# GROUND TRUTH PARA TEST
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

        parts = line.strip().split()

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

                best_iou = current_iou
                best_index = prediction_index

        matched = (
            best_index is not None
            and
            best_iou >= EVAL_MATCH_IOU
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
# TEST IMAGE
# ============================================================================

def process_test_image(
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

            width = image.width
            height = image.height

    except Exception as exc:

        print(
            f"[WARNING] No se pudo abrir "
            f"{image_path.name}: {exc}"
        )

        return []

    gt_objects = load_small_person_gt(
        label_path,
        width,
        height,
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
            f"[ERROR] Inferencia fallida: "
            f"{image_path.name} | {exc}"
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

    matches = match_gt_predictions(
        gt_objects,
        prediction_boxes,
        prediction_confidences,
    )

    rows = []

    for gt, match in zip(
        gt_objects,
        matches,
    ):

        size_sqrt = gt["size_sqrt"]

        if size_sqrt < 16:
            bucket = "<16"
        elif size_sqrt < 32:
            bucket = "16-32"
        elif size_sqrt < 64:
            bucket = "32-64"
        elif size_sqrt < 128:
            bucket = "64-128"
        elif size_sqrt < 256:
            bucket = "128-256"
        else:
            bucket = ">=256"

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
                        size_sqrt,
                        6,
                    ),

                "size_bucket":
                    bucket,

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

    fn = gt - tp

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
# CSV
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
# COMPARACIÓN
# ============================================================================

def write_comparison(
    exp04_metrics: Dict,
) -> None:

    references = [

        {
            "experiment":
                "EXP01",

            "train_imgsz":
                640,

            "intervention":
                "baseline",

            "gt":
                17879,

            "tp":
                5306,

            "fn":
                12573,

            "recall_percentage":
                29.68,
        },

        {
            "experiment":
                "EXP02",

            "train_imgsz":
                640,

            "intervention":
                "oversampling_2x",

            "gt":
                17879,

            "tp":
                5295,

            "fn":
                12584,

            "recall_percentage":
                29.62,
        },

        {
            "experiment":
                "EXP03",

            "train_imgsz":
                960,

            "intervention":
                "high_resolution",

            "gt":
                17879,

            "tp":
                5422,

            "fn":
                12457,

            "recall_percentage":
                30.33,
        },

        {
            "experiment":
                "EXP04",

            "train_imgsz":
                960,

            "intervention":
                "dense_scene_targeted_crops",

            "gt":
                exp04_metrics["gt"],

            "tp":
                exp04_metrics["tp"],

            "fn":
                exp04_metrics["fn"],

            "recall_percentage":
                exp04_metrics[
                    "recall_percentage"
                ],
        },
    ]

    for row in references:

        row["delta_vs_exp01_pp"] = (
            row["recall_percentage"]
            -
            29.68
        )

        row["delta_vs_exp03_pp"] = (
            row["recall_percentage"]
            -
            30.33
        )

    write_csv(
        COMPARISON_CSV,
        references,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def write_summary(
    crop_statistics: Dict,
    metrics: Dict,
    size_metrics: List[Dict],
    best_path: Path,
) -> None:

    delta_exp01 = (
        metrics["recall_percentage"]
        -
        29.68
    )

    delta_exp03 = (
        metrics["recall_percentage"]
        -
        30.33
    )

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "SAR YOLO26 - EXP04 DENSE SCENE TARGETED CROPS V1"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "HIPÓTESIS"
    )

    lines.append(
        "Las personas pequeñas en escenas densas necesitan "
        "exposición adicional a regiones recortadas donde "
        "la escala efectiva de las personas aumenta."
    )

    lines.append("")

    lines.append(
        "CRITERIO DENSE_SCENE"
    )

    lines.append(
        f"PERSON por imagen >= {DENSE_PERSON_THRESHOLD}"
    )

    lines.append("")

    lines.append(
        "CONFIGURACIÓN"
    )

    lines.append(
        f"Train imgsz: {TRAIN_IMAGE_SIZE}"
    )

    lines.append(
        f"Epochs:      {EPOCHS}"
    )

    lines.append(
        f"Batch:       {BATCH}"
    )

    lines.append(
        f"Workers:     {WORKERS}"
    )

    lines.append(
        f"Seed:        {SEED}"
    )

    lines.append(
        f"Crop width:  {CROP_WIDTH_RATIO:.2f}"
    )

    lines.append(
        f"Crop height: {CROP_HEIGHT_RATIO:.2f}"
    )

    lines.append(
        f"Min visibility: {MIN_BOX_VISIBILITY:.2f}"
    )

    lines.append("")

    lines.append(
        "CROPS"
    )

    lines.append(
        f"Train images:             "
        f"{crop_statistics['train_images']:,}"
    )

    lines.append(
        f"Dense images:             "
        f"{crop_statistics['dense_images']:,}"
    )

    lines.append(
        f"Generated crops:          "
        f"{crop_statistics['generated_crops']:,}"
    )

    lines.append(
        f"Persons in dense scenes:  "
        f"{crop_statistics['persons_dense']:,}"
    )

    lines.append(
        f"Small persons in crops:   "
        f"{crop_statistics['small_persons_cropped']:,}"
    )

    lines.append("")

    lines.append(
        "RESULTADO EXP04"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"SMALL PERSON GT:     "
        f"{metrics['gt']:,}"
    )

    lines.append(
        f"SMALL PERSON TP:     "
        f"{metrics['tp']:,}"
    )

    lines.append(
        f"SMALL PERSON FN:     "
        f"{metrics['fn']:,}"
    )

    lines.append(
        f"SMALL PERSON Recall: "
        f"{metrics['recall_percentage']:.2f}%"
    )

    lines.append("")

    lines.append(
        "COMPARACIÓN"
    )

    lines.append(
        f"EXP01: {29.68:.2f}%"
    )

    lines.append(
        f"EXP02: {29.62:.2f}%"
    )

    lines.append(
        f"EXP03: {30.33:.2f}%"
    )

    lines.append(
        f"EXP04: "
        f"{metrics['recall_percentage']:.2f}%"
    )

    lines.append(
        f"EXP04 - EXP01: "
        f"{delta_exp01:+.2f} pp"
    )

    lines.append(
        f"EXP04 - EXP03: "
        f"{delta_exp03:+.2f} pp"
    )

    lines.append("")

    if delta_exp03 > 1.0:

        interpretation = (
            "MEJORA FUERTE: los crops dirigidos a escenas "
            "densas aportan una mejora adicional clara sobre EXP03."
        )

    elif delta_exp03 > 0.5:

        interpretation = (
            "MEJORA MODERADA: los crops densos muestran "
            "una señal positiva adicional sobre EXP03."
        )

    elif delta_exp03 >= -0.5:

        interpretation = (
            "SIN CAMBIO RELEVANTE: la intervención "
            "de crops densos no cambia apreciablemente "
            "el SMALL PERSON Recall respecto a EXP03."
        )

    else:

        interpretation = (
            "EMPEORAMIENTO: los crops densos reducen "
            "el SMALL PERSON Recall respecto a EXP03."
        )

    lines.append(
        "INTERPRETACIÓN"
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

    if delta_exp03 > 0.5:

        lines.append(
            "La intervención DENSE_SCENE_TARGETED_CROPS "
            "aporta evidencia positiva."
        )

        lines.append(
            "Debe conservarse para la fase combinada."
        )

    else:

        lines.append(
            "La intervención no demuestra una mejora "
            "suficiente sobre EXP03."
        )

        lines.append(
            "No debe combinarse todavía en EXP07."
        )

    lines.append("")

    lines.append(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    lines.append(
        "Los crops se han generado únicamente dentro del "
        "directorio experimental de EXP04."
    )

    lines.append(
        "El YAML oficial NO ha sido modificado."
    )

    lines.append("")

    lines.append(
        "MODELO BEST"
    )

    lines.append(
        str(best_path)
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
        "# SAR YOLO26 - EXP04 DENSE SCENE TARGETED CROPS V1"
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
        "DENSE PERSON THRESHOLD:"
    )

    print(
        f"  >= {DENSE_PERSON_THRESHOLD}"
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
    # CROP GENERATION
    # ------------------------------------------------------------------------

    crop_statistics = (
        generate_dense_crops()
    )

    # ------------------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------------------

    manifest_count = (
        create_train_manifest()
    )

    # ------------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------------

    create_experiment_yaml()

    # ------------------------------------------------------------------------
    # CONFIG REPORT
    # ------------------------------------------------------------------------

    write_training_configuration(
        model_path,
        manifest_count,
        crop_statistics,
    )

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    best_path = train_exp04(
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
    # LOAD MODEL
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "EVALUACIÓN SMALL PERSON EXP04"
    )

    print(
        "=" * 72
    )

    model = YOLO(
        str(best_path)
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
            "No hay imágenes TEST_DEV."
        )

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

        rows = process_test_image(
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
        EVAL_CSV,
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
                    metrics["recall_percentage"],
            },
            {
                "metric":
                    "TRAIN_IMAGE_SIZE",
                "value":
                    TRAIN_IMAGE_SIZE,
            },
            {
                "metric":
                    "DENSE_PERSON_THRESHOLD",
                "value":
                    DENSE_PERSON_THRESHOLD,
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
        ],
    )

    write_comparison(
        metrics
    )

    write_summary(
        crop_statistics,
        metrics,
        size_metrics,
        best_path,
    )

    # ------------------------------------------------------------------------
    # CONSOLE
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "# RESULTADO EXP04"
    )

    print("=" * 72)

    print()

    print(
        f"Dense images:            "
        f"{crop_statistics['dense_images']:,}"
    )

    print(
        f"Generated crops:         "
        f"{crop_statistics['generated_crops']:,}"
    )

    print(
        f"Small persons in crops:  "
        f"{crop_statistics['small_persons_cropped']:,}"
    )

    print()

    print(
        f"SMALL PERSON GT:         "
        f"{metrics['gt']:,}"
    )

    print(
        f"SMALL PERSON TP:         "
        f"{metrics['tp']:,}"
    )

    print(
        f"SMALL PERSON FN:         "
        f"{metrics['fn']:,}"
    )

    print(
        f"SMALL PERSON Recall:     "
        f"{metrics['recall_percentage']:.2f}%"
    )

    print()

    print(
        "COMPARACIÓN"
    )

    print(
        "EXP01: 29.68%"
    )

    print(
        "EXP02: 29.62%"
    )

    print(
        "EXP03: 30.33%"
    )

    print(
        f"EXP04: "
        f"{metrics['recall_percentage']:.2f}%"
    )

    print()

    print(
        "REPORTS:"
    )

    print(
        f"[OK] {CROP_STATS_CSV}"
    )

    print(
        f"[OK] {TRAIN_CONFIG_CSV}"
    )

    print(
        f"[OK] {EVAL_CSV}"
    )

    print(
        f"[OK] {SIZE_CSV}"
    )

    print(
        f"[OK] {OBJECTS_CSV}"
    )

    print(
        f"[OK] {COMPARISON_CSV}"
    )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    print()

    print(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    print(
        "IMPORTANTE: los crops existen únicamente dentro de EXP04."
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
            "[CANCELADO] EXP04 interrumpido."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)

        print(
            "[ERROR EXP04]"
        )

        print(
            "=" * 72
        )

        print()

        print(
            str(exc)
        )

        print()

        sys.exit(1)