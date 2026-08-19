from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26
# EXP01 - SMALL PERSON RECALL BASELINE V1
# ============================================================================
#
# OBJETIVO
# --------
# Medir de forma reproducible el recall de PERSON pequeñas sobre TEST_DEV
# utilizando el modelo baseline ya entrenado.
#
# REGLAS
# ------
# - NO modifica el dataset.
# - NO modifica labels.
# - NO modifica imágenes.
# - NO modifica pesos.
# - NO crea ningún YAML.
# - Utiliza directamente el dataset V1 ya preparado.
# - Utiliza TEST_DEV, que es el conjunto independiente.
# - INPUT SIZE = 1536.
# - IoU MATCH = 0.50.
# - PERSON = class 0.
# - SMALL PERSON = area < 256 px².
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN EXPERIMENTAL
# ============================================================================

PERSON_CLASS_ID = 0
VEHICLE_CLASS_ID = 1

IMAGE_SIZE = 1536
CONF_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.50

SMALL_AREA_THRESHOLD = 256.0

DEVICE = "0"

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
# LOCALIZACIÓN DEL PROYECTO
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root() -> Path:
    """
    Localiza C:\\SARC-Drone buscando el directorio cuyo nombre sea
    'SARC-Drone' a partir de la ubicación del script.
    """

    for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents]:

        if directory.name.lower() == "sarc-drone":
            return directory

    raise RuntimeError(
        "No se pudo localizar la raíz del proyecto SARC-Drone.\n"
        f"Script:\n{SCRIPT_PATH}"
    )


PROJECT_ROOT = find_project_root()


# ============================================================================
# RUTAS REALES DEL PROYECTO
# ============================================================================

BASELINE_DIR = (
    PROJECT_ROOT
    / "01_training"
    / "experiments"
    / "sar_yolo26"
    / "baseline"
)


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


MODEL_PATH = (
    BASELINE_DIR
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)


# Report generado por el análisis previo.
PATTERNS_CSV = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "analyze_person_small_failure_patterns_v1"
    / "reports"
    / "person_small_failure_patterns_objects_v1.csv"
)


# Directorio de salida de EXP01.
OUTPUT_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "reports"
)


# ============================================================================
# OUTPUTS
# ============================================================================

OBJECTS_CSV = (
    OUTPUT_DIR
    / "exp01_small_person_objects_v1.csv"
)


SUMMARY_CSV = (
    OUTPUT_DIR
    / "exp01_small_person_recall_baseline_v1.csv"
)


SIZE_CSV = (
    OUTPUT_DIR
    / "exp01_small_person_recall_by_size_v1.csv"
)


DENSITY_CSV = (
    OUTPUT_DIR
    / "exp01_small_person_recall_by_density_v1.csv"
)


PATTERN_CSV = (
    OUTPUT_DIR
    / "exp01_small_person_recall_by_pattern_v1.csv"
)


SUMMARY_TXT = (
    OUTPUT_DIR
    / "EXP01_SMALL_PERSON_RECALL_BASELINE_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_float(value, default=0.0) -> float:

    try:

        if value is None:
            return default

        value = str(value).strip()

        if not value:
            return default

        result = float(value)

        if math.isnan(result):
            return default

        return result

    except (TypeError, ValueError):

        return default


def safe_int(value, default=0) -> int:

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def pct(value: float, total: float) -> float:

    if total == 0:
        return 0.0

    return 100.0 * value / total


def recall(tp: int, gt: int) -> float:

    if gt == 0:
        return 0.0

    return tp / gt


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

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

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

    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)

    x2 = min(
        float(image_width),
        cx + bw / 2.0,
    )

    y2 = min(
        float(image_height),
        cy + bh / 2.0,
    )

    return [x1, y1, x2, y2]


# ============================================================================
# VALIDACIÓN DE ESTRUCTURA
# ============================================================================

def validate_paths() -> None:

    print()
    print("=" * 72)
    print("VALIDANDO ESTRUCTURA")
    print("=" * 72)
    print()

    checks = [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("BASELINE_DIR", BASELINE_DIR),
        ("DATASET_ROOT", DATASET_ROOT),
        ("TEST_IMAGES_DIR", TEST_IMAGES_DIR),
        ("TEST_LABELS_DIR", TEST_LABELS_DIR),
        ("MODEL_PATH", MODEL_PATH),
        ("PATTERNS_CSV", PATTERNS_CSV),
    ]

    for name, path in checks:

        if path.exists():

            print(f"[OK] {name}")
            print(f"     {path}")

        else:

            print(f"[ERROR] {name}")
            print(f"        {path}")

            raise FileNotFoundError(
                f"No se encontró {name}:\n{path}"
            )

    if not TEST_IMAGES_DIR.is_dir():

        raise RuntimeError(
            f"No es un directorio de imágenes:\n"
            f"{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.is_dir():

        raise RuntimeError(
            f"No es un directorio de labels:\n"
            f"{TEST_LABELS_DIR}"
        )


# ============================================================================
# VALIDACIÓN DEL MODELO
# ============================================================================

def validate_model_names(
    model: YOLO,
) -> None:

    names = model.names

    print()
    print("CLASES DEL MODELO")
    print("-" * 72)

    print(names)

    if isinstance(names, dict):

        person_name = str(
            names.get(PERSON_CLASS_ID, "")
        ).lower()

        vehicle_name = str(
            names.get(VEHICLE_CLASS_ID, "")
        ).lower()

        if "person" not in person_name:

            raise ValueError(
                "El modelo no parece tener 'person' como clase 0.\n"
                f"Modelo names={names}"
            )

        if "vehicle" not in vehicle_name:

            print(
                "[WARNING] La clase 1 no contiene "
                f"'vehicle': {names}"
            )

    print("[OK] Clases compatibles con EXP01.")


# ============================================================================
# CARGA DE PATTERNS
# ============================================================================

def load_patterns() -> Dict[Tuple[str, str], Dict]:

    print()
    print("Cargando PATTERNS report...")

    if not PATTERNS_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró:\n{PATTERNS_CSV}"
        )

    patterns = {}

    with PATTERNS_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                "El PATTERNS CSV no tiene cabecera."
            )

        required = {
            "image",
            "person_gt_index",
            "area",
            "size_bucket",
            "density_bucket",
            "location",
            "border",
            "occlusion_bucket",
            "nearest_person_proximity",
            "prediction_relation",
            "dominant_pattern",
        }

        missing = required - set(
            reader.fieldnames
        )

        if missing:

            raise ValueError(
                "Faltan columnas en PATTERNS CSV:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in sorted(missing)
                )
            )

        for row in reader:

            key = (
                row["image"],
                row["person_gt_index"],
            )

            patterns[key] = row

    print(
        f"[OK] PATTERNS cargados: "
        f"{len(patterns):,}"
    )

    return patterns


# ============================================================================
# CARGA DE GROUND TRUTH
# ============================================================================

def load_person_small_gt(
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

        # Solo PERSON.
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

        # Solo PERSON pequeños.
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
                "size_sqrt": math.sqrt(area),
            }
        )

    return objects


# ============================================================================
# MATCHING
# ============================================================================

def match_gt_to_predictions(
    gt_objects: List[Dict],
    prediction_boxes: List[List[float]],
    prediction_confidences: List[float],
) -> List[Dict]:

    results = []

    used_predictions = set()

    # ------------------------------------------------------------
    # GT por GT, utilizando la mejor predicción disponible.
    # ------------------------------------------------------------

    for gt in gt_objects:

        best_iou = 0.0
        best_prediction_index = None

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
                best_prediction_index = (
                    prediction_index
                )

        matched = (
            best_prediction_index is not None
            and best_iou >= MATCH_IOU_THRESHOLD
        )

        if matched:
            used_predictions.add(
                best_prediction_index
            )

        confidence = 0.0

        if matched:
            confidence = prediction_confidences[
                best_prediction_index
            ]

        results.append(
            {
                "matched": matched,
                "iou": best_iou,
                "confidence": confidence,
            }
        )

    return results


# ============================================================================
# PROCESAR UNA IMAGEN
# ============================================================================

def process_image(
    model: YOLO,
    image_path: Path,
    patterns: Dict[Tuple[str, str], Dict],
) -> List[Dict]:

    label_path = (
        TEST_LABELS_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():

        return []

    try:

        with Image.open(image_path) as image:

            image_width = image.width
            image_height = image.height

    except Exception as exc:

        print(
            f"[WARNING] Imagen no legible: "
            f"{image_path.name} | {exc}"
        )

        return []

    gt_objects = load_person_small_gt(
        label_path,
        image_width,
        image_height,
    )

    if not gt_objects:

        return []

    # ------------------------------------------------------------
    # INFERENCIA YOLO
    # ------------------------------------------------------------

    try:

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
            save=False,
        )

    except Exception as exc:

        print(
            f"[ERROR] Inferencia falló en "
            f"{image_path.name}: {exc}"
        )

        return []

    if not results:

        return []

    result = results[0]

    prediction_boxes = []
    prediction_confidences = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()

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

        key = (
            image_path.name,
            str(gt["gt_index"]),
        )

        pattern = patterns.get(
            key,
            {},
        )

        output.append(
            {
                "image": image_path.name,

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

                "status":
                    "TP"
                    if match["matched"]
                    else "FN",

                "matched":
                    int(match["matched"]),

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

                "size_bucket":
                    pattern.get(
                        "size_bucket",
                        "",
                    ),

                "density_bucket":
                    pattern.get(
                        "density_bucket",
                        "",
                    ),

                "location":
                    pattern.get(
                        "location",
                        "",
                    ),

                "border":
                    pattern.get(
                        "border",
                        "",
                    ),

                "occlusion_bucket":
                    pattern.get(
                        "occlusion_bucket",
                        "",
                    ),

                "nearest_person_proximity":
                    pattern.get(
                        "nearest_person_proximity",
                        "",
                    ),

                "prediction_relation":
                    pattern.get(
                        "prediction_relation",
                        "",
                    ),

                "dominant_pattern":
                    pattern.get(
                        "dominant_pattern",
                        "",
                    ),
            }
        )

    return output


# ============================================================================
# GENERAL METRICS
# ============================================================================

def calculate_general_metrics(
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
        "recall": recall(tp, gt),
        "recall_percentage": pct(tp, gt),
    }


# ============================================================================
# DESGLOSE POR TAMAÑO
# ============================================================================

SIZE_BUCKETS = [
    ("<16", 0, 16),
    ("16-32", 16, 32),
    ("32-64", 32, 64),
    ("64-128", 64, 128),
    ("128-256", 128, 256),
]


def get_size_bucket(
    size_sqrt: float,
) -> str:

    for name, low, high in SIZE_BUCKETS:

        if low <= size_sqrt < high:
            return name

    return "UNKNOWN"


def calculate_size_metrics(
    rows: List[Dict],
) -> List[Dict]:

    groups = {
        name: []
        for name, _, _ in SIZE_BUCKETS
    }

    for row in rows:

        bucket = get_size_bucket(
            safe_float(
                row["size_sqrt"]
            )
        )

        if bucket in groups:

            groups[bucket].append(
                row
            )

    output = []

    for name, _, _ in SIZE_BUCKETS:

        group = groups[name]

        metrics = calculate_general_metrics(
            group
        )

        output.append(
            {
                "size_bucket": name,
                **metrics,
            }
        )

    return output


# ============================================================================
# GENERIC GROUP METRICS
# ============================================================================

def calculate_grouped_metrics(
    rows: List[Dict],
    field: str,
) -> List[Dict]:

    groups = defaultdict(list)

    for row in rows:

        value = str(
            row.get(field, "")
        ).strip()

        if not value:
            value = "UNKNOWN"

        groups[value].append(row)

    output = []

    for value, group in groups.items():

        metrics = calculate_general_metrics(
            group
        )

        output.append(
            {
                field: value,
                **metrics,
            }
        )

    output.sort(
        key=lambda item: (
            -item["fn"],
            -item["gt"],
        )
    )

    return output


# ============================================================================
# SAVE CSV
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
# SUMMARY
# ============================================================================

def generate_summary(
    metrics: Dict,
    size_metrics: List[Dict],
    density_metrics: List[Dict],
    pattern_metrics: List[Dict],
    image_count: int,
) -> None:

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "SAR YOLO26 - EXP01 SMALL PERSON RECALL BASELINE V1"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "CONFIGURACIÓN"
    )

    lines.append(
        f"Model: {MODEL_PATH}"
    )

    lines.append(
        f"Dataset: {DATASET_ROOT}"
    )

    lines.append(
        f"Split: test_dev"
    )

    lines.append(
        f"Images: {TEST_IMAGES_DIR}"
    )

    lines.append(
        f"Labels: {TEST_LABELS_DIR}"
    )

    lines.append(
        f"Input scale: {IMAGE_SIZE}"
    )

    lines.append(
        f"Confidence threshold: {CONF_THRESHOLD}"
    )

    lines.append(
        f"Match IoU threshold: {MATCH_IOU_THRESHOLD}"
    )

    lines.append(
        f"Small area threshold: < {SMALL_AREA_THRESHOLD} px²"
    )

    lines.append("")

    lines.append(
        "RESULTADO GLOBAL SMALL PERSON"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"Imágenes con SMALL PERSON: {image_count:,}"
    )

    lines.append(
        f"GT:                    {metrics['gt']:,}"
    )

    lines.append(
        f"TP:                    {metrics['tp']:,}"
    )

    lines.append(
        f"FN:                    {metrics['fn']:,}"
    )

    lines.append(
        f"Recall:                {metrics['recall']:.6f}"
    )

    lines.append(
        f"Recall (%):            {metrics['recall_percentage']:.2f}%"
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
        "TOP POR DENSIDAD"
    )

    lines.append(
        "-" * 72
    )

    for row in density_metrics[:10]:

        lines.append(
            f"{row['density_bucket']:<15} "
            f"GT={row['gt']:>6,} "
            f"TP={row['tp']:>6,} "
            f"FN={row['fn']:>6,} "
            f"Recall={row['recall_percentage']:>7.2f}%"
        )

    lines.append("")

    lines.append(
        "TOP PATRONES DE FALLO"
    )

    lines.append(
        "-" * 72
    )

    for row in pattern_metrics[:15]:

        lines.append(
            f"{row['dominant_pattern']:<35} "
            f"GT={row['gt']:>6,} "
            f"TP={row['tp']:>6,} "
            f"FN={row['fn']:>6,} "
            f"Recall={row['recall_percentage']:>7.2f}%"
        )

    lines.append("")

    lines.append(
        "REFERENCIA DEL ANÁLISIS PREVIO"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "A 1536 px, el análisis previo sobre TEST_DEV "
        "registró 17,182 TP y 9,577 FN para PERSON global."
    )

    lines.append(
        "Para SMALL PERSON, el análisis residual registró "
        "9,564 TP y 8,315 FN, recall 53.49%."
    )

    lines.append(
        "Este EXP01 se ejecuta de nuevo desde el modelo para "
        "comprobar/reproducir el baseline de forma independiente."
    )

    lines.append("")

    lines.append(
        "IMPORTANTE"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "El dataset original NO ha sido modificado."
    )

    lines.append(
        "Ningún YAML ha sido creado o modificado por este experimento."
    )

    lines.append("")

    lines.append(
        "SIGUIENTE EXPERIMENTO"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "EXP02 - TARGETED_SMALL_PERSON_OVERSAMPLING"
    )

    lines.append(
        "No iniciar EXP02 hasta revisar el baseline EXP01."
    )

    lines.append(
        ""
    )

    lines.append(
        "=" * 72
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
        "# SAR YOLO26 - EXP01 SMALL PERSON RECALL BASELINE V1"
    )
    print("=" * 72)

    print()
    print("SCRIPT:")
    print(f"  {SCRIPT_PATH}")

    print()
    print("PROJECT ROOT:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("BASELINE:")
    print(f"  {BASELINE_DIR}")

    print()
    print("DATASET:")
    print(f"  {DATASET_ROOT}")

    print()
    print("MODEL:")
    print(f"  {MODEL_PATH}")

    print()
    print("TEST:")
    print(f"  {TEST_IMAGES_DIR}")
    print(f"  {TEST_LABELS_DIR}")

    print()

    # ========================================================================
    # VALIDAR RUTAS
    # ========================================================================

    validate_paths()

    # ========================================================================
    # CREAR OUTPUT
    # ========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("OUTPUT:")
    print(f"  {OUTPUT_DIR}")

    # ========================================================================
    # CARGAR PATTERNS
    # ========================================================================

    patterns = load_patterns()

    # ========================================================================
    # CARGAR MODELO
    # ========================================================================

    print()
    print("=" * 72)
    print("CARGANDO MODELO")
    print("=" * 72)

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "[OK] Modelo cargado."
    )

    validate_model_names(
        model
    )

    # ========================================================================
    # LISTAR IMÁGENES TEST
    # ========================================================================

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
            f"No se encontraron imágenes en:\n"
            f"{TEST_IMAGES_DIR}"
        )

    print()
    print(
        f"[OK] Imágenes TEST_DEV: "
        f"{len(image_files):,}"
    )

    # ========================================================================
    # PROCESAMIENTO
    # ========================================================================

    print()
    print("=" * 72)
    print("ANALIZANDO SMALL PERSON")
    print("=" * 72)
    print()

    all_rows = []

    images_with_small_person = 0
    images_processed = 0

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        rows = process_image(
            model=model,
            image_path=image_path,
            patterns=patterns,
        )

        images_processed += 1

        if rows:
            images_with_small_person += 1
            all_rows.extend(rows)

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
            "No se encontraron PERSON pequeñas "
            "en TEST_DEV."
        )

    # ========================================================================
    # MÉTRICAS
    # ========================================================================

    metrics = calculate_general_metrics(
        all_rows
    )

    size_metrics = calculate_size_metrics(
        all_rows
    )

    density_metrics = calculate_grouped_metrics(
        all_rows,
        "density_bucket",
    )

    pattern_metrics = calculate_grouped_metrics(
        all_rows,
        "dominant_pattern",
    )

    # ========================================================================
    # REPORT OBJECTOS
    # ========================================================================

    write_csv(
        OBJECTS_CSV,
        all_rows,
    )

    # ========================================================================
    # REPORT RESUMEN NUMÉRICO
    # ========================================================================

    summary_rows = [
        {
            "experiment": "EXP01",
            "metric": "SMALL_PERSON_GT",
            "value": metrics["gt"],
        },
        {
            "experiment": "EXP01",
            "metric": "SMALL_PERSON_TP",
            "value": metrics["tp"],
        },
        {
            "experiment": "EXP01",
            "metric": "SMALL_PERSON_FN",
            "value": metrics["fn"],
        },
        {
            "experiment": "EXP01",
            "metric": "SMALL_PERSON_RECALL",
            "value": metrics["recall"],
        },
        {
            "experiment": "EXP01",
            "metric": "SMALL_PERSON_RECALL_PERCENTAGE",
            "value": metrics["recall_percentage"],
        },
        {
            "experiment": "EXP01",
            "metric": "IMAGE_SIZE",
            "value": IMAGE_SIZE,
        },
        {
            "experiment": "EXP01",
            "metric": "MATCH_IOU_THRESHOLD",
            "value": MATCH_IOU_THRESHOLD,
        },
        {
            "experiment": "EXP01",
            "metric": "CONF_THRESHOLD",
            "value": CONF_THRESHOLD,
        },
    ]

    write_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    # ========================================================================
    # SIZE REPORT
    # ========================================================================

    write_csv(
        SIZE_CSV,
        size_metrics,
    )

    # ========================================================================
    # DENSITY REPORT
    # ========================================================================

    write_csv(
        DENSITY_CSV,
        density_metrics,
    )

    # ========================================================================
    # PATTERN REPORT
    # ========================================================================

    write_csv(
        PATTERN_CSV,
        pattern_metrics,
    )

    # ========================================================================
    # SUMMARY TXT
    # ========================================================================

    generate_summary(
        metrics=metrics,
        size_metrics=size_metrics,
        density_metrics=density_metrics,
        pattern_metrics=pattern_metrics,
        image_count=images_with_small_person,
    )

    # ========================================================================
    # CONSOLA
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "# RESULTADO EXP01 SMALL PERSON RECALL BASELINE V1"
    )
    print("=" * 72)

    print()
    print(
        f"Imágenes procesadas:       "
        f"{images_processed:,}"
    )

    print(
        f"Imágenes con small person: "
        f"{images_with_small_person:,}"
    )

    print(
        f"SMALL PERSON GT:            "
        f"{metrics['gt']:,}"
    )

    print(
        f"SMALL PERSON TP:            "
        f"{metrics['tp']:,}"
    )

    print(
        f"SMALL PERSON FN:            "
        f"{metrics['fn']:,}"
    )

    print(
        f"SMALL PERSON Recall:        "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"SMALL PERSON Recall (%):    "
        f"{metrics['recall_percentage']:.2f}%"
    )

    print()
    print(
        "DESGLOSE POR TAMAÑO"
    )
    print("-" * 72)

    for row in size_metrics:

        print(
            f"{row['size_bucket']:>8} "
            f"GT={row['gt']:>6,} "
            f"TP={row['tp']:>6,} "
            f"FN={row['fn']:>6,} "
            f"Recall={row['recall_percentage']:>7.2f}%"
        )

    print()
    print(
        "REPORTS"
    )
    print("-" * 72)

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
        f"[OK] {DENSITY_CSV}"
    )

    print(
        f"[OK] {PATTERN_CSV}"
    )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
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
            "[CANCELADO] Experimento interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print("[ERROR EXP01]")
        print("=" * 72)
        print()
        print(str(exc))
        print()

        sys.exit(1)