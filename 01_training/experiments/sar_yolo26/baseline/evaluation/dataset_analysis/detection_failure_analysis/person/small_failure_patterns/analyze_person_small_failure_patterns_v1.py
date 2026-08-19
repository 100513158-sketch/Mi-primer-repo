from pathlib import Path
from collections import Counter, defaultdict
import csv
import math

from ultralytics import YOLO


# ========================================================================
# CONFIGURACIÓN
# ========================================================================

BASE_DIR = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_DIR = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = (
    BASE_DIR
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)

TEST_IMAGES_DIR = DATASET_DIR / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_DIR / "test_dev" / "labels"

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "analyze_person_small_failure_patterns_v1"
)

REPORTS_DIR = OUTPUT_DIR / "reports"

# ------------------------------------------------------------------------
# Parámetros del análisis
# ------------------------------------------------------------------------

INPUT_SCALE = 1536

PERSON_CLASS_ID = 0

# Persona pequeña = área bbox en píxeles < 256
SMALL_AREA_THRESHOLD = 256.0

# Match GT <-> predicción
MATCH_IOU_THRESHOLD = 0.50

# Predicciones YOLO
CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.70

# Consideramos una predicción "cercana" a un FN si su IoU está por
# encima de este valor aunque no llegue a 0.50.
NEAR_PRED_IOU_THRESHOLD = 0.10

# Distancia máxima relativa para considerar una predicción cercana.
NEAR_PRED_DISTANCE_FACTOR = 2.0

# Generaremos resumen de los primeros N FN.
MAX_FAILURES = None


# ========================================================================
# UTILIDADES
# ========================================================================

def ensure_directories():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_width(box):
    return max(0.0, box[2] - box[0])


def box_height(box):
    return max(0.0, box[3] - box[1])


def box_center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def intersection_area(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def iou(box_a, box_b):
    inter = intersection_area(box_a, box_b)

    if inter <= 0:
        return 0.0

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def intersection_over_small_box(box_small, box_other):
    """
    Fracción de la bbox pequeña cubierta por otra persona.
    Se utiliza como proxy de oclusión/solapamiento.
    """
    area_small = box_area(box_small)

    if area_small <= 0:
        return 0.0

    inter = intersection_area(box_small, box_other)

    return inter / area_small


def euclidean_distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
    )


def normalize(value, minimum, maximum):
    if maximum <= minimum:
        return 0.0

    return (value - minimum) / (maximum - minimum)


# ========================================================================
# BUCKETS
# ========================================================================

def size_bucket(area):
    if area < 16:
        return "<16"

    if area < 32:
        return "16-32"

    if area < 64:
        return "32-64"

    if area < 128:
        return "64-128"

    if area < 256:
        return "128-256"

    return ">=256"


def density_bucket(person_count):
    if person_count < 25:
        return "<25"

    if person_count < 50:
        return "25-49"

    if person_count < 100:
        return "50-99"

    if person_count < 200:
        return "100-199"

    if person_count < 300:
        return "200-299"

    return "300-499+"


def location_bucket(cx, cy, image_width, image_height):
    """
    Divide la imagen en 9 regiones:
    
    top_left
    top_center
    top_right
    middle_left
    center
    middle_right
    bottom_left
    bottom_center
    bottom_right
    """

    nx = normalize(cx, 0, image_width)
    ny = normalize(cy, 0, image_height)

    if nx < 1 / 3:
        horizontal = "left"
    elif nx < 2 / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    if ny < 1 / 3:
        vertical = "top"
    elif ny < 2 / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    return f"{vertical}_{horizontal}"


def border_bucket(box, image_width, image_height):
    """
    Clasifica la proximidad al borde.
    """

    x1, y1, x2, y2 = box

    distances = [
        x1,
        y1,
        image_width - x2,
        image_height - y2,
    ]

    min_distance = min(distances)

    min_dimension = min(image_width, image_height)

    relative_distance = (
        min_distance / min_dimension
        if min_dimension > 0
        else 0.0
    )

    if relative_distance < 0.02:
        return "VERY_NEAR_BORDER"

    if relative_distance < 0.05:
        return "NEAR_BORDER"

    if relative_distance < 0.10:
        return "MODERATE_BORDER"

    return "INTERIOR"


def occlusion_bucket(max_overlap):
    if max_overlap <= 0.0:
        return "NO_OVERLAP"

    if max_overlap < 0.25:
        return "LOW_OVERLAP"

    if max_overlap < 0.50:
        return "MEDIUM_OVERLAP"

    if max_overlap < 0.75:
        return "HIGH_OVERLAP"

    return "VERY_HIGH_OVERLAP"


def proximity_bucket(distance, reference_size):
    """
    Distancia entre centros normalizada por el tamaño característico
    de la persona pequeña.

    reference_size = sqrt(area)
    """

    if reference_size <= 0:
        return "UNKNOWN"

    ratio = distance / reference_size

    if ratio < 1.5:
        return "VERY_CLOSE"

    if ratio < 3.0:
        return "CLOSE"

    if ratio < 6.0:
        return "MODERATE"

    return "FAR"


# ========================================================================
# LABELS
# ========================================================================

def load_yolo_labels(label_path, image_width, image_height):
    """
    Carga labels YOLO:

        class x_center y_center width height

    Coordenadas normalizadas.
    """

    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            x1 = (xc - w / 2.0) * image_width
            y1 = (yc - h / 2.0) * image_height
            x2 = (xc + w / 2.0) * image_width
            y2 = (yc + h / 2.0) * image_height

            x1 = max(0.0, min(x1, image_width))
            y1 = max(0.0, min(y1, image_height))
            x2 = max(0.0, min(x2, image_width))
            y2 = max(0.0, min(y2, image_height))

            box = [x1, y1, x2, y2]

            objects.append(
                {
                    "class_id": class_id,
                    "box": box,
                    "area": box_area(box),
                    "width": box_width(box),
                    "height": box_height(box),
                    "center": box_center(box),
                }
            )

    return objects


# ========================================================================
# MATCHING
# ========================================================================

def match_predictions(gt_objects, predictions):
    """
    Match greedy por IoU.

    Devuelve:
        matched_gt_indices
        matched_prediction_indices
    """

    candidates = []

    for gt_index, gt in enumerate(gt_objects):
        for pred_index, pred in enumerate(predictions):

            value = iou(gt["box"], pred["box"])

            if value >= MATCH_IOU_THRESHOLD:
                candidates.append(
                    (
                        value,
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()

    for value, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

    return matched_gt, matched_predictions


# ========================================================================
# ANÁLISIS DE PATRONES
# ========================================================================

def analyze_failure_pattern(
    failure,
    all_gt_persons,
    predictions,
    image_width,
    image_height,
):
    """

    Analiza un FN pequeño y determina características asociadas.

    """

    box = failure["box"]
    center = failure["center"]
    area = failure["area"]

    # ------------------------------------------------------------
    # Tamaño
    # ------------------------------------------------------------

    size = math.sqrt(area) if area > 0 else 0.0

    # ------------------------------------------------------------
    # Densidad
    # ------------------------------------------------------------

    person_count = len(all_gt_persons)

    density = density_bucket(person_count)

    # ------------------------------------------------------------
    # Localización
    # ------------------------------------------------------------

    location = location_bucket(
        center[0],
        center[1],
        image_width,
        image_height,
    )

    border = border_bucket(
        box,
        image_width,
        image_height,
    )

    # ------------------------------------------------------------
    # Solapamiento con otras personas GT
    # ------------------------------------------------------------

    max_overlap = 0.0
    nearest_gt_distance = None
    nearest_gt_index = None

    for index, other in enumerate(all_gt_persons):

        if other is failure:
            continue

        overlap = intersection_over_small_box(
            box,
            other["box"],
        )

        if overlap > max_overlap:
            max_overlap = overlap

        other_center = other["center"]

        distance = euclidean_distance(
            center,
            other_center,
        )

        if nearest_gt_distance is None or distance < nearest_gt_distance:
            nearest_gt_distance = distance
            nearest_gt_index = index

    occlusion = occlusion_bucket(max_overlap)

    if nearest_gt_distance is None:
        nearest_person_proximity = "ISOLATED"
    else:
        nearest_person_proximity = proximity_bucket(
            nearest_gt_distance,
            size,
        )

    # ------------------------------------------------------------
    # Predicción cercana
    # ------------------------------------------------------------

    best_prediction_iou = 0.0
    best_prediction_distance = None
    best_prediction_confidence = 0.0

    for pred in predictions:

        pred_iou = iou(
            box,
            pred["box"],
        )

        if pred_iou > best_prediction_iou:
            best_prediction_iou = pred_iou
            best_prediction_confidence = pred["confidence"]

        pred_center = pred["center"]

        distance = euclidean_distance(
            center,
            pred_center,
        )

        if (
            best_prediction_distance is None
            or distance < best_prediction_distance
        ):
            best_prediction_distance = distance

    if best_prediction_iou >= NEAR_PRED_IOU_THRESHOLD:

        prediction_relation = "NEAR_MISSED_PREDICTION"

    elif (
        best_prediction_distance is not None
        and best_prediction_distance
        < NEAR_PRED_DISTANCE_FACTOR * max(size, 1.0)
    ):

        prediction_relation = "VERY_NEAR_PREDICTION"

    else:

        prediction_relation = "NO_NEAR_PREDICTION"

    # ------------------------------------------------------------
    # Patrón dominante
    # ------------------------------------------------------------

    if area < 16:

        dominant_pattern = "EXTREMELY_SMALL"

    elif max_overlap >= 0.75:

        dominant_pattern = "HEAVY_OCCLUSION"

    elif max_overlap >= 0.50:

        dominant_pattern = "HIGH_OCCLUSION"

    elif nearest_person_proximity == "VERY_CLOSE":

        dominant_pattern = "CROWDED_NEIGHBORS"

    elif border in (
        "VERY_NEAR_BORDER",
        "NEAR_BORDER",
    ):

        dominant_pattern = "BORDER_EFFECT"

    elif best_prediction_iou >= 0.30:

        dominant_pattern = "LOCALIZATION_NEAR_MISS"

    elif best_prediction_iou >= NEAR_PRED_IOU_THRESHOLD:

        dominant_pattern = "WEAK_LOCALIZATION"

    elif density in (
        "200-299",
        "300-499+",
    ):

        dominant_pattern = "HIGH_DENSITY"

    elif area < 64:

        dominant_pattern = "VERY_SMALL_OBJECT"

    elif area < 128:

        dominant_pattern = "SMALL_OBJECT"

    else:

        dominant_pattern = "SMALL_OBJECT_OTHER"

    return {
        "size_sqrt": size,
        "size_bucket": size_bucket(area),
        "density_count": person_count,
        "density_bucket": density,
        "location": location,
        "border": border,
        "max_gt_overlap": max_overlap,
        "occlusion_bucket": occlusion,
        "nearest_gt_distance": (
            nearest_gt_distance
            if nearest_gt_distance is not None
            else -1.0
        ),
        "nearest_person_proximity": nearest_person_proximity,
        "best_prediction_iou": best_prediction_iou,
        "best_prediction_confidence": best_prediction_confidence,
        "nearest_prediction_distance": (
            best_prediction_distance
            if best_prediction_distance is not None
            else -1.0
        ),
        "prediction_relation": prediction_relation,
        "dominant_pattern": dominant_pattern,
    }


# ========================================================================
# CSV
# ========================================================================

def write_objects_csv(rows):

    path = (
        REPORTS_DIR
        / "person_small_failure_patterns_objects_v1.csv"
    )

    fieldnames = [
        "image",
        "image_width",
        "image_height",
        "person_gt_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "area",
        "size_sqrt",
        "size_bucket",
        "density_count",
        "density_bucket",
        "location",
        "border",
        "max_gt_overlap",
        "occlusion_bucket",
        "nearest_gt_distance",
        "nearest_person_proximity",
        "best_prediction_iou",
        "best_prediction_confidence",
        "nearest_prediction_distance",
        "prediction_relation",
        "dominant_pattern",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[OK] {path}")

    return path


def aggregate_rows(rows, key):

    groups = defaultdict(
        lambda: {
            "gt": 0,
            "patterns": Counter(),
        }
    )

    for row in rows:

        value = row[key]

        groups[value]["gt"] += 1

        groups[value]["patterns"][
            row["dominant_pattern"]
        ] += 1

    return groups


def write_group_csv(rows, key, filename):

    groups = aggregate_rows(
        rows,
        key,
    )

    patterns = sorted(
        {
            row["dominant_pattern"]
            for row in rows
        }
    )

    path = REPORTS_DIR / filename

    fieldnames = [
        key,
        "failure_count",
        "failure_percent",
    ] + [
        f"pattern_{p}"
        for p in patterns
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        total = len(rows)

        for group_name in sorted(groups):

            group = groups[group_name]

            count = group["gt"]

            result = {
                key: group_name,
                "failure_count": count,
                "failure_percent": (
                    count / total
                    if total > 0
                    else 0.0
                ),
            }

            for pattern in patterns:

                result[
                    f"pattern_{pattern}"
                ] = group["patterns"].get(
                    pattern,
                    0,
                )

            writer.writerow(result)

    print(f"[OK] {path}")

    return path


# ========================================================================
# SUMMARY
# ========================================================================

def percentage(value, total):

    if total <= 0:
        return 0.0

    return value / total


def write_summary(
    rows,
    total_images,
    total_person_gt,
    total_person_tp,
    total_person_fn,
    small_gt,
    small_tp,
    small_fn,
):

    path = (
        REPORTS_DIR
        / "PERSON_SMALL_FAILURE_PATTERNS_V1_SUMMARY.txt"
    )

    total_failures = len(rows)

    pattern_counter = Counter(
        row["dominant_pattern"]
        for row in rows
    )

    size_counter = Counter(
        row["size_bucket"]
        for row in rows
    )

    density_counter = Counter(
        row["density_bucket"]
        for row in rows
    )

    location_counter = Counter(
        row["location"]
        for row in rows
    )

    occlusion_counter = Counter(
        row["occlusion_bucket"]
        for row in rows
    )

    proximity_counter = Counter(
        row["nearest_person_proximity"]
        for row in rows
    )

    prediction_counter = Counter(
        row["prediction_relation"]
        for row in rows
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "SAR YOLO26 - PERSON SMALL FAILURE PATTERNS ANALYSIS V1\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        f.write(
            f"Imágenes: {total_images:,}\n"
        )

        f.write(
            f"PERSON GT: {total_person_gt:,}\n"
        )

        f.write(
            f"PERSON TP: {total_person_tp:,}\n"
        )

        f.write(
            f"PERSON FN: {total_person_fn:,}\n"
        )

        person_recall = percentage(
            total_person_tp,
            total_person_gt,
        )

        f.write(
            f"PERSON Recall: {person_recall:.4f}\n\n"
        )

        f.write(
            f"SMALL PERSON GT (<256): {small_gt:,}\n"
        )

        f.write(
            f"SMALL PERSON TP (<256): {small_tp:,}\n"
        )

        f.write(
            f"SMALL PERSON FN (<256): {small_fn:,}\n"
        )

        small_recall = percentage(
            small_tp,
            small_gt,
        )

        f.write(
            f"SMALL PERSON Recall: {small_recall:.4f}\n\n"
        )

        f.write(
            f"SMALL FAILURE PATTERNS ANALYZED: "
            f"{total_failures:,}\n\n"
        )

        # ------------------------------------------------------------
        # Patrones dominantes
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "PATRONES DOMINANTES DE FALLO\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for pattern, count in pattern_counter.most_common():

            f.write(
                f"{pattern:<30}"
                f" COUNT={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # Tamaño
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "FALLOS POR TAMAÑO\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for bucket, count in sorted(
            size_counter.items(),
            key=lambda x: x[0],
        ):

            f.write(
                f"{bucket:<15}"
                f" FN={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # Densidad
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "FALLOS POR DENSIDAD\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for bucket, count in sorted(
            density_counter.items(),
            key=lambda x: x[0],
        ):

            f.write(
                f"{bucket:<15}"
                f" FN={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # Localización
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "FALLOS POR LOCALIZACIÓN\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for location, count in location_counter.most_common():

            f.write(
                f"{location:<20}"
                f" FN={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # Solapamiento
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "FALLOS POR SOLAPAMIENTO / OCLUSIÓN\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for bucket, count in occlusion_counter.most_common():

            f.write(
                f"{bucket:<20}"
                f" FN={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # Proximidad
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "FALLOS POR PROXIMIDAD A OTRAS PERSONAS\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for bucket, count in proximity_counter.most_common():

            f.write(
                f"{bucket:<20}"
                f" FN={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        # ------------------------------------------------------------
        # Predicciones cercanas
        # ------------------------------------------------------------

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "RELACIÓN CON PREDICCIONES CERCANAS\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        for relation, count in prediction_counter.most_common():

            f.write(
                f"{relation:<30}"
                f" FN={count:>6}"
                f" PCT={percentage(count, total_failures):.4f}\n"
            )

        f.write("\n")

        f.write(
            "=" * 72
            + "\n"
        )

        f.write(
            "INTERPRETACIÓN AUTOMÁTICA\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        if total_failures == 0:

            f.write(
                "No se encontraron fallos residuales pequeños.\n"
            )

        else:

            dominant, dominant_count = (
                pattern_counter.most_common(1)[0]
            )

            f.write(
                f"Patrón dominante: {dominant}\n"
            )

            f.write(
                f"Casos: {dominant_count:,}\n"
            )

            f.write(
                f"Proporción: "
                f"{percentage(dominant_count, total_failures):.4f}\n\n"
            )

            if pattern_counter.get(
                "EXTREMELY_SMALL",
                0,
            ) > 0:

                f.write(
                    "- Existe una componente importante asociada "
                    "a personas extremadamente pequeñas.\n"
                )

            if (
                pattern_counter.get(
                    "HEAVY_OCCLUSION",
                    0,
                )
                + pattern_counter.get(
                    "HIGH_OCCLUSION",
                    0,
                )
                > 0
            ):

                f.write(
                    "- Existe evidencia de fallos asociados "
                    "a solapamiento/oclusión.\n"
                )

            if pattern_counter.get(
                "CROWDED_NEIGHBORS",
                0,
            ) > 0:

                f.write(
                    "- Existe evidencia de fallos relacionados "
                    "con proximidad entre personas.\n"
                )

            if pattern_counter.get(
                "BORDER_EFFECT",
                0,
            ) > 0:

                f.write(
                    "- Existe una componente de fallos próxima "
                    "a los bordes de la imagen.\n"
                )

            if pattern_counter.get(
                "LOCALIZATION_NEAR_MISS",
                0,
            ) > 0:

                f.write(
                    "- Parte de los fallos presenta predicciones "
                    "cercanas con IoU insuficiente, indicando "
                    "problemas de localización.\n"
                )

            if pattern_counter.get(
                "WEAK_LOCALIZATION",
                0,
            ) > 0:

                f.write(
                    "- Existen predicciones cercanas débiles "
                    "que no alcanzan el IoU de matching.\n"
                )

            f.write("\n")

        f.write(
            "IMPORTANTE: el dataset NO ha sido modificado.\n"
        )

    print(f"[OK] {path}")

    return path


# ========================================================================
# MAIN
# ========================================================================

def main():

    ensure_directories()

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - PERSON SMALL FAILURE PATTERNS ANALYSIS V1"
    )
    print("=" * 72)
    print()

    print("Dataset:")
    print(DATASET_DIR)
    print()

    print("Modelo:")
    print(MODEL_PATH)
    print()

    print("Test:")
    print(TEST_IMAGES_DIR)
    print()

    print("Output:")
    print(OUTPUT_DIR)
    print()

    print(f"Input scale: {INPUT_SCALE}")
    print(
        f"Small PERSON area: < {SMALL_AREA_THRESHOLD}"
    )
    print(
        f"Match IoU threshold: {MATCH_IOU_THRESHOLD}"
    )
    print()

    image_paths = sorted(
        [
            p
            for p in TEST_IMAGES_DIR.iterdir()
            if p.suffix.lower()
            in (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
            )
        ]
    )

    total_images = len(image_paths)

    print(
        f"Imágenes encontradas: {total_images:,}"
    )
    print()

    if total_images == 0:

        print(
            "[ERROR] No se encontraron imágenes."
        )

        return

    print(
        "Cargando modelo YOLO26s..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print("[OK] Modelo cargado.")
    print()

    all_failure_rows = []

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    small_person_gt = 0
    small_person_tp = 0
    small_person_fn = 0

    # ================================================================
    # Procesamiento
    # ================================================================

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        # ------------------------------------------------------------
        # Leer imagen
        # ------------------------------------------------------------

        import cv2

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[WARNING] No se pudo leer: "
                f"{image_path}"
            )

            continue

        image_height, image_width = image.shape[:2]

        # ------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------

        label_path = (
            TEST_LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        gt_objects = load_yolo_labels(
            label_path,
            image_width,
            image_height,
        )

        gt_persons = [
            obj
            for obj in gt_objects
            if obj["class_id"] == PERSON_CLASS_ID
        ]

        total_person_gt += len(
            gt_persons
        )

        # ------------------------------------------------------------
        # Predicción
        # ------------------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=INPUT_SCALE,
            conf=CONF_THRESHOLD,
            iou=IOU_NMS_THRESHOLD,
            verbose=False,
        )

        predictions = []

        if results:

            result = results[0]

            if result.boxes is not None:

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                classes = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                for box, cls, confidence in zip(
                    boxes,
                    classes,
                    confidences,
                ):

                    class_id = int(cls)

                    if class_id != PERSON_CLASS_ID:
                        continue

                    box = [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ]

                    predictions.append(
                        {
                            "class_id": class_id,
                            "box": box,
                            "area": box_area(box),
                            "width": box_width(box),
                            "height": box_height(box),
                            "center": box_center(box),
                            "confidence": float(
                                confidence
                            ),
                        }
                    )

        # ------------------------------------------------------------
        # Match
        # ------------------------------------------------------------

        matched_gt, matched_predictions = (
            match_predictions(
                gt_persons,
                predictions,
            )
        )

        total_person_tp += len(
            matched_gt
        )

        total_person_fn += (
            len(gt_persons)
            - len(matched_gt)
        )

        # ------------------------------------------------------------
        # SMALL GT
        # ------------------------------------------------------------

        small_indices = []

        for gt_index, gt in enumerate(
            gt_persons
        ):

            if gt["area"] < SMALL_AREA_THRESHOLD:

                small_person_gt += 1

                small_indices.append(
                    gt_index
                )

        small_matched = (
            set(small_indices)
            & matched_gt
        )

        small_person_tp += len(
            small_matched
        )

        # ------------------------------------------------------------
        # Residual FN
        # ------------------------------------------------------------

        for gt_index in small_indices:

            if gt_index in matched_gt:
                continue

            small_person_fn += 1

            failure = gt_persons[
                gt_index
            ]

            pattern = analyze_failure_pattern(
                failure=failure,
                all_gt_persons=gt_persons,
                predictions=predictions,
                image_width=image_width,
                image_height=image_height,
            )

            x1, y1, x2, y2 = (
                failure["box"]
            )

            row = {
                "image": image_path.name,
                "image_width": image_width,
                "image_height": image_height,
                "person_gt_index": gt_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": failure["width"],
                "height": failure["height"],
                "area": failure["area"],
                **pattern,
            }

            all_failure_rows.append(
                row
            )

            if (
                MAX_FAILURES is not None
                and len(all_failure_rows)
                >= MAX_FAILURES
            ):
                break

        # ------------------------------------------------------------
        # Progress
        # ------------------------------------------------------------

        if (
            image_index % 100 == 0
            or image_index == total_images
        ):

            print(
                f"Analizadas: "
                f"{image_index:,}/"
                f"{total_images:,}"
            )

        if (
            MAX_FAILURES is not None
            and len(all_failure_rows)
            >= MAX_FAILURES
        ):
            break

    # ====================================================================
    # RESULTADOS
    # ====================================================================

    print()
    print("=" * 72)
    print(
        "# RESULTADO PERSON SMALL FAILURE PATTERNS V1"
    )
    print("=" * 72)
    print()

    person_recall = percentage(
        total_person_tp,
        total_person_gt,
    )

    small_recall = percentage(
        small_person_tp,
        small_person_gt,
    )

    print(
        f"Imágenes:              "
        f"{total_images:,}"
    )

    print(
        f"PERSON GT:             "
        f"{total_person_gt:,}"
    )

    print(
        f"PERSON TP:             "
        f"{total_person_tp:,}"
    )

    print(
        f"PERSON FN:             "
        f"{total_person_fn:,}"
    )

    print(
        f"PERSON Recall:         "
        f"{person_recall:.4f}"
    )

    print()

    print(
        f"SMALL PERSON GT:       "
        f"{small_person_gt:,}"
    )

    print(
        f"SMALL PERSON TP:       "
        f"{small_person_tp:,}"
    )

    print(
        f"SMALL PERSON FN:       "
        f"{small_person_fn:,}"
    )

    print(
        f"SMALL PERSON Recall:   "
        f"{small_recall:.4f}"
    )

    print()

    print(
        f"Fallos residuales analizados: "
        f"{len(all_failure_rows):,}"
    )

    print()

    # ====================================================================
    # CSV PRINCIPAL
    # ====================================================================

    write_objects_csv(
        all_failure_rows
    )

    write_group_csv(
        all_failure_rows,
        "size_bucket",
        "person_small_failure_patterns_by_size_v1.csv",
    )

    write_group_csv(
        all_failure_rows,
        "density_bucket",
        "person_small_failure_patterns_by_density_v1.csv",
    )

    write_group_csv(
        all_failure_rows,
        "location",
        "person_small_failure_patterns_by_location_v1.csv",
    )

    write_group_csv(
        all_failure_rows,
        "occlusion_bucket",
        "person_small_failure_patterns_by_occlusion_v1.csv",
    )

    write_group_csv(
        all_failure_rows,
        "nearest_person_proximity",
        "person_small_failure_patterns_by_proximity_v1.csv",
    )

    write_summary(
        rows=all_failure_rows,
        total_images=total_images,
        total_person_gt=total_person_gt,
        total_person_tp=total_person_tp,
        total_person_fn=total_person_fn,
        small_gt=small_person_gt,
        small_tp=small_person_tp,
        small_fn=small_person_fn,
    )

    print()
    print(
        "[OK] Reports generados."
    )

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()