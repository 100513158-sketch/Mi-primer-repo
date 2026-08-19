from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from ultralytics import YOLO


# ============================================================================
# CONFIGURACION
# ============================================================================

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
    / "analyze_person_small_failure_interactions_v1"
)

REPORTS_DIR = OUTPUT_DIR / "reports"

INPUT_SCALE = 1536
PERSON_CLASS_ID = 0

SMALL_AREA_THRESHOLD = 256.0
MATCH_IOU_THRESHOLD = 0.50

CONF_THRESHOLD = 0.25

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
# UTILIDADES
# ============================================================================

def ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def xywhn_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> List[float]:

    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(img_w), x2),
        min(float(img_h), y2),
    ]


def box_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_width(box: List[float]) -> float:
    return max(0.0, box[2] - box[0])


def box_height(box: List[float]) -> float:
    return max(0.0, box[3] - box[1])


def box_center(box: List[float]) -> Tuple[float, float]:
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def intersection_area(
    box_a: List[float],
    box_b: List[float],
) -> float:

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(
    box_a: List[float],
    box_b: List[float],
) -> float:

    inter = intersection_area(box_a, box_b)

    if inter <= 0:
        return 0.0

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def intersection_over_gt(
    gt_box: List[float],
    other_box: List[float],
) -> float:

    inter = intersection_area(gt_box, other_box)
    gt_area = box_area(gt_box)

    if gt_area <= 0:
        return 0.0

    return inter / gt_area


def center_distance(
    box_a: List[float],
    box_b: List[float],
) -> float:

    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return math.sqrt(
        (ax - bx) ** 2 +
        (ay - by) ** 2
    )


# ============================================================================
# CLASIFICACION DE TAMAÑO
# ============================================================================

def size_bucket(area: float) -> str:

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

    if area < 512:
        return "256-512"

    if area < 1024:
        return "512-1024"

    if area < 2048:
        return "1024-2048"

    return ">2048"


# ============================================================================
# DENSIDAD
# ============================================================================

def density_category(
    gt_box: List[float],
    all_person_boxes: List[List[float]],
) -> Tuple[str, int]:

    neighbors = 0

    gt_center_x, gt_center_y = box_center(gt_box)

    gt_w = box_width(gt_box)
    gt_h = box_height(gt_box)

    reference_size = max(
        1.0,
        min(gt_w, gt_h)
    )

    # Radio adaptativo.
    radius = max(
        32.0,
        reference_size * 3.0
    )

    for other in all_person_boxes:

        if other is gt_box:
            continue

        ox, oy = box_center(other)

        distance = math.sqrt(
            (gt_center_x - ox) ** 2 +
            (gt_center_y - oy) ** 2
        )

        if distance <= radius:
            neighbors += 1

    if neighbors == 0:
        category = "ISOLATED"

    elif neighbors <= 2:
        category = "LOW_DENSITY"

    elif neighbors <= 5:
        category = "MEDIUM_DENSITY"

    else:
        category = "HIGH_DENSITY"

    return category, neighbors


# ============================================================================
# PROXIMIDAD
# ============================================================================

def proximity_category(
    gt_box: List[float],
    all_person_boxes: List[List[float]],
) -> Tuple[str, int]:

    close_neighbors = 0

    gt_cx, gt_cy = box_center(gt_box)

    gt_w = box_width(gt_box)
    gt_h = box_height(gt_box)

    reference_size = max(
        1.0,
        min(gt_w, gt_h)
    )

    close_distance = max(
        12.0,
        reference_size * 1.5
    )

    for other in all_person_boxes:

        if other is gt_box:
            continue

        distance = center_distance(
            gt_box,
            other
        )

        if distance <= close_distance:
            close_neighbors += 1

    if close_neighbors == 0:
        category = "NO_CLOSE_NEIGHBOR"

    elif close_neighbors == 1:
        category = "ONE_CLOSE_NEIGHBOR"

    elif close_neighbors <= 3:
        category = "MULTIPLE_CLOSE_NEIGHBORS"

    else:
        category = "VERY_CLOSE_CROWD"

    return category, close_neighbors


# ============================================================================
# OCULUSION
# ============================================================================

def occlusion_category(
    gt_box: List[float],
    all_person_boxes: List[List[float]],
) -> Tuple[str, float, int]:

    max_overlap = 0.0
    overlapping_persons = 0

    for other in all_person_boxes:

        if other is gt_box:
            continue

        overlap = intersection_over_gt(
            gt_box,
            other
        )

        if overlap > 0.05:
            overlapping_persons += 1

        max_overlap = max(
            max_overlap,
            overlap
        )

    if max_overlap < 0.05:
        category = "NO_OVERLAP"

    elif max_overlap < 0.20:
        category = "LOW_OCCLUSION"

    elif max_overlap < 0.50:
        category = "MEDIUM_OCCLUSION"

    else:
        category = "HIGH_OCCLUSION"

    return category, max_overlap, overlapping_persons


# ============================================================================
# LOCALIZACION
# ============================================================================

def location_category(
    box: List[float],
    img_w: int,
    img_h: int,
) -> str:

    cx, cy = box_center(box)

    margin_x = img_w * 0.10
    margin_y = img_h * 0.10

    left = cx < margin_x
    right = cx > img_w - margin_x
    top = cy < margin_y
    bottom = cy > img_h - margin_y

    if top and left:
        return "TOP_LEFT"

    if top and right:
        return "TOP_RIGHT"

    if bottom and left:
        return "BOTTOM_LEFT"

    if bottom and right:
        return "BOTTOM_RIGHT"

    if top:
        return "TOP"

    if bottom:
        return "BOTTOM"

    if left:
        return "LEFT"

    if right:
        return "RIGHT"

    return "CENTER"


# ============================================================================
# RELACION DE PREDICCION
# ============================================================================

def prediction_relation(
    gt_box: List[float],
    predictions: List[List[float]],
) -> Tuple[str, float]:

    best_iou = 0.0

    for pred in predictions:

        current_iou = iou(
            gt_box,
            pred
        )

        if current_iou > best_iou:
            best_iou = current_iou

    if best_iou >= MATCH_IOU_THRESHOLD:
        return "NO_FAILURE", best_iou

    if best_iou > 0:
        return "LOCALIZATION_ERROR", best_iou

    return "NO_PREDICTION", best_iou


# ============================================================================
# LECTURA DE LABELS
# ============================================================================

def read_person_labels(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[List[float]]:

    boxes = []

    if not label_path.exists():
        return boxes

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                continue

            if cls != PERSON_CLASS_ID:
                continue

            box = xywhn_to_xyxy(
                x,
                y,
                w,
                h,
                img_w,
                img_h,
            )

            boxes.append(box)

    return boxes


# ============================================================================
# PREDICCIONES
# ============================================================================

def get_predictions(
    model: YOLO,
    image_path: Path,
) -> List[List[float]]:

    results = model.predict(
        source=str(image_path),
        imgsz=INPUT_SCALE,
        conf=CONF_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    predictions = []

    if not results:
        return predictions

    result = results[0]

    if result.boxes is None:
        return predictions

    if len(result.boxes) == 0:
        return predictions

    boxes = result.boxes.xyxy.cpu().tolist()

    for box in boxes:
        if len(box) >= 4:
            predictions.append([
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            ])

    return predictions


# ============================================================================
# CARGA DE DIMENSIONES DE IMAGEN
# ============================================================================

def get_image_size(
    image_path: Path,
) -> Tuple[int, int]:

    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.width, img.height

    except Exception:
        return 1, 1


# ============================================================================
# GENERACION DE FACTORES
# ============================================================================

def analyze_small_failure(
    gt_box: List[float],
    all_person_boxes: List[List[float]],
    predictions: List[List[float]],
    img_w: int,
    img_h: int,
) -> Optional[Dict]:

    area = box_area(gt_box)

    if area >= SMALL_AREA_THRESHOLD:
        return None

    size = size_bucket(area)

    density, neighbor_count = density_category(
        gt_box,
        all_person_boxes,
    )

    proximity, close_neighbor_count = proximity_category(
        gt_box,
        all_person_boxes,
    )

    occlusion, max_overlap, overlapping_persons = occlusion_category(
        gt_box,
        all_person_boxes,
    )

    location = location_category(
        gt_box,
        img_w,
        img_h,
    )

    pred_relation, best_iou = prediction_relation(
        gt_box,
        predictions,
    )

    if pred_relation == "NO_FAILURE":
        return None

    factors = []

    # Tamaño extremo
    if area < 32:
        factors.append("EXTREME_SMALL")

    # Densidad
    if density in {
        "MEDIUM_DENSITY",
        "HIGH_DENSITY",
    }:
        factors.append("DENSE_SCENE")

    # Proximidad
    if proximity in {
        "MULTIPLE_CLOSE_NEIGHBORS",
        "VERY_CLOSE_CROWD",
    }:
        factors.append("CLOSE_NEIGHBORS")

    # Oclusión
    if occlusion in {
        "MEDIUM_OCCLUSION",
        "HIGH_OCCLUSION",
    }:
        factors.append("OCCLUSION")

    # Borde
    if location != "CENTER":
        factors.append("EDGE_LOCATION")

    # Localización
    if pred_relation == "LOCALIZATION_ERROR":
        factors.append("LOCALIZATION_ERROR")

    # Sin predicción
    if pred_relation == "NO_PREDICTION":
        factors.append("NO_PREDICTION")

    if not factors:
        factors.append("PURE_SMALL_SCALE")

    if len(factors) == 1:
        interaction_class = factors[0]

    elif len(factors) == 2:
        interaction_class = (
            factors[0] +
            "+" +
            factors[1]
        )

    else:
        interaction_class = "MULTI_FACTOR_FAILURE"

    return {
        "area": area,
        "size_bucket": size,
        "density": density,
        "neighbor_count": neighbor_count,
        "proximity": proximity,
        "close_neighbor_count": close_neighbor_count,
        "occlusion": occlusion,
        "max_person_overlap": max_overlap,
        "overlapping_persons": overlapping_persons,
        "location": location,
        "prediction_relation": pred_relation,
        "best_iou": best_iou,
        "factor_count": len(factors),
        "factors": "|".join(factors),
        "interaction_class": interaction_class,
    }


# ============================================================================
# CSV GENERICO
# ============================================================================

def write_csv(
    path: Path,
    rows: List[Dict],
    fieldnames: List[str],
) -> None:

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================================
# MATRICES DE INTERACCION
# ============================================================================

def build_pair_matrix(
    rows: List[Dict],
    field_a: str,
    field_b: str,
) -> List[Dict]:

    counter = Counter()

    for row in rows:

        a = row[field_a]
        b = row[field_b]

        counter[(a, b)] += 1

    output = []

    for (a, b), count in sorted(counter.items()):

        output.append({
            field_a: a,
            field_b: b,
            "failures": count,
            "percentage": safe_div(
                count,
                len(rows)
            ),
        })

    return output


# ============================================================================
# FACTORES
# ============================================================================

def build_factor_statistics(
    rows: List[Dict],
) -> List[Dict]:

    counter = Counter()

    for row in rows:

        factors = row["factors"].split("|")

        for factor in factors:
            counter[factor] += 1

    output = []

    for factor, count in counter.most_common():

        output.append({
            "factor": factor,
            "failures": count,
            "percentage": safe_div(
                count,
                len(rows),
            ),
        })

    return output


# ============================================================================
# INTERACCIONES
# ============================================================================

def build_interaction_statistics(
    rows: List[Dict],
) -> List[Dict]:

    counter = Counter()

    for row in rows:

        factors = row["factors"].split("|")

        unique_factors = sorted(
            set(factors)
        )

        if len(unique_factors) < 2:
            continue

        if len(unique_factors) == 2:
            key = (
                unique_factors[0] +
                " + " +
                unique_factors[1]
            )

        else:
            key = " + ".join(
                unique_factors
            )

        counter[key] += 1

    output = []

    for interaction, count in counter.most_common():

        output.append({
            "interaction": interaction,
            "failures": count,
            "percentage": safe_div(
                count,
                len(rows),
            ),
        })

    return output


# ============================================================================
# RESUMEN POR NUMERO DE FACTORES
# ============================================================================

def build_factor_count_statistics(
    rows: List[Dict],
) -> List[Dict]:

    counter = Counter(
        row["factor_count"]
        for row in rows
    )

    output = []

    for count, failures in sorted(
        counter.items()
    ):

        output.append({
            "factor_count": count,
            "failures": failures,
            "percentage": safe_div(
                failures,
                len(rows),
            ),
        })

    return output


# ============================================================================
# RESUMEN
# ============================================================================

def write_summary(
    path: Path,
    image_count: int,
    person_gt: int,
    person_tp: int,
    person_fn: int,
    small_gt: int,
    small_tp: int,
    small_fn: int,
    rows: List[Dict],
    factor_stats: List[Dict],
    interaction_stats: List[Dict],
    factor_count_stats: List[Dict],
) -> None:

    person_recall = safe_div(
        person_tp,
        person_gt,
    )

    small_recall = safe_div(
        small_tp,
        small_gt,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "========================================================================\n"
        )
        f.write(
            "SAR YOLO26 - PERSON SMALL FAILURE INTERACTIONS V1\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        f.write(
            f"Imágenes:              {image_count:,}\n"
        )

        f.write(
            f"PERSON GT:             {person_gt:,}\n"
        )

        f.write(
            f"PERSON TP:             {person_tp:,}\n"
        )

        f.write(
            f"PERSON FN:             {person_fn:,}\n"
        )

        f.write(
            f"PERSON Recall:         {person_recall:.4f}\n\n"
        )

        f.write(
            f"SMALL PERSON GT:       {small_gt:,}\n"
        )

        f.write(
            f"SMALL PERSON TP:       {small_tp:,}\n"
        )

        f.write(
            f"SMALL PERSON FN:       {small_fn:,}\n"
        )

        f.write(
            f"SMALL PERSON Recall:   {small_recall:.4f}\n\n"
        )

        f.write(
            f"SMALL FAILURE INTERACTIONS: {len(rows):,}\n\n"
        )

        f.write(
            "========================================================================\n"
        )
        f.write(
            "FACTORES INDIVIDUALES\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        for item in factor_stats:

            f.write(
                f"{item['factor']:<30} "
                f"{item['failures']:>7,} "
                f"{item['percentage'] * 100:>7.2f}%\n"
            )

        f.write("\n")

        f.write(
            "========================================================================\n"
        )
        f.write(
            "INTERACCIONES ENTRE FACTORES\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        for item in interaction_stats:

            f.write(
                f"{item['interaction']:<55} "
                f"{item['failures']:>7,} "
                f"{item['percentage'] * 100:>7.2f}%\n"
            )

        f.write("\n")

        f.write(
            "========================================================================\n"
        )
        f.write(
            "NUMERO DE FACTORES POR FALLO\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        for item in factor_count_stats:

            f.write(
                f"{item['factor_count']} factores: "
                f"{item['failures']:>7,} "
                f"{item['percentage'] * 100:>7.2f}%\n"
            )

        f.write("\n")

        f.write(
            "========================================================================\n"
        )
        f.write(
            "CONFIGURACION\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        f.write(
            f"Input scale:             {INPUT_SCALE}\n"
        )

        f.write(
            f"Small area threshold:    {SMALL_AREA_THRESHOLD}\n"
        )

        f.write(
            f"Match IoU threshold:     {MATCH_IOU_THRESHOLD}\n"
        )

        f.write(
            "\nIMPORTANTE: el dataset NO ha sido modificado.\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - PERSON SMALL FAILURE INTERACTIONS ANALYSIS V1"
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

    ensure_output_dirs()

    image_paths = sorted(
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(
        f"Imágenes encontradas: {len(image_paths):,}"
    )
    print()

    if not image_paths:
        raise RuntimeError(
            "No se encontraron imágenes de test."
        )

    print("Cargando modelo YOLO26s...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("[OK] Modelo cargado.")
    print()

    all_rows = []

    total_person_gt = 0
    total_person_tp = 0

    small_person_gt = 0
    small_person_tp = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        img_w, img_h = get_image_size(
            image_path
        )

        label_path = (
            TEST_LABELS_DIR /
            f"{image_path.stem}.txt"
        )

        gt_boxes = read_person_labels(
            label_path,
            img_w,
            img_h,
        )

        predictions = get_predictions(
            model,
            image_path,
        )

        total_person_gt += len(
            gt_boxes
        )

        matched_gt_indices = set()

        for gt_index, gt_box in enumerate(
            gt_boxes
        ):

            area = box_area(gt_box)

            if area < SMALL_AREA_THRESHOLD:
                small_person_gt += 1

            best_iou = 0.0
            best_prediction_index = None

            for pred_index, pred_box in enumerate(
                predictions
            ):

                current_iou = iou(
                    gt_box,
                    pred_box,
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_prediction_index = (
                        pred_index
                    )

            if (
                best_iou >= MATCH_IOU_THRESHOLD
                and best_prediction_index is not None
            ):

                if best_prediction_index not in matched_gt_indices:
                    total_person_tp += 1
                    matched_gt_indices.add(
                        best_prediction_index
                    )

            if area < SMALL_AREA_THRESHOLD:

                if (
                    best_iou >= MATCH_IOU_THRESHOLD
                ):
                    small_person_tp += 1

                else:

                    analysis = analyze_small_failure(
                        gt_box,
                        gt_boxes,
                        predictions,
                        img_w,
                        img_h,
                    )

                    if analysis is not None:

                        analysis[
                            "image"
                        ] = image_path.name

                        analysis[
                            "image_width"
                        ] = img_w

                        analysis[
                            "image_height"
                        ] = img_h

                        analysis[
                            "gt_index"
                        ] = gt_index

                        analysis[
                            "x1"
                        ] = gt_box[0]

                        analysis[
                            "y1"
                        ] = gt_box[1]

                        analysis[
                            "x2"
                        ] = gt_box[2]

                        analysis[
                            "y2"
                        ] = gt_box[3]

                        all_rows.append(
                            analysis
                        )

        if (
            index % 100 == 0
            or index == len(image_paths)
        ):

            print(
                f"Analizadas: {index:,}/{len(image_paths):,}"
            )

    total_person_fn = (
        total_person_gt -
        total_person_tp
    )

    small_person_fn = (
        small_person_gt -
        small_person_tp
    )

    print()
    print("=" * 72)
    print(
        "# RESULTADO PERSON SMALL FAILURE INTERACTIONS V1"
    )
    print("=" * 72)
    print()

    print(
        f"Imágenes:              {len(image_paths):,}"
    )

    print(
        f"PERSON GT:             {total_person_gt:,}"
    )

    print(
        f"PERSON TP:             {total_person_tp:,}"
    )

    print(
        f"PERSON FN:             {total_person_fn:,}"
    )

    print(
        f"PERSON Recall:         "
        f"{safe_div(total_person_tp, total_person_gt):.4f}"
    )

    print()

    print(
        f"SMALL PERSON GT:       {small_person_gt:,}"
    )

    print(
        f"SMALL PERSON TP:       {small_person_tp:,}"
    )

    print(
        f"SMALL PERSON FN:       {small_person_fn:,}"
    )

    print(
        f"SMALL PERSON Recall:   "
        f"{safe_div(small_person_tp, small_person_gt):.4f}"
    )

    print()

    print(
        f"SMALL FAILURE INTERACTIONS: {len(all_rows):,}"
    )

    print()

    # ------------------------------------------------------------------------
    # CSV principal
    # ------------------------------------------------------------------------

    objects_csv = (
        REPORTS_DIR /
        "person_small_failure_interactions_objects_v1.csv"
    )

    object_fields = [
        "image",
        "gt_index",
        "image_width",
        "image_height",
        "x1",
        "y1",
        "x2",
        "y2",
        "area",
        "size_bucket",
        "density",
        "neighbor_count",
        "proximity",
        "close_neighbor_count",
        "occlusion",
        "max_person_overlap",
        "overlapping_persons",
        "location",
        "prediction_relation",
        "best_iou",
        "factor_count",
        "factors",
        "interaction_class",
    ]

    write_csv(
        objects_csv,
        all_rows,
        object_fields,
    )

    print(
        f"[OK] {objects_csv}"
    )

    # ------------------------------------------------------------------------
    # FACTORES
    # ------------------------------------------------------------------------

    factor_stats = build_factor_statistics(
        all_rows
    )

    factor_csv = (
        REPORTS_DIR /
        "person_small_failure_interactions_by_factor_v1.csv"
    )

    write_csv(
        factor_csv,
        factor_stats,
        [
            "factor",
            "failures",
            "percentage",
        ],
    )

    print(
        f"[OK] {factor_csv}"
    )

    # ------------------------------------------------------------------------
    # INTERACCIONES
    # ------------------------------------------------------------------------

    interaction_stats = build_interaction_statistics(
        all_rows
    )

    interaction_csv = (
        REPORTS_DIR /
        "person_small_failure_interactions_v1.csv"
    )

    write_csv(
        interaction_csv,
        interaction_stats,
        [
            "interaction",
            "failures",
            "percentage",
        ],
    )

    print(
        f"[OK] {interaction_csv}"
    )

    # ------------------------------------------------------------------------
    # NUMERO DE FACTORES
    # ------------------------------------------------------------------------

    factor_count_stats = build_factor_count_statistics(
        all_rows
    )

    factor_count_csv = (
        REPORTS_DIR /
        "person_small_failure_factor_count_v1.csv"
    )

    write_csv(
        factor_count_csv,
        factor_count_stats,
        [
            "factor_count",
            "failures",
            "percentage",
        ],
    )

    print(
        f"[OK] {factor_count_csv}"
    )

    # ------------------------------------------------------------------------
    # MATRICES
    # ------------------------------------------------------------------------

    pair_definitions = [
        (
            "size_bucket",
            "proximity",
            "person_small_failure_size_x_proximity_v1.csv",
        ),
        (
            "size_bucket",
            "occlusion",
            "person_small_failure_size_x_occlusion_v1.csv",
        ),
        (
            "size_bucket",
            "prediction_relation",
            "person_small_failure_size_x_prediction_relation_v1.csv",
        ),
        (
            "proximity",
            "occlusion",
            "person_small_failure_proximity_x_occlusion_v1.csv",
        ),
        (
            "proximity",
            "prediction_relation",
            "person_small_failure_proximity_x_prediction_relation_v1.csv",
        ),
        (
            "density",
            "proximity",
            "person_small_failure_density_x_proximity_v1.csv",
        ),
        (
            "density",
            "occlusion",
            "person_small_failure_density_x_occlusion_v1.csv",
        ),
        (
            "location",
            "prediction_relation",
            "person_small_failure_location_x_prediction_relation_v1.csv",
        ),
    ]

    for field_a, field_b, filename in pair_definitions:

        matrix = build_pair_matrix(
            all_rows,
            field_a,
            field_b,
        )

        path = REPORTS_DIR / filename

        write_csv(
            path,
            matrix,
            [
                field_a,
                field_b,
                "failures",
                "percentage",
            ],
        )

        print(
            f"[OK] {path}"
        )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    summary_path = (
        REPORTS_DIR /
        "PERSON_SMALL_FAILURE_INTERACTIONS_V1_SUMMARY.txt"
    )

    write_summary(
        summary_path,
        len(image_paths),
        total_person_gt,
        total_person_tp,
        total_person_fn,
        small_person_gt,
        small_person_tp,
        small_person_fn,
        all_rows,
        factor_stats,
        interaction_stats,
        factor_count_stats,
    )

    print()
    print(
        f"[OK] {summary_path}"
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