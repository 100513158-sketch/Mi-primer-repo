from pathlib import Path
from collections import defaultdict
import csv
import math

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_DIR = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
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


SCRIPT_DIR = (
    BASE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_residual"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "analyze_person_small_failure_residual_v1"
)

REPORTS_DIR = OUTPUT_DIR / "reports"

IMAGES_DIR = OUTPUT_DIR / "images"
TOP_FAILURES_DIR = IMAGES_DIR / "top_residual_failures"


# ============================================================================
# MODEL / ANALYSIS PARAMETERS
# ============================================================================

# Input scale selected from previous analysis.
INPUT_SIZE = 1536

# PERSON class in the SAR 2-class dataset.
PERSON_CLASS_ID = 0

# Confidence used to obtain candidate detections.
# A low value is intentional because we are studying FN.
CONF_THRESHOLD = 0.001

# IoU required to consider a GT PERSON detected.
MATCH_IOU_THRESHOLD = 0.50

# Maximum GT bounding-box area considered "small".
#
# Previous analysis used:
# <16
# 16-32
# 32-64
# 64-128
# 128-256
#
# Therefore, this residual analysis concentrates on:
# area < 256 pixels^2
SMALL_PERSON_MAX_AREA = 256.0

# Number of visual residual cases to save.
TOP_VISUAL_FAILURES = 100

# YOLO inference batch size.
BATCH_SIZE = 1

# Image extensions.
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
# TERMINAL HELPERS
# ============================================================================

def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_progress(current, total):
    if current % 100 == 0 or current == total:
        print(f"Analizadas: {current:,}/{total:,}")


# ============================================================================
# FILESYSTEM
# ============================================================================

def prepare_output_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TOP_FAILURES_DIR.mkdir(parents=True, exist_ok=True)


def find_images():
    images = [
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images.sort()

    return images


def get_label_path(image_path):
    return TEST_LABELS_DIR / f"{image_path.stem}.txt"


# ============================================================================
# LABELS
# ============================================================================

def load_person_labels(label_path):
    """
    Reads YOLO normalized labels.

    Expected:
        class_id x_center y_center width height

    Returns only PERSON objects.
    """

    objects = []

    if not label_path.exists():
        return objects

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return objects

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        if class_id != PERSON_CLASS_ID:
            continue

        objects.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "line_number": line_number,
            }
        )

    return objects


def yolo_to_xyxy(obj, image_width, image_height):
    x_center = obj["x_center"] * image_width
    y_center = obj["y_center"] * image_height

    width = obj["width"] * image_width
    height = obj["height"] * image_height

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))

    return [x1, y1, x2, y2]


# ============================================================================
# GEOMETRY
# ============================================================================

def box_area(box):
    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================================
# PERSON SIZE
# ============================================================================

def get_size_bucket(area):

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


def is_small_person(area):
    return area < SMALL_PERSON_MAX_AREA


# ============================================================================
# DENSITY
# ============================================================================

def get_density_bucket(person_count):

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


# ============================================================================
# LOCATION
# ============================================================================

def get_location_bucket(cx, cy, image_width, image_height):

    normalized_x = cx / image_width if image_width > 0 else 0.5
    normalized_y = cy / image_height if image_height > 0 else 0.5

    if normalized_x < 0.5 and normalized_y < 0.5:
        return "top_left"

    if normalized_x >= 0.5 and normalized_y < 0.5:
        return "top_right"

    if normalized_x < 0.5 and normalized_y >= 0.5:
        return "bottom_left"

    return "bottom_right"


# ============================================================================
# MODEL DETECTIONS
# ============================================================================

def get_person_detections(result):

    detections = []

    if result.boxes is None:
        return detections

    boxes = result.boxes

    if boxes.xyxy is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()

    if boxes.conf is not None:
        confs = boxes.conf.cpu().numpy()
    else:
        confs = np.ones(len(xyxy), dtype=float)

    if boxes.cls is not None:
        classes = boxes.cls.cpu().numpy()
    else:
        classes = np.zeros(len(xyxy), dtype=float)

    for box, conf, cls_id in zip(xyxy, confs, classes):

        if int(cls_id) != PERSON_CLASS_ID:
            continue

        detections.append(
            {
                "box": box.tolist(),
                "confidence": float(conf),
                "class_id": int(cls_id),
            }
        )

    return detections


# ============================================================================
# MATCHING
# ============================================================================

def match_person_objects(gt_objects, detections):

    """
    One-to-one greedy matching.

    Each GT PERSON can be matched only once.
    Each detection can be matched only once.
    """

    candidates = []

    for gt_index, gt in enumerate(gt_objects):

        for det_index, det in enumerate(detections):

            iou = calculate_iou(
                gt["box"],
                det["box"],
            )

            if iou >= MATCH_IOU_THRESHOLD:

                candidates.append(
                    (
                        iou,
                        gt_index,
                        det_index,
                    )
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_det = set()

    matches = []

    for iou, gt_index, det_index in candidates:

        if gt_index in matched_gt:
            continue

        if det_index in matched_det:
            continue

        matched_gt.add(gt_index)
        matched_det.add(det_index)

        matches.append(
            {
                "gt_index": gt_index,
                "det_index": det_index,
                "iou": iou,
            }
        )

    return matches, matched_gt, matched_det


# ============================================================================
# IMAGE VISUALIZATION
# ============================================================================

def draw_residual_failure(
    image,
    gt_box,
    image_name,
    area,
    size_bucket,
    density_bucket,
):

    canvas = image.copy()

    x1, y1, x2, y2 = [
        int(round(v))
        for v in gt_box
    ]

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2,
    )

    text_lines = [
        "PERSON FN - SMALL RESIDUAL",
        f"area={area:.2f}",
        f"size={size_bucket}",
        f"density={density_bucket}",
        image_name,
    ]

    y = 30

    for text in text_lines:

        cv2.putText(
            canvas,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        y += 28

    return canvas


# ============================================================================
# CSV WRITERS
# ============================================================================

def write_csv(path, rows, fieldnames):

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


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_image(
    model,
    image_path,
):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image_height, image_width = image.shape[:2]

    label_path = get_label_path(image_path)

    raw_gt = load_person_labels(label_path)

    gt_objects = []

    for gt in raw_gt:

        box = yolo_to_xyxy(
            gt,
            image_width,
            image_height,
        )

        area = box_area(box)

        cx, cy = box_center(box)

        gt_objects.append(
            {
                "box": box,
                "area": area,
                "size_bucket": get_size_bucket(area),
                "cx": cx,
                "cy": cy,
                "location": get_location_bucket(
                    cx,
                    cy,
                    image_width,
                    image_height,
                ),
            }
        )

    person_count = len(gt_objects)

    density_bucket = get_density_bucket(person_count)

    # ------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=INPUT_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    if not results:
        detections = []
    else:
        detections = get_person_detections(results[0])

    # ------------------------------------------------------------
    # MATCHING
    # ------------------------------------------------------------

    matches, matched_gt, matched_det = match_person_objects(
        gt_objects,
        detections,
    )

    match_by_gt = {
        m["gt_index"]: m
        for m in matches
    }

    residuals = []

    for gt_index, gt in enumerate(gt_objects):

        # Only small persons.
        if not is_small_person(gt["area"]):
            continue

        # If GT was matched, it is not a residual failure.
        if gt_index in matched_gt:
            continue

        residuals.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_width": image_width,
                "image_height": image_height,
                "person_count": person_count,
                "density_bucket": density_bucket,
                "gt_index": gt_index,
                "area": gt["area"],
                "size_bucket": gt["size_bucket"],
                "width": gt["box"][2] - gt["box"][0],
                "height": gt["box"][3] - gt["box"][1],
                "center_x": gt["cx"],
                "center_y": gt["cy"],
                "center_x_norm": gt["cx"] / image_width,
                "center_y_norm": gt["cy"] / image_height,
                "location": gt["location"],
                "input_size": INPUT_SIZE,
                "iou": 0.0,
                "matched": 0,
            }
        )

    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "person_gt": len(gt_objects),
        "person_tp": len(matched_gt),
        "person_fn": len(gt_objects) - len(matched_gt),
        "person_detections": len(detections),
        "density_bucket": density_bucket,
        "residuals": residuals,
        "image": image,
    }


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_by_size(residuals):

    groups = defaultdict(
        lambda: {
            "gt_small": 0,
            "residual_fn": 0,
        }
    )

    # Every small object considered belongs to GT population.
    # Residuals contain only unmatched objects.
    #
    # We therefore need the complete GT counts separately in main().

    for row in residuals:

        key = row["size_bucket"]

        groups[key]["residual_fn"] += 1

    return groups


def aggregate_by_density(residuals):

    groups = defaultdict(
        lambda: {
            "residual_fn": 0,
        }
    )

    for row in residuals:

        groups[row["density_bucket"]]["residual_fn"] += 1

    return groups


def aggregate_by_location(residuals):

    groups = defaultdict(
        lambda: {
            "residual_fn": 0,
        }
    )

    for row in residuals:

        groups[row["location"]]["residual_fn"] += 1

    return groups


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "# SAR YOLO26 - PERSON SMALL FAILURE RESIDUAL ANALYSIS V1"
    )

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
    print(f"Input scale: {INPUT_SIZE}")
    print(f"Small PERSON area: < {SMALL_PERSON_MAX_AREA}")
    print(f"Match IoU threshold: {MATCH_IOU_THRESHOLD}")

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"No existe el dataset:\n{DATASET_DIR}"
        )

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio de imágenes:\n"
            f"{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio de labels:\n"
            f"{TEST_LABELS_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo:\n{MODEL_PATH}"
        )

    prepare_output_dirs()

    images = find_images()

    print()
    print(f"Imágenes encontradas: {len(images):,}")

    if not images:
        raise RuntimeError(
            "No se encontraron imágenes de test."
        )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    print()
    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL_PATH))

    print("[OK] Modelo cargado.")

    # ------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------

    print()
    print("Analizando SMALL PERSON residual failures...")

    all_residuals = []

    total_images = len(images)

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    total_small_gt = 0
    total_small_tp = 0
    total_small_fn = 0

    image_rows = []

    # Complete GT statistics by size.
    size_statistics = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    density_statistics = defaultdict(
        lambda: {
            "small_gt": 0,
            "small_tp": 0,
            "small_fn": 0,
        }
    )

    location_statistics = defaultdict(
        lambda: {
            "small_gt": 0,
            "small_tp": 0,
            "small_fn": 0,
        }
    )

    visual_candidates = []

    for index, image_path in enumerate(images, start=1):

        result = analyze_image(
            model,
            image_path,
        )

        if result is None:
            continue

        total_person_gt += result["person_gt"]
        total_person_tp += result["person_tp"]
        total_person_fn += result["person_fn"]

        image_rows.append(
            {
                "image_name": result["image_name"],
                "person_gt": result["person_gt"],
                "person_tp": result["person_tp"],
                "person_fn": result["person_fn"],
                "person_detections": result["person_detections"],
                "density_bucket": result["density_bucket"],
                "small_residual_fn": len(result["residuals"]),
            }
        )

        # --------------------------------------------------------
        # Reconstruct complete small GT statistics.
        # --------------------------------------------------------

        label_path = get_label_path(image_path)

        raw_gt = load_person_labels(label_path)

        image_height = result["image_height"]
        image_width = result["image_width"]

        # Re-run GT conversion only.
        gt_objects = []

        for gt in raw_gt:

            box = yolo_to_xyxy(
                gt,
                image_width,
                image_height,
            )

            area = box_area(box)

            cx, cy = box_center(box)

            gt_objects.append(
                {
                    "box": box,
                    "area": area,
                    "size_bucket": get_size_bucket(area),
                    "cx": cx,
                    "cy": cy,
                    "location": get_location_bucket(
                        cx,
                        cy,
                        image_width,
                        image_height,
                    ),
                }
            )

        # Inference again to obtain matching information.
        inference_results = model.predict(
            source=str(image_path),
            imgsz=INPUT_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        if inference_results:
            detections = get_person_detections(
                inference_results[0]
            )
        else:
            detections = []

        matches, matched_gt, matched_det = match_person_objects(
            gt_objects,
            detections,
        )

        for gt_index, gt in enumerate(gt_objects):

            size_bucket = gt["size_bucket"]

            if is_small_person(gt["area"]):

                total_small_gt += 1

                size_statistics[size_bucket]["gt"] += 1

                density_bucket = result["density_bucket"]

                density_statistics[density_bucket][
                    "small_gt"
                ] += 1

                location = gt["location"]

                location_statistics[location][
                    "small_gt"
                ] += 1

                if gt_index in matched_gt:

                    total_small_tp += 1

                    size_statistics[size_bucket]["tp"] += 1

                    density_statistics[density_bucket][
                        "small_tp"
                    ] += 1

                    location_statistics[location][
                        "small_tp"
                    ] += 1

                else:

                    total_small_fn += 1

                    size_statistics[size_bucket]["fn"] += 1

                    density_statistics[density_bucket][
                        "small_fn"
                    ] += 1

                    location_statistics[location][
                        "small_fn"
                    ] += 1

        # --------------------------------------------------------
        # Residual objects.
        # --------------------------------------------------------

        for residual in result["residuals"]:

            all_residuals.append(residual)

            visual_candidates.append(
                (
                    residual["area"],
                    result["image_name"],
                    result["image"],
                    residual,
                )
            )

        print_progress(
            index,
            total_images,
        )

    # ------------------------------------------------------------
    # GLOBAL METRICS
    # ------------------------------------------------------------

    global_recall = (
        total_person_tp / total_person_gt
        if total_person_gt > 0
        else 0.0
    )

    small_recall = (
        total_small_tp / total_small_gt
        if total_small_gt > 0
        else 0.0
    )

    small_failure_rate = (
        total_small_fn / total_small_gt
        if total_small_gt > 0
        else 0.0
    )

    # ------------------------------------------------------------
    # OBJECT CSV
    # ------------------------------------------------------------

    objects_csv = (
        REPORTS_DIR
        / "person_small_failure_residual_objects_v1.csv"
    )

    object_fields = [
        "image_name",
        "image_path",
        "image_width",
        "image_height",
        "person_count",
        "density_bucket",
        "gt_index",
        "area",
        "size_bucket",
        "width",
        "height",
        "center_x",
        "center_y",
        "center_x_norm",
        "center_y_norm",
        "location",
        "input_size",
        "iou",
        "matched",
    ]

    write_csv(
        objects_csv,
        all_residuals,
        object_fields,
    )

    print()
    print(f"[OK] {objects_csv}")

    # ------------------------------------------------------------
    # SIZE CSV
    # ------------------------------------------------------------

    size_order = [
        "<16",
        "16-32",
        "32-64",
        "64-128",
        "128-256",
    ]

    size_rows = []

    for bucket in size_order:

        data = size_statistics[bucket]

        gt = data["gt"]
        tp = data["tp"]
        fn = data["fn"]

        recall = (
            tp / gt
            if gt > 0
            else 0.0
        )

        residual_rate = (
            fn / gt
            if gt > 0
            else 0.0
        )

        size_rows.append(
            {
                "size_bucket": bucket,
                "person_gt": gt,
                "person_tp": tp,
                "person_fn": fn,
                "recall": f"{recall:.6f}",
                "residual_failure_rate": f"{residual_rate:.6f}",
            }
        )

    size_csv = (
        REPORTS_DIR
        / "person_small_failure_residual_by_size_v1.csv"
    )

    write_csv(
        size_csv,
        size_rows,
        [
            "size_bucket",
            "person_gt",
            "person_tp",
            "person_fn",
            "recall",
            "residual_failure_rate",
        ],
    )

    print(f"[OK] {size_csv}")

    # ------------------------------------------------------------
    # DENSITY CSV
    # ------------------------------------------------------------

    density_order = [
        "<25",
        "25-49",
        "50-99",
        "100-199",
        "200-299",
        "300-499+",
    ]

    density_rows = []

    for bucket in density_order:

        data = density_statistics[bucket]

        gt = data["small_gt"]
        tp = data["small_tp"]
        fn = data["small_fn"]

        recall = (
            tp / gt
            if gt > 0
            else 0.0
        )

        density_rows.append(
            {
                "density_bucket": bucket,
                "small_person_gt": gt,
                "small_person_tp": tp,
                "small_person_fn": fn,
                "recall": f"{recall:.6f}",
            }
        )

    density_csv = (
        REPORTS_DIR
        / "person_small_failure_residual_by_density_v1.csv"
    )

    write_csv(
        density_csv,
        density_rows,
        [
            "density_bucket",
            "small_person_gt",
            "small_person_tp",
            "small_person_fn",
            "recall",
        ],
    )

    print(f"[OK] {density_csv}")

    # ------------------------------------------------------------
    # LOCATION CSV
    # ------------------------------------------------------------

    location_order = [
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]

    location_rows = []

    for location in location_order:

        data = location_statistics[location]

        gt = data["small_gt"]
        tp = data["small_tp"]
        fn = data["small_fn"]

        recall = (
            tp / gt
            if gt > 0
            else 0.0
        )

        location_rows.append(
            {
                "location": location,
                "small_person_gt": gt,
                "small_person_tp": tp,
                "small_person_fn": fn,
                "recall": f"{recall:.6f}",
            }
        )

    location_csv = (
        REPORTS_DIR
        / "person_small_failure_residual_by_location_v1.csv"
    )

    write_csv(
        location_csv,
        location_rows,
        [
            "location",
            "small_person_gt",
            "small_person_tp",
            "small_person_fn",
            "recall",
        ],
    )

    print(f"[OK] {location_csv}")

    # ------------------------------------------------------------
    # VISUAL RESIDUALS
    # ------------------------------------------------------------

    # Smallest objects first.
    visual_candidates.sort(
        key=lambda x: x[0]
    )

    visual_candidates = visual_candidates[
        :TOP_VISUAL_FAILURES
    ]

    saved_visuals = 0

    for (
        area,
        image_name,
        image,
        residual,
    ) in visual_candidates:

        annotated = draw_residual_failure(
            image=image,
            gt_box=[
                residual["center_x"]
                - residual["width"] / 2.0,
                residual["center_y"]
                - residual["height"] / 2.0,
                residual["center_x"]
                + residual["width"] / 2.0,
                residual["center_y"]
                + residual["height"] / 2.0,
            ],
            image_name=image_name,
            area=area,
            size_bucket=residual["size_bucket"],
            density_bucket=residual["density_bucket"],
        )

        output_name = (
            f"{saved_visuals + 1:04d}_"
            f"{Path(image_name).stem}_"
            f"{residual['size_bucket'].replace('-', '_')}_"
            f"{int(round(area))}.jpg"
        )

        output_path = (
            TOP_FAILURES_DIR
            / output_name
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        saved_visuals += 1

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    summary_path = (
        REPORTS_DIR
        / "PERSON_SMALL_FAILURE_RESIDUAL_V1_SUMMARY.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON SMALL FAILURE RESIDUAL ANALYSIS V1\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            f"Dataset: {DATASET_DIR}\n"
        )

        f.write(
            f"Modelo: {MODEL_PATH}\n"
        )

        f.write(
            f"Input scale: {INPUT_SIZE}\n"
        )

        f.write(
            f"Small PERSON threshold: < {SMALL_PERSON_MAX_AREA} px²\n"
        )

        f.write(
            f"Match IoU threshold: {MATCH_IOU_THRESHOLD}\n\n"
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

        f.write(
            f"PERSON Recall: {global_recall:.4f}\n\n"
        )

        f.write(
            "SMALL PERSON\n"
        )

        f.write("-" * 72 + "\n")

        f.write(
            f"Small PERSON GT: {total_small_gt:,}\n"
        )

        f.write(
            f"Small PERSON TP: {total_small_tp:,}\n"
        )

        f.write(
            f"Small PERSON FN: {total_small_fn:,}\n"
        )

        f.write(
            f"Small PERSON Recall: {small_recall:.4f}\n"
        )

        f.write(
            f"Small PERSON residual failure rate: "
            f"{small_failure_rate:.4f}\n\n"
        )

        f.write(
            "RESIDUAL POR TAMAÑO\n"
        )

        f.write("-" * 72 + "\n")

        for row in size_rows:

            f.write(
                f"{row['size_bucket']:>8} "
                f"GT={int(row['person_gt']):>6,} "
                f"TP={int(row['person_tp']):>6,} "
                f"FN={int(row['person_fn']):>6,} "
                f"Recall={float(row['recall']):.4f}\n"
            )

        f.write("\n")

        f.write(
            "RESIDUAL POR DENSIDAD\n"
        )

        f.write("-" * 72 + "\n")

        for row in density_rows:

            f.write(
                f"{row['density_bucket']:>8} "
                f"GT={int(row['small_person_gt']):>6,} "
                f"TP={int(row['small_person_tp']):>6,} "
                f"FN={int(row['small_person_fn']):>6,} "
                f"Recall={float(row['recall']):.4f}\n"
            )

        f.write("\n")

        f.write(
            "RESIDUAL POR LOCALIZACION\n"
        )

        f.write("-" * 72 + "\n")

        for row in location_rows:

            f.write(
                f"{row['location']:>15} "
                f"GT={int(row['small_person_gt']):>6,} "
                f"TP={int(row['small_person_tp']):>6,} "
                f"FN={int(row['small_person_fn']):>6,} "
                f"Recall={float(row['recall']):.4f}\n"
            )

        f.write("\n")

        f.write(
            f"Visualizaciones generadas: {saved_visuals}\n"
        )

        f.write("\n")
        f.write(
            "IMPORTANTE: el dataset NO ha sido modificado.\n"
        )

    print(f"[OK] {summary_path}")

    # ------------------------------------------------------------
    # FINAL CONSOLE RESULT
    # ------------------------------------------------------------

    print_header(
        "# RESULTADO PERSON SMALL FAILURE RESIDUAL V1"
    )

    print()

    print(
        f"Imágenes:              {total_images:,}"
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
        f"PERSON Recall:         {global_recall:.4f}"
    )

    print()

    print(
        f"SMALL PERSON GT:       {total_small_gt:,}"
    )

    print(
        f"SMALL PERSON TP:       {total_small_tp:,}"
    )

    print(
        f"SMALL PERSON FN:       {total_small_fn:,}"
    )

    print(
        f"SMALL PERSON Recall:   {small_recall:.4f}"
    )

    print()

    print(
        "RESIDUAL POR TAMAÑO"
    )

    print()

    for row in size_rows:

        print(
            f"{row['size_bucket']:>8} "
            f"GT={int(row['person_gt']):>6,} "
            f"TP={int(row['person_tp']):>6,} "
            f"FN={int(row['person_fn']):>6,} "
            f"Recall={float(row['recall']):.4f}"
        )

    print()

    print(
        f"Fallos residuales visualizados: "
        f"{saved_visuals}"
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