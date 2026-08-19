from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26 - PERSON RECALL BY PERSON SIZE ANALYSIS V1
# ============================================================================

print()
print("=" * 72)
print("# SAR YOLO26 - PERSON RECALL BY PERSON SIZE ANALYSIS V1")
print("=" * 72)
print()


# ============================================================================
# CONFIGURATION
# ============================================================================

BASELINE_DIR = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_DIR = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = (
    BASELINE_DIR
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)

TEST_IMAGES_DIR = DATASET_DIR / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_DIR / "test_dev" / "labels"

OUTPUT_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "recall_by_person_size"
    / "analyze_person_recall_by_person_size_v1"
)

REPORTS_DIR = OUTPUT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# YOLO CONFIGURATION
# ============================================================================
#
# IMPORTANTE:
#
# Estos parámetros deben mantenerse iguales en los siguientes análisis para
# poder comparar los resultados correctamente.
#
# Si los scripts anteriores utilizan otros valores, cámbialos aquí para
# mantener una metodología única.
# ============================================================================

PERSON_CLASS_ID = 0
VEHICLE_CLASS_ID = 1

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.70

MATCH_IOU_THRESHOLD = 0.50

IMAGE_SIZE = 640


# ============================================================================
# PERSON SIZE BINS
# ============================================================================
#
# El área se calcula en píxeles cuadrados sobre la imagen original.
#
# ============================================================================

SIZE_BINS = [
    ("<16", 0, 16),
    ("16-32", 16, 32),
    ("32-64", 32, 64),
    ("64-128", 64, 128),
    ("128-256", 128, 256),
    ("256-512", 256, 512),
    ("512-1024", 512, 1024),
    ("1024-2048", 1024, 2048),
    (">2048", 2048, float("inf")),
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_iou(box_a, box_b):
    """
    Calcula IoU entre dos bounding boxes en formato:

        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def get_size_bin(area):
    """
    Devuelve el intervalo de tamaño correspondiente al área.
    """

    for name, lower, upper in SIZE_BINS:
        if lower <= area < upper:
            return name

    return ">2048"


def load_yolo_labels(label_path, image_width, image_height):
    """
    Carga labels YOLO:

        class x_center y_center width height

    y devuelve bounding boxes en píxeles:

        [x1, y1, x2, y2]
    """

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))

        if class_id != PERSON_CLASS_ID:
            continue

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        x1 = max(0.0, min(x1, image_width))
        y1 = max(0.0, min(y1, image_height))
        x2 = max(0.0, min(x2, image_width))
        y2 = max(0.0, min(y2, image_height))

        boxes.append({
            "class_id": class_id,
            "bbox": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1),
        })

    return boxes


def get_person_predictions(result):
    """
    Extrae únicamente predicciones PERSON.
    """

    predictions = []

    if result.boxes is None:
        return predictions

    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    for bbox, class_id, confidence in zip(
        xyxy,
        classes,
        confidences
    ):

        class_id = int(class_id)

        if class_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = bbox

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        predictions.append({
            "class_id": class_id,
            "bbox": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],
            "area": float(area),
            "confidence": float(confidence),
        })

    return predictions


def match_person_predictions(gt_boxes, predictions):
    """
    Matching greedy GT -> prediction utilizando IoU.

    Cada predicción solamente puede utilizarse una vez.
    """

    matches = []

    used_predictions = set()

    for gt_index, gt in enumerate(gt_boxes):

        best_prediction = None
        best_iou = 0.0

        for pred_index, pred in enumerate(predictions):

            if pred_index in used_predictions:
                continue

            iou = calculate_iou(
                gt["bbox"],
                pred["bbox"]
            )

            if iou > best_iou:
                best_iou = iou
                best_prediction = pred_index

        if (
            best_prediction is not None
            and best_iou >= MATCH_IOU_THRESHOLD
        ):

            used_predictions.add(best_prediction)

            matches.append({
                "gt_index": gt_index,
                "pred_index": best_prediction,
                "iou": best_iou,
                "matched": True,
            })

        else:

            matches.append({
                "gt_index": gt_index,
                "pred_index": None,
                "iou": best_iou,
                "matched": False,
            })

    return matches


# ============================================================================
# CHECK PATHS
# ============================================================================

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

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"No existe el dataset:\n{DATASET_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No existe el modelo:\n{MODEL_PATH}"
    )

if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"No existe el directorio de imágenes:\n{TEST_IMAGES_DIR}"
    )

if not TEST_LABELS_DIR.exists():
    raise FileNotFoundError(
        f"No existe el directorio de labels:\n{TEST_LABELS_DIR}"
    )


# ============================================================================
# FIND IMAGES
# ============================================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

image_paths = sorted(
    [
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ]
)

print(
    f"Imágenes encontradas: {len(image_paths)}"
)

print()


# ============================================================================
# LOAD MODEL
# ============================================================================

print("Cargando modelo YOLO26s...")

model = YOLO(str(MODEL_PATH))

print("[OK] Modelo cargado.")

print()


# ============================================================================
# STORAGE
# ============================================================================

object_records = []
image_records = []

size_statistics = defaultdict(
    lambda: {
        "images": set(),
        "gt": 0,
        "tp": 0,
        "fn": 0,
        "area_sum": 0.0,
        "area_values": [],
    }
)


# ============================================================================
# ANALYSIS
# ============================================================================

total_images = len(image_paths)

for index, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"[WARNING] No se pudo leer: {image_path.name}"
        )
        continue

    image_height, image_width = image.shape[:2]

    label_path = (
        TEST_LABELS_DIR
        / f"{image_path.stem}.txt"
    )

    gt_boxes = load_yolo_labels(
        label_path,
        image_width,
        image_height,
    )

    # ------------------------------------------------------------
    # YOLO inference
    # ------------------------------------------------------------

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_NMS_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    predictions = []

    if results:
        predictions = get_person_predictions(
            results[0]
        )

    # ------------------------------------------------------------
    # MATCHING
    # ------------------------------------------------------------

    matches = match_person_predictions(
        gt_boxes,
        predictions,
    )

    image_tp = 0
    image_fn = 0

    # ------------------------------------------------------------
    # PERSON OBJECTS
    # ------------------------------------------------------------

    for gt_index, gt in enumerate(gt_boxes):

        area = gt["area"]

        size_bin = get_size_bin(area)

        match = matches[gt_index]

        if match["matched"]:
            status = "TP"
            image_tp += 1
        else:
            status = "FN"
            image_fn += 1

        size_statistics[size_bin]["images"].add(
            image_path.name
        )

        size_statistics[size_bin]["gt"] += 1

        if status == "TP":
            size_statistics[size_bin]["tp"] += 1
        else:
            size_statistics[size_bin]["fn"] += 1

        size_statistics[size_bin]["area_sum"] += area
        size_statistics[size_bin]["area_values"].append(area)

        object_records.append({
            "image": image_path.name,
            "width": image_width,
            "height": image_height,
            "gt_index": gt_index,
            "person_area_px2": area,
            "person_size_bin": size_bin,
            "status": status,
            "matched_iou": match["iou"],
        })

    image_records.append({
        "image": image_path.name,
        "width": image_width,
        "height": image_height,
        "image_area_px2": image_width * image_height,
        "person_gt": len(gt_boxes),
        "person_tp": image_tp,
        "person_fn": image_fn,
        "person_recall": (
            image_tp / len(gt_boxes)
            if len(gt_boxes) > 0
            else 0.0
        ),
    })

    if (
        index % 100 == 0
        or index == total_images
    ):
        print(
            f"Analizadas: {index:,}/{total_images:,}"
        )


# ============================================================================
# DATAFRAME
# ============================================================================

objects_df = pd.DataFrame(
    object_records
)

images_df = pd.DataFrame(
    image_records
)


# ============================================================================
# SIZE ANALYSIS
# ============================================================================

size_rows = []

for size_name, lower, upper in SIZE_BINS:

    stats = size_statistics[size_name]

    gt = stats["gt"]
    tp = stats["tp"]
    fn = stats["fn"]

    recall = (
        tp / gt
        if gt > 0
        else 0.0
    )

    percentage = (
        gt / len(objects_df) * 100
        if len(objects_df) > 0
        else 0.0
    )

    areas = stats["area_values"]

    median_area = (
        float(np.median(areas))
        if areas
        else 0.0
    )

    mean_area = (
        float(np.mean(areas))
        if areas
        else 0.0
    )

    size_rows.append({
        "person_size_bin": size_name,
        "lower_area_px2": lower,
        "upper_area_px2": (
            upper
            if np.isfinite(upper)
            else None
        ),
        "images": len(stats["images"]),
        "gt": gt,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "percentage_of_person_gt": percentage,
        "mean_area_px2": mean_area,
        "median_area_px2": median_area,
    })


size_df = pd.DataFrame(
    size_rows
)


# ============================================================================
# GLOBAL STATISTICS
# ============================================================================

total_gt = len(objects_df)

total_tp = int(
    (objects_df["status"] == "TP").sum()
)

total_fn = int(
    (objects_df["status"] == "FN").sum()
)

global_recall = (
    total_tp / total_gt
    if total_gt > 0
    else 0.0
)


# ============================================================================
# PERSON SIZE STATISTICS
# ============================================================================

all_areas = objects_df[
    "person_area_px2"
].to_numpy()

statistics_rows = []

if len(all_areas) > 0:

    statistics_rows.append({
        "metric": "PERSON_GT",
        "value": total_gt,
    })

    statistics_rows.append({
        "metric": "PERSON_TP",
        "value": total_tp,
    })

    statistics_rows.append({
        "metric": "PERSON_FN",
        "value": total_fn,
    })

    statistics_rows.append({
        "metric": "PERSON_RECALL",
        "value": global_recall,
    })

    statistics_rows.append({
        "metric": "MIN_PERSON_AREA_PX2",
        "value": float(np.min(all_areas)),
    })

    statistics_rows.append({
        "metric": "P10_PERSON_AREA_PX2",
        "value": float(np.percentile(all_areas, 10)),
    })

    statistics_rows.append({
        "metric": "P25_PERSON_AREA_PX2",
        "value": float(np.percentile(all_areas, 25)),
    })

    statistics_rows.append({
        "metric": "MEDIAN_PERSON_AREA_PX2",
        "value": float(np.median(all_areas)),
    })

    statistics_rows.append({
        "metric": "P75_PERSON_AREA_PX2",
        "value": float(np.percentile(all_areas, 75)),
    })

    statistics_rows.append({
        "metric": "P90_PERSON_AREA_PX2",
        "value": float(np.percentile(all_areas, 90)),
    })

    statistics_rows.append({
        "metric": "P95_PERSON_AREA_PX2",
        "value": float(np.percentile(all_areas, 95)),
    })

    statistics_rows.append({
        "metric": "MAX_PERSON_AREA_PX2",
        "value": float(np.max(all_areas)),
    })


statistics_df = pd.DataFrame(
    statistics_rows
)


# ============================================================================
# SAVE REPORTS
# ============================================================================

objects_csv = (
    REPORTS_DIR
    / "person_recall_by_person_size_objects_v1.csv"
)

size_csv = (
    REPORTS_DIR
    / "person_recall_by_person_size_v1.csv"
)

statistics_csv = (
    REPORTS_DIR
    / "person_size_statistics_v1.csv"
)

summary_txt = (
    REPORTS_DIR
    / "PERSON_RECALL_BY_PERSON_SIZE_V1_SUMMARY.txt"
)


objects_df.to_csv(
    objects_csv,
    index=False,
    encoding="utf-8-sig",
)

size_df.to_csv(
    size_csv,
    index=False,
    encoding="utf-8-sig",
)

statistics_df.to_csv(
    statistics_csv,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================================
# SUMMARY
# ============================================================================

with open(
    summary_txt,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SAR YOLO26 - PERSON RECALL BY PERSON SIZE ANALYSIS V1\n"
    )

    f.write("=" * 72 + "\n\n")

    f.write(
        f"Dataset: {DATASET_DIR}\n"
    )

    f.write(
        f"Modelo: {MODEL_PATH}\n"
    )

    f.write(
        f"Test: {TEST_IMAGES_DIR}\n"
    )

    f.write(
        f"Imágenes: {len(images_df):,}\n"
    )

    f.write(
        f"PERSON GT: {total_gt:,}\n"
    )

    f.write(
        f"PERSON TP: {total_tp:,}\n"
    )

    f.write(
        f"PERSON FN: {total_fn:,}\n"
    )

    f.write(
        f"PERSON Recall: {global_recall:.4f}\n\n"
    )

    f.write(
        "CONFIGURACIÓN DE INFERENCIA\n"
    )

    f.write("-" * 72 + "\n")

    f.write(
        f"IMAGE_SIZE: {IMAGE_SIZE}\n"
    )

    f.write(
        f"CONF_THRESHOLD: {CONF_THRESHOLD}\n"
    )

    f.write(
        f"IOU_NMS_THRESHOLD: {IOU_NMS_THRESHOLD}\n"
    )

    f.write(
        f"MATCH_IOU_THRESHOLD: {MATCH_IOU_THRESHOLD}\n\n"
    )

    f.write(
        "ESTADÍSTICAS GLOBALES DE TAMAÑO\n"
    )

    f.write("-" * 72 + "\n")

    for _, row in statistics_df.iterrows():

        f.write(
            f"{row['metric']}: "
            f"{row['value']}\n"
        )

    f.write("\n")

    f.write(
        "RECALL POR TAMAÑO DE PERSONA\n"
    )

    f.write("-" * 72 + "\n")

    for _, row in size_df.iterrows():

        f.write(
            f"{row['person_size_bin']:>10} "
            f"GT={int(row['gt']):>7,} "
            f"TP={int(row['tp']):>7,} "
            f"FN={int(row['fn']):>7,} "
            f"Recall={row['recall']:.4f} "
            f"({row['percentage_of_person_gt']:6.2f} %)\n"
        )

    f.write("\n")

    f.write(
        "IMPORTANTE: el dataset NO ha sido modificado.\n"
    )


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

print()
print("=" * 72)
print("# RESULTADO PERSON RECALL BY PERSON SIZE V1")
print("=" * 72)
print()

print(
    f"Imágenes:              {len(images_df):,}"
)

print(
    f"PERSON GT:             {total_gt:,}"
)

print(
    f"PERSON TP:             {total_tp:,}"
)

print(
    f"PERSON FN:             {total_fn:,}"
)

print(
    f"PERSON Recall:         {global_recall:.4f}"
)

print()
print("RECALL POR TAMAÑO DE PERSONA")
print()

for _, row in size_df.iterrows():

    print(
        f"{row['person_size_bin']:>10} "
        f"GT={int(row['gt']):>7,} "
        f"TP={int(row['tp']):>7,} "
        f"FN={int(row['fn']):>7,} "
        f"Recall={row['recall']:.4f} "
        f"({row['percentage_of_person_gt']:6.2f} %)"
    )

print()

print(
    f"[OK] {objects_csv}"
)

print(
    f"[OK] {size_csv}"
)

print(
    f"[OK] {statistics_csv}"
)

print(
    f"[OK] {summary_txt}"
)

print()

print(
    "IMPORTANTE: el dataset NO ha sido modificado."
)

print()
print("=" * 72)