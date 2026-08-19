from pathlib import Path
from collections import defaultdict
import csv
import shutil
import math

from ultralytics import YOLO


# ============================================================================
# SAR YOLO26 - DETECTION FAILURE ANALYSIS V1
# ============================================================================
#
# Objetivo:
#   Analizar los fallos de detección del baseline YOLO26s sobre test_dev.
#
#   GT (Ground Truth)  <-----> Predicciones YOLO26s
#
#   Se analizan:
#       - TP
#       - FN
#       - FP
#       - errores de clase
#       - IoU
#       - confidence
#       - tamaño del objeto
#       - objetos pequeños
#       - objetos cerca del borde
#
# IMPORTANTE:
#   Este script NO modifica el dataset.
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)

TEST_IMAGES = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS = DATASET_ROOT / "test_dev" / "labels"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "analyze_detection_failures_v1"
)

PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

REPORTS_DIR = OUTPUT_ROOT / "reports"


# ============================================================================
# PARÁMETROS DEL EXPERIMENTO
# ============================================================================

CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.50

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

CLASS_NAMES = {
    0: "PERSON",
    1: "VEHICLE",
}


# ============================================================================
# RANGOS DE ÁREA
# ============================================================================

AREA_BINS = [
    ("<16", 0, 16),
    ("16-32", 16, 32),
    ("32-64", 32, 64),
    ("64-128", 64, 128),
    ("128-256", 128, 256),
    ("256-512", 256, 512),
    ("512-1024", 512, 1024),
    (">1024", 1024, float("inf")),
]


# ============================================================================
# UTILIDADES
# ============================================================================

def ensure_directories():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h

    bw = w * img_w
    bh = h * img_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return x1, y1, x2, y2


def bbox_area(box):
    x1, y1, x2, y2 = box

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    return w * h


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def get_area_bin(area):
    for name, low, high in AREA_BINS:
        if low <= area < high:
            return name

    return ">1024"


def is_near_border(box, img_w, img_h, margin_ratio=0.02):
    x1, y1, x2, y2 = box

    margin_x = img_w * margin_ratio
    margin_y = img_h * margin_ratio

    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= img_w - margin_x
        or y2 >= img_h - margin_y
    )


def is_partial_bbox(box, img_w, img_h):
    x1, y1, x2, y2 = box

    return (
        x1 < 0
        or y1 < 0
        or x2 > img_w
        or y2 > img_h
    )


def load_ground_truth(label_path, img_w, img_h):
    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                cls = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            box = xywhn_to_xyxy(
                x,
                y,
                w,
                h,
                img_w,
                img_h,
            )

            objects.append(
                {
                    "gt_index": len(objects),
                    "class_id": cls,
                    "class_name": CLASS_NAMES.get(
                        cls,
                        f"CLASS_{cls}",
                    ),
                    "bbox": box,
                    "area": bbox_area(box),
                    "area_bin": get_area_bin(
                        bbox_area(box)
                    ),
                    "near_border": is_near_border(
                        box,
                        img_w,
                        img_h,
                    ),
                    "partial_bbox": is_partial_bbox(
                        box,
                        img_w,
                        img_h,
                    ),
                    "matched": False,
                }
            )

    return objects


# ============================================================================
# IMAGE DISCOVERY
# ============================================================================

def find_images():
    images = []

    for path in TEST_IMAGES.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images.append(path)

    images.sort()

    return images


# ============================================================================
# PREDICTION
# ============================================================================

def run_predictions(model, images):

    print()
    print("Generando predicciones YOLO26s...")
    print()

    prediction_records = {}

    total = len(images)

    for index, image_path in enumerate(images, start=1):

        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False,
            save=False,
        )

        result = results[0]

        boxes = []

        if result.boxes is not None:

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(
                xyxy,
                confs,
                classes,
            ):

                boxes.append(
                    {
                        "pred_index": len(boxes),
                        "class_id": int(cls),
                        "class_name": CLASS_NAMES.get(
                            int(cls),
                            f"CLASS_{int(cls)}",
                        ),
                        "bbox": tuple(
                            float(v) for v in box
                        ),
                        "confidence": float(conf),
                    }
                )

        prediction_records[str(image_path)] = boxes

        if index % 100 == 0 or index == total:
            print(
                f"Predicciones: {index:,}/{total:,}"
            )

    return prediction_records


# ============================================================================
# MATCHING
# ============================================================================

def match_predictions(gt_objects, predictions):

    matches = []

    candidate_pairs = []

    for gt in gt_objects:

        for pred in predictions:

            iou = bbox_iou(
                gt["bbox"],
                pred["bbox"],
            )

            if iou >= IOU_MATCH_THRESHOLD:

                candidate_pairs.append(
                    (
                        iou,
                        gt["gt_index"],
                        pred["pred_index"],
                    )
                )

    candidate_pairs.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    used_gt = set()
    used_pred = set()

    for iou, gt_index, pred_index in candidate_pairs:

        if gt_index in used_gt:
            continue

        if pred_index in used_pred:
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)

        gt = gt_objects[gt_index]
        pred = predictions[pred_index]

        same_class = (
            gt["class_id"] == pred["class_id"]
        )

        matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "iou": iou,
                "same_class": same_class,
                "gt_class": gt["class_name"],
                "pred_class": pred["class_name"],
                "confidence": pred["confidence"],
            }
        )

    return matches, used_gt, used_pred


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze():

    print("=" * 72)
    print("# SAR YOLO26 - DETECTION FAILURE ANALYSIS V1")
    print("=" * 72)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Modelo:")
    print(MODEL_PATH)

    print()
    print("Test:")
    print(TEST_IMAGES)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()

    ensure_directories()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No existe el modelo:\n{MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():

        raise FileNotFoundError(
            f"No existe test_dev/images:\n{TEST_IMAGES}"
        )

    if not TEST_LABELS.exists():

        raise FileNotFoundError(
            f"No existe test_dev/labels:\n{TEST_LABELS}"
        )

    images = find_images()

    print(
        f"Imágenes encontradas: {len(images):,}"
    )

    if not images:

        raise RuntimeError(
            "No se encontraron imágenes."
        )

    print()
    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL_PATH))

    print("[OK] Modelo cargado.")

    # ------------------------------------------------------------------------
    # GENERAR PREDICCIONES
    # ------------------------------------------------------------------------

    predictions = run_predictions(
        model,
        images,
    )

    # ------------------------------------------------------------------------
    # VARIABLES GLOBALES
    # ------------------------------------------------------------------------

    object_rows = []
    image_rows = []

    area_statistics = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    class_statistics = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
        }
    )

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_class_errors = 0

    # ------------------------------------------------------------------------
    # ANALIZAR CADA IMAGEN
    # ------------------------------------------------------------------------

    for image_index, image_path in enumerate(
        images,
        start=1,
    ):

        from PIL import Image

        with Image.open(image_path) as img:

            img_w, img_h = img.size

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        gt_objects = load_ground_truth(
            label_path,
            img_w,
            img_h,
        )

        image_predictions = predictions.get(
            str(image_path),
            [],
        )

        matches, used_gt, used_pred = match_predictions(
            gt_objects,
            image_predictions,
        )

        image_tp = 0
        image_fn = 0
        image_fp = 0
        image_class_errors = 0

        # --------------------------------------------------------------------
        # GT
        # --------------------------------------------------------------------

        for gt in gt_objects:

            total_gt += 1

            cls = gt["class_name"]
            area_bin = gt["area_bin"]

            class_statistics[cls]["gt"] += 1

            area_statistics[
                (cls, area_bin)
            ]["gt"] += 1

            matching = None

            for match in matches:

                if (
                    match["gt_index"]
                    == gt["gt_index"]
                ):
                    matching = match
                    break

            if matching is None:

                total_fn += 1
                image_fn += 1

                class_statistics[cls]["fn"] += 1

                area_statistics[
                    (cls, area_bin)
                ]["fn"] += 1

                object_rows.append(
                    {
                        "split": "test_dev",
                        "image": str(image_path),
                        "gt_index": gt["gt_index"],
                        "gt_class": cls,
                        "gt_area": gt["area"],
                        "area_bin": area_bin,
                        "near_border": int(
                            gt["near_border"]
                        ),
                        "partial_bbox": int(
                            gt["partial_bbox"]
                        ),
                        "status": "FN",
                        "pred_class": "",
                        "confidence": "",
                        "iou": "",
                        "error_type": "MISSED",
                    }
                )

                continue

            pred = image_predictions[
                matching["pred_index"]
            ]

            iou = matching["iou"]
            confidence = matching[
                "confidence"
            ]

            if matching["same_class"]:

                total_tp += 1
                image_tp += 1

                class_statistics[
                    cls
                ]["tp"] += 1

                area_statistics[
                    (cls, area_bin)
                ]["tp"] += 1

                status = "TP"
                error_type = ""

            else:

                total_fn += 1
                image_fn += 1

                total_class_errors += 1
                image_class_errors += 1

                class_statistics[
                    cls
                ]["fn"] += 1

                area_statistics[
                    (cls, area_bin)
                ]["fn"] += 1

                status = "CLASS_ERROR"
                error_type = (
                    f"{cls}_AS_"
                    f"{pred['class_name']}"
                )

            object_rows.append(
                {
                    "split": "test_dev",
                    "image": str(image_path),
                    "gt_index": gt["gt_index"],
                    "gt_class": cls,
                    "gt_area": gt["area"],
                    "area_bin": area_bin,
                    "near_border": int(
                        gt["near_border"]
                    ),
                    "partial_bbox": int(
                        gt["partial_bbox"]
                    ),
                    "status": status,
                    "pred_class": pred[
                        "class_name"
                    ],
                    "confidence": confidence,
                    "iou": iou,
                    "error_type": error_type,
                }
            )

        # --------------------------------------------------------------------
        # PREDICCIONES SIN GT
        # --------------------------------------------------------------------

        for pred_index, pred in enumerate(
            image_predictions
        ):

            total_pred += 1

            if pred_index in used_pred:
                continue

            total_fp += 1
            image_fp += 1

            class_statistics[
                pred["class_name"]
            ]["fp"] += 1

            object_rows.append(
                {
                    "split": "test_dev",
                    "image": str(image_path),
                    "gt_index": "",
                    "gt_class": "",
                    "gt_area": "",
                    "area_bin": "",
                    "near_border": "",
                    "partial_bbox": "",
                    "status": "FP",
                    "pred_class": pred[
                        "class_name"
                    ],
                    "confidence": pred[
                        "confidence"
                    ],
                    "iou": "",
                    "error_type": "FALSE_POSITIVE",
                }
            )

        image_rows.append(
            {
                "split": "test_dev",
                "image": str(image_path),
                "gt_objects": len(gt_objects),
                "predictions": len(
                    image_predictions
                ),
                "tp": image_tp,
                "fn": image_fn,
                "fp": image_fp,
                "class_errors": image_class_errors,
            }
        )

        if image_index % 100 == 0 or image_index == len(images):

            print(
                f"Analizadas: "
                f"{image_index:,}/{len(images):,}"
            )

    # ------------------------------------------------------------------------
    # EXPORTAR OBJETOS
    # ------------------------------------------------------------------------

    objects_csv = (
        REPORTS_DIR
        / "detection_objects_v1.csv"
    )

    object_fields = [
        "split",
        "image",
        "gt_index",
        "gt_class",
        "gt_area",
        "area_bin",
        "near_border",
        "partial_bbox",
        "status",
        "pred_class",
        "confidence",
        "iou",
        "error_type",
    ]

    with open(
        objects_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=object_fields,
        )

        writer.writeheader()
        writer.writerows(object_rows)

    # ------------------------------------------------------------------------
    # EXPORTAR IMÁGENES
    # ------------------------------------------------------------------------

    images_csv = (
        REPORTS_DIR
        / "detection_images_v1.csv"
    )

    image_fields = [
        "split",
        "image",
        "gt_objects",
        "predictions",
        "tp",
        "fn",
        "fp",
        "class_errors",
    ]

    with open(
        images_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=image_fields,
        )

        writer.writeheader()
        writer.writerows(image_rows)

    # ------------------------------------------------------------------------
    # RECALL POR ÁREA
    # ------------------------------------------------------------------------

    area_csv = (
        REPORTS_DIR
        / "recall_by_area_v1.csv"
    )

    with open(
        area_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "class",
            "area_bin",
            "gt",
            "tp",
            "fn",
            "recall",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for cls in [
            "PERSON",
            "VEHICLE",
        ]:

            for area_name, _, _ in AREA_BINS:

                stats = area_statistics[
                    (cls, area_name)
                ]

                gt = stats["gt"]
                tp = stats["tp"]
                fn = stats["fn"]

                recall = (
                    tp / gt
                    if gt > 0
                    else 0
                )

                writer.writerow(
                    {
                        "class": cls,
                        "area_bin": area_name,
                        "gt": gt,
                        "tp": tp,
                        "fn": fn,
                        "recall": recall,
                    }
                )

    # ------------------------------------------------------------------------
    # RECALL POR CLASE
    # ------------------------------------------------------------------------

    class_csv = (
        REPORTS_DIR
        / "recall_by_class_v1.csv"
    )

    with open(
        class_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "class",
            "gt",
            "tp",
            "fn",
            "fp",
            "recall",
            "precision",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for cls in [
            "PERSON",
            "VEHICLE",
        ]:

            stats = class_statistics[cls]

            gt = stats["gt"]
            tp = stats["tp"]
            fn = stats["fn"]
            fp = stats["fp"]

            recall = (
                tp / gt
                if gt > 0
                else 0
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            writer.writerow(
                {
                    "class": cls,
                    "gt": gt,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "recall": recall,
                    "precision": precision,
                }
            )

    # ------------------------------------------------------------------------
    # FALSE NEGATIVES
    # ------------------------------------------------------------------------

    fn_csv = (
        REPORTS_DIR
        / "false_negatives_v1.csv"
    )

    fn_rows = [
        row
        for row in object_rows
        if row["status"]
        in {
            "FN",
            "CLASS_ERROR",
        }
    ]

    with open(
        fn_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=object_fields,
        )

        writer.writeheader()
        writer.writerows(fn_rows)

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    recall_global = (
        total_tp / total_gt
        if total_gt > 0
        else 0
    )

    precision_global = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    summary_path = (
        REPORTS_DIR
        / "DETECTION_FAILURE_ANALYSIS_V1_SUMMARY.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - DETECTION FAILURE ANALYSIS V1\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(
            f"Dataset:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Model:\n{MODEL_PATH}\n\n"
        )

        f.write(
            f"Test images: {len(images):,}\n"
        )

        f.write(
            f"Ground Truth objects: {total_gt:,}\n"
        )

        f.write(
            f"Predictions: {total_pred:,}\n"
        )

        f.write(
            f"TP: {total_tp:,}\n"
        )

        f.write(
            f"FN: {total_fn:,}\n"
        )

        f.write(
            f"FP: {total_fp:,}\n"
        )

        f.write(
            f"Class errors: "
            f"{total_class_errors:,}\n\n"
        )

        f.write(
            f"Recall global: "
            f"{recall_global:.6f}\n"
        )

        f.write(
            f"Precision global: "
            f"{precision_global:.6f}\n\n"
        )

        f.write(
            "CLASS RESULTS\n"
        )
        f.write("-" * 72 + "\n")

        for cls in [
            "PERSON",
            "VEHICLE",
        ]:

            stats = class_statistics[cls]

            gt = stats["gt"]
            tp = stats["tp"]
            fn = stats["fn"]
            fp = stats["fp"]

            recall = (
                tp / gt
                if gt > 0
                else 0
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            f.write(
                f"{cls}\n"
            )

            f.write(
                f"  GT: {gt:,}\n"
            )

            f.write(
                f"  TP: {tp:,}\n"
            )

            f.write(
                f"  FN: {fn:,}\n"
            )

            f.write(
                f"  FP: {fp:,}\n"
            )

            f.write(
                f"  Recall: {recall:.6f}\n"
            )

            f.write(
                f"  Precision: {precision:.6f}\n\n"
            )

        f.write(
            "RECALL BY AREA\n"
        )
        f.write("-" * 72 + "\n")

        for cls in [
            "PERSON",
            "VEHICLE",
        ]:

            f.write(
                f"\n{cls}\n"
            )

            for area_name, _, _ in AREA_BINS:

                stats = area_statistics[
                    (cls, area_name)
                ]

                gt = stats["gt"]
                tp = stats["tp"]
                fn = stats["fn"]

                recall = (
                    tp / gt
                    if gt > 0
                    else 0
                )

                f.write(
                    f"  {area_name:>8}: "
                    f"GT={gt:6,} "
                    f"TP={tp:6,} "
                    f"FN={fn:6,} "
                    f"Recall={recall:.4f}\n"
                )

        f.write("\n")
        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "This script is diagnostic only.\n"
        )

        f.write(
            "The original and cleaned datasets were not modified.\n"
        )

    # ------------------------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("# RESULTADO DETECTION FAILURE ANALYSIS V1")
    print("=" * 72)

    print()
    print(
        f"Imágenes test_dev:      {len(images):,}"
    )

    print(
        f"Ground Truth:            {total_gt:,}"
    )

    print(
        f"Predicciones:            {total_pred:,}"
    )

    print(
        f"TP:                      {total_tp:,}"
    )

    print(
        f"FN:                      {total_fn:,}"
    )

    print(
        f"FP:                      {total_fp:,}"
    )

    print(
        f"Errores de clase:        {total_class_errors:,}"
    )

    print()
    print(
        f"Recall global:            {recall_global:.4f}"
    )

    print(
        f"Precision global:         {precision_global:.4f}"
    )

    print()
    print("PERSON")

    stats = class_statistics["PERSON"]

    person_recall = (
        stats["tp"] / stats["gt"]
        if stats["gt"] > 0
        else 0
    )

    print(
        f"GT:                       {stats['gt']:,}"
    )

    print(
        f"TP:                       {stats['tp']:,}"
    )

    print(
        f"FN:                       {stats['fn']:,}"
    )

    print(
        f"Recall:                   {person_recall:.4f}"
    )

    print()
    print("VEHICLE")

    stats = class_statistics["VEHICLE"]

    vehicle_recall = (
        stats["tp"] / stats["gt"]
        if stats["gt"] > 0
        else 0
    )

    print(
        f"GT:                       {stats['gt']:,}"
    )

    print(
        f"TP:                       {stats['tp']:,}"
    )

    print(
        f"FN:                       {stats['fn']:,}"
    )

    print(
        f"Recall:                   {vehicle_recall:.4f}"
    )

    print()
    print("Reports:")

    print(objects_csv)
    print(images_csv)
    print(area_csv)
    print(class_csv)
    print(fn_csv)
    print(summary_path)

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    print()
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    analyze()