from pathlib import Path
from collections import defaultdict
import csv
import math

from PIL import Image
from ultralytics import YOLO


# ============================================================
# SAR YOLO26 - PERSON RECALL BY SCALE ANALYSIS V1
# ============================================================

DATASET = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\training\runs\baseline_v1\weights\best.pt"
)

OUTPUT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\detection_failure_analysis"
    r"\person_recall_by_scale"
    r"\analyze_person_recall_by_scale_v1"
)

TEST_IMAGES = DATASET / "test_dev" / "images"
TEST_LABELS = DATASET / "test_dev" / "labels"

PERSON_CLASS = 0
VEHICLE_CLASS = 1

IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.25

SCALE_BINS = [
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


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_labels(label_path, width, height):
    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            objects.append({
                "class": cls,
                "box": [x1, y1, x2, y2],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            })

    return objects


def scale_bucket(area):
    for name, low, high in SCALE_BINS:
        if low <= area < high:
            return name

    return ">2048"


def match_person_gt(gt_objects, predictions):
    """
    Greedy one-to-one matching for PERSON using IoU >= threshold.
    Only predictions with class PERSON are considered.
    """

    person_gt = [
        obj for obj in gt_objects
        if obj["class"] == PERSON_CLASS
    ]

    person_pred = [
        pred for pred in predictions
        if pred["class"] == PERSON_CLASS
    ]

    matches = set()

    for gt_index, gt in enumerate(person_gt):

        best_iou = 0.0
        best_pred = None

        for pred_index, pred in enumerate(person_pred):

            if pred_index in matches:
                continue

            current_iou = iou(gt["box"], pred["box"])

            if current_iou > best_iou:
                best_iou = current_iou
                best_pred = pred_index

        if best_iou >= IOU_THRESHOLD and best_pred is not None:
            matches.add(best_pred)

            gt["_matched"] = True
        else:
            gt["_matched"] = False

    return person_gt


def main():

    print()
    print("=" * 72)
    print("# SAR YOLO26 - PERSON RECALL BY SCALE ANALYSIS V1")
    print("=" * 72)
    print()

    print("Dataset:")
    print(DATASET)
    print()

    print("Modelo:")
    print(MODEL)
    print()

    print("Test:")
    print(TEST_IMAGES)
    print()

    print("Output:")
    print(OUTPUT)
    print()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = OUTPUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    object_report = reports / "person_recall_by_scale_objects_v1.csv"
    scale_report = reports / "person_recall_by_scale_v1.csv"
    summary_report = reports / "PERSON_RECALL_BY_SCALE_V1_SUMMARY.txt"

    image_files = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    print(f"Imágenes encontradas: {len(image_files)}")
    print()

    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL))

    print("[OK] Modelo cargado.")
    print()

    statistics = {
        name: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
        for name, _, _ in SCALE_BINS
    }

    object_rows = []

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    for index, image_path in enumerate(image_files, start=1):

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        label_path = TEST_LABELS / f"{image_path.stem}.txt"

        gt_objects = load_labels(
            label_path,
            width,
            height
        )

        predictions_result = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        predictions = []

        if predictions_result.boxes is not None:

            boxes = predictions_result.boxes.xyxy.cpu().numpy()
            classes = predictions_result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):

                predictions.append({
                    "class": int(cls),
                    "box": box.tolist()
                })

        person_gt = match_person_gt(
            gt_objects,
            predictions
        )

        for obj_index, gt in enumerate(person_gt):

            area = gt["area"]
            bucket = scale_bucket(area)

            matched = bool(gt.get("_matched", False))

            statistics[bucket]["gt"] += 1

            if matched:
                statistics[bucket]["tp"] += 1
                total_person_tp += 1
            else:
                statistics[bucket]["fn"] += 1
                total_person_fn += 1

            total_person_gt += 1

            object_rows.append({
                "image": image_path.name,
                "image_path": str(image_path),
                "object_index": obj_index,
                "area_px2": round(area, 4),
                "scale": bucket,
                "matched": int(matched),
                "result": "TP" if matched else "FN",
            })

        if index % 100 == 0 or index == len(image_files):
            print(
                f"Analizadas: {index:,}/{len(image_files):,}"
            )

    print()
    print("=" * 72)
    print("# RESULTADO PERSON RECALL BY SCALE V1")
    print("=" * 72)
    print()

    print(
        f"PERSON GT:              {total_person_gt:,}"
    )
    print(
        f"PERSON TP:              {total_person_tp:,}"
    )
    print(
        f"PERSON FN:              {total_person_fn:,}"
    )

    global_recall = (
        total_person_tp / total_person_gt
        if total_person_gt
        else 0
    )

    print(
        f"PERSON Recall:          {global_recall:.4f}"
    )

    print()
    print("RECALL POR ESCALA")
    print()

    for name, _, _ in SCALE_BINS:

        data = statistics[name]

        gt = data["gt"]
        tp = data["tp"]
        fn = data["fn"]

        recall = tp / gt if gt else 0
        percentage = gt / total_person_gt if total_person_gt else 0

        print(
            f"{name:>10} px²"
            f"  GT={gt:7,}"
            f"  TP={tp:7,}"
            f"  FN={fn:7,}"
            f"  Recall={recall:.4f}"
            f"  ({percentage * 100:6.2f} %)"
        )

    print()

    # ------------------------------------------------------------
    # CSV OBJECTS
    # ------------------------------------------------------------

    with open(
        object_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "image_path",
                "object_index",
                "area_px2",
                "scale",
                "matched",
                "result",
            ]
        )

        writer.writeheader()
        writer.writerows(object_rows)

    # ------------------------------------------------------------
    # CSV SCALE
    # ------------------------------------------------------------

    with open(
        scale_report,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "scale",
            "gt",
            "tp",
            "fn",
            "recall",
            "percentage_of_person_gt",
        ])

        for name, _, _ in SCALE_BINS:

            data = statistics[name]

            gt = data["gt"]
            tp = data["tp"]
            fn = data["fn"]

            recall = tp / gt if gt else 0
            percentage = (
                gt / total_person_gt
                if total_person_gt
                else 0
            )

            writer.writerow([
                name,
                gt,
                tp,
                fn,
                f"{recall:.6f}",
                f"{percentage:.6f}",
            ])

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    with open(
        summary_report,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON RECALL BY SCALE ANALYSIS V1\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(f"Dataset:\n{DATASET}\n\n")
        f.write(f"Modelo:\n{MODEL}\n\n")

        f.write(f"Test images: {len(image_files):,}\n\n")

        f.write(
            f"PERSON GT:     {total_person_gt:,}\n"
        )
        f.write(
            f"PERSON TP:     {total_person_tp:,}\n"
        )
        f.write(
            f"PERSON FN:     {total_person_fn:,}\n"
        )
        f.write(
            f"PERSON Recall: {global_recall:.6f}\n\n"
        )

        f.write("RECALL POR ESCALA\n")
        f.write("-" * 72 + "\n")

        for name, _, _ in SCALE_BINS:

            data = statistics[name]

            gt = data["gt"]
            tp = data["tp"]
            fn = data["fn"]

            recall = tp / gt if gt else 0
            percentage = (
                gt / total_person_gt
                if total_person_gt
                else 0
            )

            f.write(
                f"{name:>10} px² | "
                f"GT={gt:7,} | "
                f"TP={tp:7,} | "
                f"FN={fn:7,} | "
                f"Recall={recall:.6f} | "
                f"{percentage * 100:.2f}%\n"
            )

        f.write("\n")
        f.write(
            "IMPORTANTE: el dataset NO ha sido modificado.\n"
        )

    print(f"[OK] {object_report}")
    print(f"[OK] {scale_report}")
    print(f"[OK] {summary_report}")
    print()
    print(
        "# IMPORTANTE: el dataset NO ha sido modificado."
    )
    print()


if __name__ == "__main__":
    main()