from pathlib import Path
from collections import defaultdict
import csv
import statistics

from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned"
    r"\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\training\runs\baseline_v1\weights\best.pt"
)

OUTPUT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\detection_failure_analysis"
    r"\person_recall_by_density"
    r"\analyze_person_recall_by_density_v1"
)

REPORTS = OUTPUT / "reports"

SPLITS = [
    "train",
    "val",
    "test_dev",
]

PERSON_CLASS = 0
VEHICLE_CLASS = 1

CONF = 0.001
IOU = 0.50

# Umbrales de densidad total de objetos por imagen
DENSITY_BINS = [
    ("<25", 0, 24),
    ("25-49", 25, 49),
    ("50-99", 50, 99),
    ("100-199", 100, 199),
    ("200-299", 200, 299),
    ("300-499", 300, 499),
    (">=500", 500, float("inf")),
]


# ============================================================
# UTILIDADES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_images(folder):
    if not folder.exists():
        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def find_label(image_path):
    """
    Busca el label YOLO correspondiente.
    """
    label_path = (
        image_path.parent.parent
        / "labels"
        / image_path.relative_to(image_path.parent.parent / "images")
    ).with_suffix(".txt")

    if label_path.exists():
        return label_path

    # Fallback robusto para estructuras con subdirectorios.
    parts = list(image_path.parts)

    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts[:]
        label_parts[idx] = "labels"
        label_path = Path(*label_parts).with_suffix(".txt")

        if label_path.exists():
            return label_path

    return None


def read_yolo_labels(label_path):
    """
    Devuelve:
        [
            {
                "class_id": int,
                "x": float,
                "y": float,
                "w": float,
                "h": float,
                "area_norm": float
            }
        ]
    """
    objects = []

    if label_path is None or not label_path.exists():
        return objects

    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                if w <= 0 or h <= 0:
                    continue

                objects.append(
                    {
                        "class_id": class_id,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "area_norm": w * h,
                    }
                )

    except Exception:
        return []

    return objects


def bbox_iou(box1, box2):
    """
    Boxes:
        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )

    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def normalized_to_pixel_bbox(obj, width, height):
    cx = obj["x"] * width
    cy = obj["y"] * height

    bw = obj["w"] * width
    bh = obj["h"] * height

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    return [x1, y1, x2, y2]


def get_density_bin(total_objects):
    for name, low, high in DENSITY_BINS:
        if low <= total_objects <= high:
            return name

    return "UNKNOWN"


def safe_recall(tp, gt):
    if gt == 0:
        return 0.0

    return tp / gt


def safe_percent(value, total):
    if total == 0:
        return 0.0

    return 100.0 * value / total


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("# SAR YOLO26 - PERSON RECALL BY DENSITY ANALYSIS V1")
    print("=" * 72)
    print()

    print("Dataset:")
    print(DATASET)
    print()

    print("Modelo:")
    print(MODEL_PATH)
    print()

    test_path = DATASET / "test_dev" / "images"

    print("Test:")
    print(test_path)
    print()

    print("Output:")
    print(OUTPUT)
    print()

    REPORTS.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------------

    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL_PATH))

    print("[OK] Modelo cargado.")
    print()

    # --------------------------------------------------------
    # Recoger imágenes de todos los splits
    # --------------------------------------------------------

    all_images = []

    for split in SPLITS:

        image_dir = DATASET / split / "images"

        images = find_images(image_dir)

        print(f"## Analizando: {split}")
        print()
        print(f"Imágenes encontradas: {len(images)}")

        for image_path in images:
            all_images.append((split, image_path))

        print()

    print(f"Total imágenes encontradas: {len(all_images)}")
    print()

    # --------------------------------------------------------
    # Solo test_dev para evaluación de recall
    # --------------------------------------------------------

    test_images = [
        (split, path)
        for split, path in all_images
        if split == "test_dev"
    ]

    print(
        f"Imágenes test_dev utilizadas para evaluación: "
        f"{len(test_images)}"
    )
    print()

    # --------------------------------------------------------
    # Estadísticas
    # --------------------------------------------------------

    density_stats = {}

    for name, _, _ in DENSITY_BINS:
        density_stats[name] = {
            "images": 0,
            "person_gt": 0,
            "person_tp": 0,
            "person_fn": 0,
            "total_objects": 0,
            "person_area_values": [],
            "persons_per_image": [],
        }

    object_rows = []
    image_rows = []

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    total_images = len(test_images)

    # --------------------------------------------------------
    # Procesamiento
    # --------------------------------------------------------

    for index, (split, image_path) in enumerate(test_images, start=1):

        label_path = find_label(image_path)

        gt_objects = read_yolo_labels(label_path)

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        total_objects = len(gt_objects)

        person_gt = [
            obj
            for obj in gt_objects
            if obj["class_id"] == PERSON_CLASS
        ]

        density_bin = get_density_bin(total_objects)

        stats = density_stats[density_bin]

        stats["images"] += 1
        stats["person_gt"] += len(person_gt)
        stats["total_objects"] += total_objects
        stats["persons_per_image"].append(len(person_gt))

        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------

        results = model.predict(
            source=str(image_path),
            conf=CONF,
            iou=IOU,
            verbose=False,
            classes=[PERSON_CLASS],
        )

        predictions = []

        if results:

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(
                    boxes,
                    classes,
                    confs,
                ):

                    if int(cls) != PERSON_CLASS:
                        continue

                    predictions.append(
                        {
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2]),
                                float(box[3]),
                            ],
                            "confidence": float(conf),
                        }
                    )

        # ----------------------------------------------------
        # Matching GT -> prediction
        # ----------------------------------------------------

        matched_predictions = set()

        image_tp = 0
        image_fn = 0

        for gt_index, gt in enumerate(person_gt):

            gt_bbox = normalized_to_pixel_bbox(
                gt,
                width,
                height,
            )

            best_iou = 0.0
            best_prediction = None

            for pred_index, pred in enumerate(predictions):

                if pred_index in matched_predictions:
                    continue

                iou = bbox_iou(
                    gt_bbox,
                    pred["bbox"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_prediction = pred_index

            matched = (
                best_prediction is not None
                and best_iou >= IOU
            )

            area_px2 = gt["area_norm"] * width * height

            if matched:

                matched_predictions.add(best_prediction)
                image_tp += 1

                status = "TP"

            else:

                image_fn += 1
                status = "FN"

            object_rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "density_bin": density_bin,
                    "total_objects": total_objects,
                    "person_index": gt_index,
                    "person_area_px2": area_px2,
                    "person_width_px": gt["w"] * width,
                    "person_height_px": gt["h"] * height,
                    "status": status,
                    "best_iou": best_iou,
                }
            )

            stats["person_area_values"].append(area_px2)

        image_rows.append(
            {
                "split": split,
                "image": str(image_path),
                "total_objects": total_objects,
                "density_bin": density_bin,
                "person_gt": len(person_gt),
                "person_tp": image_tp,
                "person_fn": image_fn,
                "person_recall": safe_recall(
                    image_tp,
                    len(person_gt),
                ),
            }
        )

        stats["person_tp"] += image_tp
        stats["person_fn"] += image_fn

        total_person_gt += len(person_gt)
        total_person_tp += image_tp
        total_person_fn += image_fn

        if index % 100 == 0 or index == total_images:
            print(
                f"Analizadas: {index:,}/{total_images:,}"
            )

    # --------------------------------------------------------
    # CSV 1 - Objects
    # --------------------------------------------------------

    objects_csv = (
        REPORTS
        / "person_recall_by_density_objects_v1.csv"
    )

    with objects_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image",
                "density_bin",
                "total_objects",
                "person_index",
                "person_area_px2",
                "person_width_px",
                "person_height_px",
                "status",
                "best_iou",
            ],
        )

        writer.writeheader()
        writer.writerows(object_rows)

    print()
    print(f"[OK] {objects_csv}")

    # --------------------------------------------------------
    # CSV 2 - Density summary
    # --------------------------------------------------------

    density_csv = (
        REPORTS
        / "person_recall_by_density_v1.csv"
    )

    with density_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "density_bin",
            "images",
            "image_percent",
            "person_gt",
            "person_tp",
            "person_fn",
            "recall",
            "person_percent",
            "mean_objects_per_image",
            "median_persons_per_image",
            "mean_person_area_px2",
            "median_person_area_px2",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for name, _, _ in DENSITY_BINS:

            stats = density_stats[name]

            mean_objects = (
                stats["total_objects"]
                / stats["images"]
                if stats["images"] > 0
                else 0
            )

            person_area_values = stats[
                "person_area_values"
            ]

            mean_area = (
                statistics.mean(person_area_values)
                if person_area_values
                else 0
            )

            median_area = (
                statistics.median(person_area_values)
                if person_area_values
                else 0
            )

            median_persons = (
                statistics.median(
                    stats["persons_per_image"]
                )
                if stats["persons_per_image"]
                else 0
            )

            writer.writerow(
                {
                    "density_bin": name,
                    "images": stats["images"],
                    "image_percent": safe_percent(
                        stats["images"],
                        total_images,
                    ),
                    "person_gt": stats["person_gt"],
                    "person_tp": stats["person_tp"],
                    "person_fn": stats["person_fn"],
                    "recall": safe_recall(
                        stats["person_tp"],
                        stats["person_gt"],
                    ),
                    "person_percent": safe_percent(
                        stats["person_gt"],
                        total_person_gt,
                    ),
                    "mean_objects_per_image": mean_objects,
                    "median_persons_per_image": median_persons,
                    "mean_person_area_px2": mean_area,
                    "median_person_area_px2": median_area,
                }
            )

    print(f"[OK] {density_csv}")

    # --------------------------------------------------------
    # CSV 3 - Image statistics
    # --------------------------------------------------------

    image_stats_csv = (
        REPORTS
        / "density_statistics_v1.csv"
    )

    with image_stats_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image",
                "total_objects",
                "density_bin",
                "person_gt",
                "person_tp",
                "person_fn",
                "person_recall",
            ],
        )

        writer.writeheader()
        writer.writerows(image_rows)

    print(f"[OK] {image_stats_csv}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = (
        REPORTS
        / "PERSON_RECALL_BY_DENSITY_V1_SUMMARY.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON RECALL BY DENSITY ANALYSIS V1\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(f"Dataset:\n{DATASET}\n\n")
        f.write(f"Modelo:\n{MODEL_PATH}\n\n")

        f.write("RESULTADO GLOBAL\n")
        f.write("-" * 72 + "\n")

        f.write(f"Imágenes:       {total_images:,}\n")
        f.write(f"PERSON GT:      {total_person_gt:,}\n")
        f.write(f"PERSON TP:      {total_person_tp:,}\n")
        f.write(f"PERSON FN:      {total_person_fn:,}\n")

        f.write(
            f"PERSON Recall:  "
            f"{safe_recall(total_person_tp, total_person_gt):.4f}\n\n"
        )

        f.write("RECALL POR DENSIDAD\n")
        f.write("-" * 72 + "\n\n")

        for name, _, _ in DENSITY_BINS:

            stats = density_stats[name]

            recall = safe_recall(
                stats["person_tp"],
                stats["person_gt"],
            )

            percentage = safe_percent(
                stats["person_gt"],
                total_person_gt,
            )

            f.write(
                f"{name:>8} "
                f"Images={stats['images']:6,} "
                f"GT={stats['person_gt']:7,} "
                f"TP={stats['person_tp']:7,} "
                f"FN={stats['person_fn']:7,} "
                f"Recall={recall:.4f} "
                f"({percentage:6.2f} %)\n"
            )

        f.write("\n")
        f.write("NOTA\n")
        f.write("-" * 72 + "\n")
        f.write(
            "Este análisis utiliza exclusivamente test_dev.\n"
        )
        f.write(
            "La densidad corresponde al número total de objetos "
            "anotados en la imagen.\n"
        )
        f.write(
            "El matching utiliza IoU >= 0.50.\n"
        )
        f.write(
            "El dataset original NO ha sido modificado.\n"
        )

    print(f"[OK] {summary_path}")

    # --------------------------------------------------------
    # Resultado consola
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("# RESULTADO PERSON RECALL BY DENSITY V1")
    print("=" * 72)
    print()

    print(f"Imágenes:              {total_images:,}")
    print(f"PERSON GT:             {total_person_gt:,}")
    print(f"PERSON TP:             {total_person_tp:,}")
    print(f"PERSON FN:             {total_person_fn:,}")
    print(
        f"PERSON Recall:         "
        f"{safe_recall(total_person_tp, total_person_gt):.4f}"
    )

    print()
    print("RECALL POR DENSIDAD")
    print()

    for name, _, _ in DENSITY_BINS:

        stats = density_stats[name]

        recall = safe_recall(
            stats["person_tp"],
            stats["person_gt"],
        )

        percentage = safe_percent(
            stats["person_gt"],
            total_person_gt,
        )

        print(
            f"{name:>8} "
            f"Images={stats['images']:6,} "
            f"GT={stats['person_gt']:7,} "
            f"TP={stats['person_tp']:7,} "
            f"FN={stats['person_fn']:7,} "
            f"Recall={recall:.4f} "
            f"({percentage:6.2f} %)"
        )

    print()
    print(f"[OK] {objects_csv}")
    print(f"[OK] {density_csv}")
    print(f"[OK] {image_stats_csv}")
    print(f"[OK] {summary_path}")
    print()
    print("# IMPORTANTE: el dataset NO ha sido modificado.")
    print("=" * 72)


if __name__ == "__main__":
    main()