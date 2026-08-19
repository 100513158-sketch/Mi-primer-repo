from pathlib import Path
from collections import defaultdict
import csv
import math

from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\training\runs\baseline_v1\weights\best.pt"
)

BASE_OUTPUT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\detection_failure_analysis"
    r"\person_recall_by_image_size"
    r"\analyze_person_recall_by_image_size_v1"
)

REPORTS = BASE_OUTPUT / "reports"

TEST_SPLIT = "test_dev"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

PERSON_CLASS = 0

# Umbral IoU utilizado para considerar una predicción
# como True Positive.
IOU_THRESHOLD = 0.50

# Confianza de inferencia.
CONF_THRESHOLD = 0.25

# Tamaño de inferencia.
IMG_SIZE = 640

# ============================================================
# FUNCIONES
# ============================================================


def calculate_iou(box1, box2):
    """
    Calcula IoU entre dos cajas:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    if intersection <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * \
            max(0.0, box1[3] - box1[1])

    area2 = max(0.0, box2[2] - box2[0]) * \
            max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_person_ground_truth(label_path, width, height):
    """
    Lee únicamente PERSON (class 0) de un label YOLO.

    Devuelve:
        [
            {
                "box": [x1,y1,x2,y2],
                "area": area
            },
            ...
        ]
    """

    persons = []

    if not label_path.exists():
        return persons

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue

            if cls != PERSON_CLASS:
                continue

            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height

            x1 = max(0.0, min(width, x1))
            y1 = max(0.0, min(height, y1))
            x2 = max(0.0, min(width, x2))
            y2 = max(0.0, min(height, y2))

            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)

            area = box_width * box_height

            if area <= 0:
                continue

            persons.append(
                {
                    "box": [x1, y1, x2, y2],
                    "area": area,
                }
            )

    return persons


def match_person_predictions(gt_boxes, predictions):
    """
    Matching greedy GT <-> prediction usando IoU.

    Cada GT solo puede tener una predicción.
    Cada predicción solo puede corresponder a un GT.
    """

    if not gt_boxes:
        return 0, 0

    if not predictions:
        return 0, len(gt_boxes)

    candidates = []

    for gt_idx, gt in enumerate(gt_boxes):

        for pred_idx, pred in enumerate(predictions):

            iou = calculate_iou(
                gt["box"],
                pred
            )

            if iou >= IOU_THRESHOLD:
                candidates.append(
                    (
                        iou,
                        gt_idx,
                        pred_idx,
                    )
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matched_gt = set()
    matched_pred = set()

    tp = 0

    for iou, gt_idx, pred_idx in candidates:

        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

        tp += 1

    fn = len(gt_boxes) - tp

    return tp, fn


def get_image_size_group(width, height):
    """
    Clasificación basada en el lado mayor de la imagen.

    Esto evita mezclar directamente imágenes con distintas
    relaciones de aspecto.

    <640
    640-999
    1000-1499
    1500-1999
    >=2000
    """

    max_side = max(width, height)

    if max_side < 640:
        return "<640"

    if max_side < 1000:
        return "640-999"

    if max_side < 1500:
        return "1000-1499"

    if max_side < 2000:
        return "1500-1999"

    return ">=2000"


def get_resolution_group(width, height):
    """
    Clasificación exacta de resolución.
    """

    return f"{width}x{height}"


def safe_recall(tp, gt):

    if gt == 0:
        return 0.0

    return tp / gt


def write_csv(path, rows, fieldnames):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 72)
    print("# SAR YOLO26 - PERSON RECALL BY IMAGE SIZE ANALYSIS V1")
    print("=" * 72)
    print()

    print("Dataset:")
    print(DATASET)
    print()

    print("Modelo:")
    print(MODEL_PATH)
    print()

    test_images_dir = DATASET / TEST_SPLIT / "images"

    print("Test:")
    print(test_images_dir)
    print()

    print("Output:")
    print(BASE_OUTPUT)
    print()

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Dataset no encontrado:\n{DATASET}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Modelo no encontrado:\n{MODEL_PATH}"
        )

    if not test_images_dir.exists():

        raise FileNotFoundError(
            f"Directorio test_dev no encontrado:\n"
            f"{test_images_dir}"
        )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        [
            p
            for p in test_images_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    print(
        f"Imágenes encontradas: {len(image_paths):,}"
    )
    print()

    print("Cargando modelo YOLO26s...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("[OK] Modelo cargado.")
    print()

    # ========================================================
    # ACUMULADORES
    # ========================================================

    image_rows = []

    size_groups = defaultdict(
        lambda: {
            "images": 0,
            "gt": 0,
            "tp": 0,
            "fn": 0,
            "persons_small_16": 0,
            "persons_small_32": 0,
            "persons_small_64": 0,
        }
    )

    resolution_groups = defaultdict(
        lambda: {
            "images": 0,
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    total_gt = 0
    total_tp = 0
    total_fn = 0

    # ========================================================
    # ANALISIS
    # ========================================================

    for idx, image_path in enumerate(
        image_paths,
        start=1
    ):

        label_path = (
            DATASET
            / TEST_SPLIT
            / "labels"
            / f"{image_path.stem}.txt"
        )

        try:

            with Image.open(image_path) as img:

                width, height = img.size

        except Exception as exc:

            print(
                f"[WARNING] No se pudo leer: "
                f"{image_path}"
            )

            continue

        gt_persons = load_person_ground_truth(
            label_path,
            width,
            height
        )

        # ----------------------------------------------------
        # PREDICCIÓN
        # ----------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        person_predictions = []

        if results:

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes

                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for box, cls in zip(
                    xyxy,
                    classes
                ):

                    if int(cls) != PERSON_CLASS:
                        continue

                    person_predictions.append(
                        [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ]
                    )

        tp, fn = match_person_predictions(
            gt_persons,
            person_predictions
        )

        gt_count = len(gt_persons)

        recall = safe_recall(
            tp,
            gt_count
        )

        size_group = get_image_size_group(
            width,
            height
        )

        resolution_group = get_resolution_group(
            width,
            height
        )

        tiny16 = sum(
            1
            for obj in gt_persons
            if obj["area"] < 16
        )

        tiny32 = sum(
            1
            for obj in gt_persons
            if obj["area"] < 32
        )

        tiny64 = sum(
            1
            for obj in gt_persons
            if obj["area"] < 64
        )

        # ----------------------------------------------------
        # POR TAMAÑO DE IMAGEN
        # ----------------------------------------------------

        sg = size_groups[size_group]

        sg["images"] += 1
        sg["gt"] += gt_count
        sg["tp"] += tp
        sg["fn"] += fn
        sg["persons_small_16"] += tiny16
        sg["persons_small_32"] += tiny32
        sg["persons_small_64"] += tiny64

        # ----------------------------------------------------
        # POR RESOLUCIÓN EXACTA
        # ----------------------------------------------------

        rg = resolution_groups[
            resolution_group
        ]

        rg["images"] += 1
        rg["gt"] += gt_count
        rg["tp"] += tp
        rg["fn"] += fn

        # ----------------------------------------------------
        # TOTALES
        # ----------------------------------------------------

        total_gt += gt_count
        total_tp += tp
        total_fn += fn

        image_rows.append(
            {
                "image": str(image_path),
                "width": width,
                "height": height,
                "pixels": width * height,
                "max_side": max(width, height),
                "image_size_group": size_group,
                "resolution": resolution_group,
                "person_gt": gt_count,
                "person_tp": tp,
                "person_fn": fn,
                "person_recall": round(
                    recall,
                    6
                ),
                "person_tiny16": tiny16,
                "person_tiny32": tiny32,
                "person_tiny64": tiny64,
                "person_predictions": len(
                    person_predictions
                ),
            }
        )

        if (
            idx % 100 == 0
            or idx == len(image_paths)
        ):

            print(
                f"Analizadas: "
                f"{idx:,}/{len(image_paths):,}"
            )

    # ========================================================
    # RESUMEN POR TAMAÑO DE IMAGEN
    # ========================================================

    size_order = [
        "<640",
        "640-999",
        "1000-1499",
        "1500-1999",
        ">=2000",
    ]

    size_rows = []

    for group in size_order:

        data = size_groups[group]

        gt = data["gt"]
        tp = data["tp"]
        fn = data["fn"]

        size_rows.append(
            {
                "image_size_group": group,
                "images": data["images"],
                "person_gt": gt,
                "person_tp": tp,
                "person_fn": fn,
                "person_recall": round(
                    safe_recall(tp, gt),
                    6
                ),
                "tiny16": data[
                    "persons_small_16"
                ],
                "tiny32": data[
                    "persons_small_32"
                ],
                "tiny64": data[
                    "persons_small_64"
                ],
                "tiny16_pct": round(
                    (
                        data["persons_small_16"]
                        / gt * 100
                    )
                    if gt else 0,
                    4
                ),
                "tiny32_pct": round(
                    (
                        data["persons_small_32"]
                        / gt * 100
                    )
                    if gt else 0,
                    4
                ),
                "tiny64_pct": round(
                    (
                        data["persons_small_64"]
                        / gt * 100
                    )
                    if gt else 0,
                    4
                ),
            }
        )

    # ========================================================
    # RESUMEN POR RESOLUCIÓN
    # ========================================================

    resolution_rows = []

    for resolution in sorted(
        resolution_groups.keys()
    ):

        data = resolution_groups[
            resolution
        ]

        gt = data["gt"]
        tp = data["tp"]
        fn = data["fn"]

        resolution_rows.append(
            {
                "resolution": resolution,
                "images": data["images"],
                "person_gt": gt,
                "person_tp": tp,
                "person_fn": fn,
                "person_recall": round(
                    safe_recall(tp, gt),
                    6
                ),
            }
        )

    # ========================================================
    # CSV
    # ========================================================

    write_csv(
        REPORTS
        / "person_recall_by_image_size_objects_v1.csv",
        image_rows,
        [
            "image",
            "width",
            "height",
            "pixels",
            "max_side",
            "image_size_group",
            "resolution",
            "person_gt",
            "person_tp",
            "person_fn",
            "person_recall",
            "person_tiny16",
            "person_tiny32",
            "person_tiny64",
            "person_predictions",
        ]
    )

    print()
    print(
        "[OK]",
        REPORTS
        / "person_recall_by_image_size_objects_v1.csv"
    )

    write_csv(
        REPORTS
        / "person_recall_by_image_size_v1.csv",
        size_rows,
        [
            "image_size_group",
            "images",
            "person_gt",
            "person_tp",
            "person_fn",
            "person_recall",
            "tiny16",
            "tiny32",
            "tiny64",
            "tiny16_pct",
            "tiny32_pct",
            "tiny64_pct",
        ]
    )

    print(
        "[OK]",
        REPORTS
        / "person_recall_by_image_size_v1.csv"
    )

    write_csv(
        REPORTS
        / "image_size_statistics_v1.csv",
        resolution_rows,
        [
            "resolution",
            "images",
            "person_gt",
            "person_tp",
            "person_fn",
            "person_recall",
        ]
    )

    print(
        "[OK]",
        REPORTS
        / "image_size_statistics_v1.csv"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    global_recall = safe_recall(
        total_tp,
        total_gt
    )

    summary_path = (
        REPORTS
        / "PERSON_RECALL_BY_IMAGE_SIZE_V1_SUMMARY.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON RECALL BY IMAGE SIZE "
            "ANALYSIS V1\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            f"Dataset: {DATASET}\n"
        )

        f.write(
            f"Modelo: {MODEL_PATH}\n"
        )

        f.write(
            f"Test: {test_images_dir}\n\n"
        )

        f.write(
            f"Imágenes: {len(image_rows):,}\n"
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
            "RECALL POR TAMAÑO DE IMAGEN\n\n"
        )

        f.write(
            f"{'Grupo':<15}"
            f"{'Images':>10}"
            f"{'GT':>10}"
            f"{'TP':>10}"
            f"{'FN':>10}"
            f"{'Recall':>12}\n"
        )

        f.write("-" * 67 + "\n")

        for row in size_rows:

            f.write(
                f"{row['image_size_group']:<15}"
                f"{row['images']:>10,}"
                f"{row['person_gt']:>10,}"
                f"{row['person_tp']:>10,}"
                f"{row['person_fn']:>10,}"
                f"{row['person_recall']:>12.4f}\n"
            )

        f.write("\n")

        f.write(
            "RESOLUCIONES ENCONTRADAS\n\n"
        )

        for row in resolution_rows:

            f.write(
                f"{row['resolution']}: "
                f"images={row['images']:,}; "
                f"GT={row['person_gt']:,}; "
                f"TP={row['person_tp']:,}; "
                f"FN={row['person_fn']:,}; "
                f"recall={row['person_recall']:.4f}\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANTE: este script SOLO diagnostica.\n"
            "El dataset NO ha sido modificado.\n"
        )

    print(
        "[OK]",
        summary_path
    )

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print()
    print("=" * 72)
    print("# RESULTADO PERSON RECALL BY IMAGE SIZE V1")
    print("=" * 72)
    print()

    print(
        f"Imágenes:              "
        f"{len(image_rows):,}"
    )

    print(
        f"PERSON GT:              "
        f"{total_gt:,}"
    )

    print(
        f"PERSON TP:              "
        f"{total_tp:,}"
    )

    print(
        f"PERSON FN:              "
        f"{total_fn:,}"
    )

    print(
        f"PERSON Recall:          "
        f"{global_recall:.4f}"
    )

    print()
    print("RECALL POR TAMAÑO DE IMAGEN")
    print()

    for row in size_rows:

        print(
            f"{row['image_size_group']:>10} "
            f"Images={row['images']:>5,} "
            f"GT={row['person_gt']:>7,} "
            f"TP={row['person_tp']:>7,} "
            f"FN={row['person_fn']:>7,} "
            f"Recall={row['person_recall']:.4f}"
        )

    print()
    print(
        "# IMPORTANTE: el dataset NO ha sido modificado."
    )
    print()


if __name__ == "__main__":
    main()