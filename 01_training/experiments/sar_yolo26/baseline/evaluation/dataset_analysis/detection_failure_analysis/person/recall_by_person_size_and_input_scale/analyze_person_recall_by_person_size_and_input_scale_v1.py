from pathlib import Path
from collections import defaultdict
import csv
import math

from ultralytics import YOLO


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASELINE_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = (
    BASELINE_ROOT
    / "training"
    / "runs"
    / "baseline_v1"
    / "weights"
    / "best.pt"
)

TEST_IMAGES_DIR = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test_dev" / "labels"

OUTPUT_ROOT = (
    BASELINE_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "recall_by_person_size_and_input_scale"
    / "analyze_person_recall_by_person_size_and_input_scale_v1"
)

REPORTS_DIR = OUTPUT_ROOT / "reports"

OBJECTS_CSV = (
    REPORTS_DIR
    / "person_recall_by_person_size_and_input_scale_objects_v1.csv"
)

SUMMARY_CSV = (
    REPORTS_DIR
    / "person_recall_by_person_size_and_input_scale_v1.csv"
)

STATISTICS_CSV = (
    REPORTS_DIR
    / "person_size_input_scale_statistics_v1.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "PERSON_RECALL_BY_PERSON_SIZE_AND_INPUT_SCALE_V1_SUMMARY.txt"
)


# ============================================================================
# PARÁMETROS DEL ANÁLISIS
# ============================================================================

PERSON_CLASS_ID = 0

IOU_THRESHOLD = 0.50

INPUT_SCALES = [
    640,
    960,
    1280,
    1536,
]

# Categorías de área de la persona en píxeles².
PERSON_SIZE_BINS = [
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
# UTILIDADES
# ============================================================================

def ensure_directories():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def list_images():
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    images = [
        p
        for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    images.sort()

    return images


def load_gt_labels(label_path):
    """
    Lee etiquetas YOLO:

        class x_center y_center width height

    Devuelve únicamente PERSON class=0.
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
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                continue

            if class_id != PERSON_CLASS_ID:
                continue

            persons.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )

    return persons


def yolo_to_xyxy(box, image_width, image_height):
    x_center = box["x_center"] * image_width
    y_center = box["y_center"] * image_height

    width = box["width"] * image_width
    height = box["height"] * image_height

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0

    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def xywhn_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):
    x_center *= image_width
    y_center *= image_height

    width *= image_width
    height *= image_height

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0

    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def box_area(box):
    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    if intersection <= 0:
        return 0.0

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def classify_person_size(area):
    """
    Clasifica usando exactamente el área GT original en píxeles².
    """

    for label, minimum, maximum in PERSON_SIZE_BINS:

        if minimum <= area < maximum:
            return label

    return ">2048"


def density_from_gt_count(count):
    if count < 25:
        return "<25"

    if count < 50:
        return "25-49"

    if count < 100:
        return "50-99"

    if count < 200:
        return "100-199"

    if count < 300:
        return "200-299"

    return "300-499"


# ============================================================================
# MATCHING
# ============================================================================

def match_person_predictions(gt_boxes, pred_boxes):
    """
    Matching greedy por IoU.

    Cada GT puede tener como máximo un TP.
    Cada predicción puede utilizarse una sola vez.

    Se utiliza IoU >= 0.50.
    """

    if not gt_boxes:
        return []

    if not pred_boxes:
        return [None] * len(gt_boxes)

    candidates = []

    for gt_index, gt_box in enumerate(gt_boxes):

        for pred_index, pred_box in enumerate(pred_boxes):

            iou = calculate_iou(gt_box, pred_box)

            if iou >= IOU_THRESHOLD:
                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    # Mejor IoU primero.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()

    matches = [None] * len(gt_boxes)

    for iou, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_pred:
            continue

        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

        matches[gt_index] = {
            "pred_index": pred_index,
            "iou": iou,
        }

    return matches


# ============================================================================
# ANÁLISIS DE UNA IMAGEN
# ============================================================================

def analyze_image(
    model,
    image_path,
    input_scale,
):
    """
    Analiza una imagen para un input_scale.

    IMPORTANTE:
    - GT se obtiene de las etiquetas originales.
    - El tamaño de la persona se calcula usando el GT original.
    - TP se determina mediante IoU >= 0.50.
    """

    label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"

    gt_persons = load_gt_labels(label_path)

    # Leer dimensiones originales mediante resultado YOLO.
    results = model.predict(
        source=str(image_path),
        imgsz=input_scale,
        conf=0.001,
        iou=0.7,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    original_height, original_width = result.orig_shape

    # ----------------------------------------------------------------------
    # GT boxes
    # ----------------------------------------------------------------------

    gt_boxes = []

    for gt in gt_persons:

        xyxy = yolo_to_xyxy(
            gt,
            original_width,
            original_height,
        )

        gt_boxes.append(xyxy)

    # ----------------------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------------------

    pred_boxes = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()

        classes = result.boxes.cls.cpu().numpy()

        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_id, confidence in zip(
            boxes_xyxy,
            classes,
            confidences,
        ):

            if int(cls_id) != PERSON_CLASS_ID:
                continue

            pred_boxes.append(
                {
                    "xyxy": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "confidence": float(confidence),
                }
            )

    prediction_xyxy = [
        p["xyxy"]
        for p in pred_boxes
    ]

    # ----------------------------------------------------------------------
    # Matching
    # ----------------------------------------------------------------------

    matches = match_person_predictions(
        gt_boxes,
        prediction_xyxy,
    )

    # ----------------------------------------------------------------------
    # Construir registros por PERSON GT
    # ----------------------------------------------------------------------

    image_records = []

    total_persons_in_image = len(gt_boxes)

    density = density_from_gt_count(
        total_persons_in_image
    )

    for index, gt_box in enumerate(gt_boxes):

        area = box_area(gt_box)

        size_label = classify_person_size(area)

        match = matches[index]

        if match is None:

            tp = 0
            iou = 0.0

        else:

            tp = 1
            iou = match["iou"]

        fn = 1 - tp

        image_records.append(
            {
                "image": image_path.name,
                "input_scale": input_scale,
                "gt_index": index,
                "person_size": size_label,
                "area": area,
                "density": density,
                "gt": 1,
                "tp": tp,
                "fn": fn,
                "iou": iou,
            }
        )

    return image_records


# ============================================================================
# CSV OBJECTS
# ============================================================================

def write_objects_csv(records):

    fields = [
        "image",
        "input_scale",
        "gt_index",
        "person_size",
        "area",
        "density",
        "gt",
        "tp",
        "fn",
        "iou",
    ]

    with OBJECTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in records:

            writer.writerow(
                {
                    "image": row["image"],
                    "input_scale": row["input_scale"],
                    "gt_index": row["gt_index"],
                    "person_size": row["person_size"],
                    "area": round(row["area"], 4),
                    "density": row["density"],
                    "gt": row["gt"],
                    "tp": row["tp"],
                    "fn": row["fn"],
                    "iou": round(row["iou"], 6),
                }
            )


# ============================================================================
# RESUMEN POR INPUT SCALE
# ============================================================================

def get_scale_summary(records):

    grouped = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    for row in records:

        scale = row["input_scale"]

        grouped[scale]["gt"] += row["gt"]
        grouped[scale]["tp"] += row["tp"]
        grouped[scale]["fn"] += row["fn"]

    rows = []

    for scale in INPUT_SCALES:

        values = grouped[scale]

        gt = values["gt"]
        tp = values["tp"]
        fn = values["fn"]

        recall = tp / gt if gt > 0 else 0.0

        rows.append(
            {
                "input_scale": scale,
                "person_gt": gt,
                "person_tp": tp,
                "person_fn": fn,
                "person_recall": recall,
            }
        )

    return rows


# ============================================================================
# RESUMEN POR TAMAÑO + INPUT SCALE
# ============================================================================

def get_size_scale_summary(records):

    grouped = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    for row in records:

        key = (
            row["person_size"],
            row["input_scale"],
        )

        grouped[key]["gt"] += row["gt"]
        grouped[key]["tp"] += row["tp"]
        grouped[key]["fn"] += row["fn"]

    rows = []

    for size_label, _, _ in PERSON_SIZE_BINS:

        for scale in INPUT_SCALES:

            values = grouped[
                (
                    size_label,
                    scale,
                )
            ]

            gt = values["gt"]
            tp = values["tp"]
            fn = values["fn"]

            recall = (
                tp / gt
                if gt > 0
                else 0.0
            )

            rows.append(
                {
                    "person_size": size_label,
                    "input_scale": scale,
                    "person_gt": gt,
                    "person_tp": tp,
                    "person_fn": fn,
                    "person_recall": recall,
                }
            )

    return rows


# ============================================================================
# ESTADÍSTICAS
# ============================================================================

def get_statistics(records):

    grouped = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
            "area_sum": 0.0,
            "area_min": float("inf"),
            "area_max": 0.0,
        }
    )

    for row in records:

        key = (
            row["person_size"],
            row["input_scale"],
        )

        data = grouped[key]

        area = row["area"]

        data["gt"] += row["gt"]
        data["tp"] += row["tp"]
        data["fn"] += row["fn"]

        data["area_sum"] += area

        data["area_min"] = min(
            data["area_min"],
            area,
        )

        data["area_max"] = max(
            data["area_max"],
            area,
        )

    rows = []

    for size_label, _, _ in PERSON_SIZE_BINS:

        for scale in INPUT_SCALES:

            key = (
                size_label,
                scale,
            )

            data = grouped[key]

            gt = data["gt"]
            tp = data["tp"]
            fn = data["fn"]

            if gt > 0:

                mean_area = (
                    data["area_sum"] / gt
                )

                recall = tp / gt

                min_area = data["area_min"]

                max_area = data["area_max"]

            else:

                mean_area = 0.0
                recall = 0.0
                min_area = 0.0
                max_area = 0.0

            rows.append(
                {
                    "person_size": size_label,
                    "input_scale": scale,
                    "person_gt": gt,
                    "person_tp": tp,
                    "person_fn": fn,
                    "person_recall": recall,
                    "mean_area": mean_area,
                    "min_area": min_area,
                    "max_area": max_area,
                }
            )

    return rows


def write_summary_csv(rows):

    fields = [
        "input_scale",
        "person_gt",
        "person_tp",
        "person_fn",
        "person_recall",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    "input_scale": row["input_scale"],
                    "person_gt": row["person_gt"],
                    "person_tp": row["person_tp"],
                    "person_fn": row["person_fn"],
                    "person_recall": round(
                        row["person_recall"],
                        6,
                    ),
                }
            )


def write_statistics_csv(rows):

    fields = [
        "person_size",
        "input_scale",
        "person_gt",
        "person_tp",
        "person_fn",
        "person_recall",
        "mean_area",
        "min_area",
        "max_area",
    ]

    with STATISTICS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    "person_size": row["person_size"],
                    "input_scale": row["input_scale"],
                    "person_gt": row["person_gt"],
                    "person_tp": row["person_tp"],
                    "person_fn": row["person_fn"],
                    "person_recall": round(
                        row["person_recall"],
                        6,
                    ),
                    "mean_area": round(
                        row["mean_area"],
                        4,
                    ),
                    "min_area": round(
                        row["min_area"],
                        4,
                    ),
                    "max_area": round(
                        row["max_area"],
                        4,
                    ),
                }
            )


# ============================================================================
# SUMMARY TXT
# ============================================================================

def write_summary_txt(
    images_count,
    scale_summary,
    size_scale_summary,
):

    total_gt = (
        scale_summary[0]["person_gt"]
        if scale_summary
        else 0
    )

    with SUMMARY_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "========================================================================\n"
        )
        f.write(
            "# SAR YOLO26 - PERSON RECALL BY PERSON SIZE AND INPUT SCALE V1\n"
        )
        f.write(
            "========================================================================\n\n"
        )

        f.write(
            f"Imágenes:              {images_count:,}\n"
        )

        f.write(
            f"PERSON GT:             {total_gt:,}\n\n"
        )

        f.write(
            "RECALL POR INPUT SCALE\n\n"
        )

        for row in scale_summary:

            f.write(
                f"INPUT SCALE {row['input_scale']}: "
                f"GT={row['person_gt']:,} "
                f"TP={row['person_tp']:,} "
                f"FN={row['person_fn']:,} "
                f"Recall={row['person_recall']:.4f}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "RECALL POR TAMAÑO DE PERSONA + INPUT SCALE\n\n"
        )

        current_size = None

        for row in size_scale_summary:

            size = row["person_size"]

            if size != current_size:

                f.write(
                    f"\nPERSON SIZE: {size}\n"
                )

                current_size = size

            f.write(
                f"Input {row['input_scale']:4d} "
                f"GT={row['person_gt']:7,} "
                f"TP={row['person_tp']:7,} "
                f"FN={row['person_fn']:7,} "
                f"Recall={row['person_recall']:.4f}\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "========================================================================\n"
        )

        f.write(
            "IMPORTANTE: el dataset NO ha sido modificado.\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print(
        "========================================================================"
    )
    print(
        "# SAR YOLO26 - PERSON RECALL BY PERSON SIZE AND INPUT SCALE ANALYSIS V1"
    )
    print(
        "========================================================================"
    )
    print()

    print("Dataset:")
    print(DATASET_ROOT)
    print()

    print("Modelo:")
    print(MODEL_PATH)
    print()

    print("Test:")
    print(TEST_IMAGES_DIR)
    print()

    print("Output:")
    print(OUTPUT_ROOT)
    print()

    ensure_directories()

    images = list_images()

    print(
        f"Imágenes encontradas: {len(images):,}"
    )
    print()

    if not images:

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

    print(
        "[OK] Modelo cargado."
    )
    print()

    all_records = []

    # ========================================================================
    # Ejecutar cada input scale
    # ========================================================================

    for input_scale in INPUT_SCALES:

        print(
            "------------------------------------------------------------------------"
        )

        print(
            f"ANALIZANDO INPUT SCALE: {input_scale}"
        )

        print(
            "------------------------------------------------------------------------"
        )

        scale_records = []

        total_images = len(images)

        for index, image_path in enumerate(
            images,
            start=1,
        ):

            records = analyze_image(
                model=model,
                image_path=image_path,
                input_scale=input_scale,
            )

            scale_records.extend(
                records
            )

            if (
                index % 100 == 0
                or index == total_images
            ):

                print(
                    f"Analizadas: "
                    f"{index:,}/{total_images:,}"
                )

        all_records.extend(
            scale_records
        )

        gt = sum(
            r["gt"]
            for r in scale_records
        )

        tp = sum(
            r["tp"]
            for r in scale_records
        )

        fn = sum(
            r["fn"]
            for r in scale_records
        )

        recall = (
            tp / gt
            if gt > 0
            else 0.0
        )

        print()

        print(
            f"INPUT SCALE {input_scale}: "
            f"GT={gt:,} "
            f"TP={tp:,} "
            f"FN={fn:,} "
            f"Recall={recall:.4f}"
        )

        print()

    # ========================================================================
    # GUARDAR OBJETOS
    # ========================================================================

    write_objects_csv(
        all_records
    )

    print(
        f"[OK] {OBJECTS_CSV}"
    )

    # ========================================================================
    # RESUMEN POR SCALE
    # ========================================================================

    scale_summary = get_scale_summary(
        all_records
    )

    write_summary_csv(
        scale_summary
    )

    print(
        f"[OK] {SUMMARY_CSV}"
    )

    # ========================================================================
    # RESUMEN SIZE + SCALE
    # ========================================================================

    size_scale_summary = get_size_scale_summary(
        all_records
    )

    # ========================================================================
    # ESTADÍSTICAS
    # ========================================================================

    statistics_rows = get_statistics(
        all_records
    )

    write_statistics_csv(
        statistics_rows
    )

    print(
        f"[OK] {STATISTICS_CSV}"
    )

    # ========================================================================
    # SUMMARY TXT
    # ========================================================================

    write_summary_txt(
        images_count=len(images),
        scale_summary=scale_summary,
        size_scale_summary=size_scale_summary,
    )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    # ========================================================================
    # RESULTADO FINAL
    # ========================================================================

    print()
    print(
        "========================================================================"
    )
    print(
        "# RESULTADO PERSON RECALL BY PERSON SIZE AND INPUT SCALE V1"
    )
    print(
        "========================================================================"
    )
    print()

    print(
        f"Imágenes:              {len(images):,}"
    )

    total_gt = (
        scale_summary[0]["person_gt"]
        if scale_summary
        else 0
    )

    print(
        f"PERSON GT:             {total_gt:,}"
    )

    print()

    for row in scale_summary:

        print(
            f"INPUT SCALE {row['input_scale']}: "
            f"GT={row['person_gt']:,} "
            f"TP={row['person_tp']:,} "
            f"FN={row['person_fn']:,} "
            f"Recall={row['person_recall']:.4f}"
        )

    print()

    print(
        "RECALL POR TAMAÑO DE PERSONA + INPUT SCALE"
    )

    print()

    current_size = None

    for row in size_scale_summary:

        size = row["person_size"]

        if size != current_size:

            print(
                f"\nPERSON SIZE: {size}"
            )

            current_size = size

        print(
            f"Input {row['input_scale']:4d} "
            f"GT={row['person_gt']:7,} "
            f"TP={row['person_tp']:7,} "
            f"FN={row['person_fn']:7,} "
            f"Recall={row['person_recall']:.4f}"
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

    print(
        "========================================================================"
    )


if __name__ == "__main__":
    main()