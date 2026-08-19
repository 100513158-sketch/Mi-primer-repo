from pathlib import Path
from collections import defaultdict
import csv
import math

from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASELINE_DIR = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

DATASET_DIR = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
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
    / "recall_by_person_size_and_density"
    / "analyze_person_recall_by_person_size_and_density_v1"
)

REPORTS_DIR = OUTPUT_DIR / "reports"

OBJECTS_CSV = (
    REPORTS_DIR
    / "person_recall_by_person_size_and_density_objects_v1.csv"
)

SUMMARY_CSV = (
    REPORTS_DIR
    / "person_recall_by_person_size_and_density_v1.csv"
)

STATISTICS_CSV = (
    REPORTS_DIR
    / "person_size_density_statistics_v1.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "PERSON_RECALL_BY_PERSON_SIZE_AND_DENSITY_V1_SUMMARY.txt"
)


# ============================================================
# PARÁMETROS
# ============================================================

PERSON_CLASS = 0

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


# ============================================================
# ESCALAS DE PERSONA
# ============================================================

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


# ============================================================
# DENSIDAD DE PERSONAS
#
# Se utiliza el número TOTAL de objetos anotados en la imagen,
# manteniendo el mismo criterio utilizado anteriormente.
# ============================================================

DENSITY_BINS = [
    ("<25", 0, 25),
    ("25-49", 25, 50),
    ("50-99", 50, 100),
    ("100-199", 100, 200),
    ("200-299", 200, 300),
    ("300-499", 300, 500),
    (">=500", 500, float("inf")),
]


# ============================================================
# UTILIDADES
# ============================================================

def ensure_output_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_images():
    images = []

    for path in TEST_IMAGES_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def find_label_file(image_path):
    return TEST_LABELS_DIR / f"{image_path.stem}.txt"


def load_yolo_labels(label_path):
    """
    Devuelve:

    [
        {
            "class_id": int,
            "x_center": float,
            "y_center": float,
            "width": float,
            "height": float
        }
    ]

    Formato YOLO:
    class x_center y_center width height
    """

    objects = []

    if not label_path.exists():
        return objects

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
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            objects.append(
                {
                    "class_id": class_id,
                    "x_center": xc,
                    "y_center": yc,
                    "width": w,
                    "height": h,
                }
            )

    return objects


def normalized_box_to_pixel(obj, image_width, image_height):
    xc = obj["x_center"] * image_width
    yc = obj["y_center"] * image_height

    w = obj["width"] * image_width
    h = obj["height"] * image_height

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return [x1, y1, x2, y2]


def box_area(box):
    x1, y1, x2, y2 = box

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    return w * h


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return (ix2 - ix1) * (iy2 - iy1)


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


def get_person_size_bin(area):
    for label, low, high in PERSON_SIZE_BINS:
        if low <= area < high:
            return label

    return ">2048"


def get_density_bin(total_objects):
    for label, low, high in DENSITY_BINS:
        if low <= total_objects < high:
            return label

    return ">=500"


def calculate_recall(tp, gt):
    if gt == 0:
        return 0.0

    return tp / gt


# ============================================================
# MATCHING
# ============================================================

def match_person_predictions(gt_persons, predictions):
    """
    Matching greedy por IoU.

    Solo se consideran predicciones PERSON.

    Devuelve:
        matched_gt_indices
    """

    candidates = []

    for gt_idx, gt in enumerate(gt_persons):

        for pred_idx, pred in enumerate(predictions):

            score = iou(gt["box"], pred["box"])

            if score >= IOU_MATCH_THRESHOLD:
                candidates.append(
                    (
                        score,
                        gt_idx,
                        pred_idx,
                    )
                )

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for score, gt_idx, pred_idx in candidates:

        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    return matched_gt


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("# SAR YOLO26 - PERSON RECALL BY PERSON SIZE AND DENSITY ANALYSIS V1")
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

    ensure_output_dirs()

    # --------------------------------------------------------
    # IMÁGENES
    # --------------------------------------------------------

    image_paths = get_images()

    print()
    print(f"Imágenes encontradas: {len(image_paths)}")

    if not image_paths:
        raise RuntimeError("No se encontraron imágenes.")

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    print()
    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL_PATH))

    print("[OK] Modelo cargado.")

    # --------------------------------------------------------
    # ESTRUCTURAS
    # --------------------------------------------------------

    objects_rows = []

    combinations = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    size_statistics = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
            "areas": [],
            "images": set(),
        }
    )

    density_statistics = defaultdict(
        lambda: {
            "images": set(),
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    # --------------------------------------------------------
    # PROCESAMIENTO
    # --------------------------------------------------------

    print()
    print("Analizando PERSON por tamaño + densidad...")

    for index, image_path in enumerate(image_paths, start=1):

        # ----------------------------------------------------
        # IMAGEN
        # ----------------------------------------------------

        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as exc:
            print(f"[WARN] No se pudo abrir {image_path}: {exc}")
            continue

        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        label_path = find_label_file(image_path)

        all_gt = load_yolo_labels(label_path)

        # DENSIDAD:
        # total de objetos anotados en la imagen
        total_objects = len(all_gt)

        density_bin = get_density_bin(total_objects)

        # ----------------------------------------------------
        # PERSON GT
        # ----------------------------------------------------

        gt_persons = []

        for obj in all_gt:

            if obj["class_id"] != PERSON_CLASS:
                continue

            box = normalized_box_to_pixel(
                obj,
                image_width,
                image_height,
            )

            area = box_area(box)

            size_bin = get_person_size_bin(area)

            gt_persons.append(
                {
                    "box": box,
                    "area": area,
                    "size_bin": size_bin,
                }
            )

        # ----------------------------------------------------
        # PREDICCIONES
        # ----------------------------------------------------

        try:

            results = model.predict(
                source=str(image_path),
                conf=CONF_THRESHOLD,
                verbose=False,
            )

        except Exception as exc:

            print(
                f"[WARN] Error procesando "
                f"{image_path.name}: {exc}"
            )

            continue

        predictions = []

        if results:

            result = results[0]

            if result.boxes is not None:

                for box_obj in result.boxes:

                    cls = int(
                        box_obj.cls.item()
                    )

                    if cls != PERSON_CLASS:
                        continue

                    xyxy = (
                        box_obj.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    confidence = float(
                        box_obj.conf.item()
                    )

                    predictions.append(
                        {
                            "box": xyxy,
                            "confidence": confidence,
                        }
                    )

        # ----------------------------------------------------
        # MATCHING
        # ----------------------------------------------------

        matched_gt = match_person_predictions(
            gt_persons,
            predictions,
        )

        # ----------------------------------------------------
        # PERSON OBJECTS
        # ----------------------------------------------------

        for gt_idx, gt in enumerate(gt_persons):

            is_tp = gt_idx in matched_gt

            is_fn = not is_tp

            if is_tp:
                total_person_tp += 1
            else:
                total_person_fn += 1

            total_person_gt += 1

            size_bin = gt["size_bin"]

            # ----------------------------
            # combinación tamaño+density
            # ----------------------------

            key = (
                size_bin,
                density_bin,
            )

            combinations[key]["gt"] += 1

            if is_tp:
                combinations[key]["tp"] += 1
            else:
                combinations[key]["fn"] += 1

            # ----------------------------
            # estadísticas por tamaño
            # ----------------------------

            size_statistics[size_bin]["gt"] += 1
            size_statistics[size_bin]["areas"].append(
                gt["area"]
            )
            size_statistics[size_bin]["images"].add(
                image_path.name
            )

            if is_tp:
                size_statistics[size_bin]["tp"] += 1
            else:
                size_statistics[size_bin]["fn"] += 1

            # ----------------------------
            # estadísticas por densidad
            # ----------------------------

            density_statistics[density_bin]["images"].add(
                image_path.name
            )

            density_statistics[density_bin]["gt"] += 1

            if is_tp:
                density_statistics[density_bin]["tp"] += 1
            else:
                density_statistics[density_bin]["fn"] += 1

            # ----------------------------
            # objeto
            # ----------------------------

            objects_rows.append(
                {
                    "image": image_path.name,
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_area": image_width * image_height,
                    "total_objects": total_objects,
                    "density_bin": density_bin,
                    "person_area": round(gt["area"], 4),
                    "person_size_bin": size_bin,
                    "result": "TP" if is_tp else "FN",
                }
            )

        # ----------------------------------------------------
        # PROGRESO
        # ----------------------------------------------------

        if index % 100 == 0 or index == len(image_paths):

            print(
                f"Analizadas: "
                f"{index:,}/{len(image_paths):,}"
            )

    # ========================================================
    # GENERAR SUMMARY
    # ========================================================

    overall_recall = calculate_recall(
        total_person_tp,
        total_person_gt,
    )

    # --------------------------------------------------------
    # OBJECT CSV
    # --------------------------------------------------------

    with OBJECTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "image_width",
                "image_height",
                "image_area",
                "total_objects",
                "density_bin",
                "person_area",
                "person_size_bin",
                "result",
            ],
        )

        writer.writeheader()
        writer.writerows(objects_rows)

    print()
    print(f"[OK] {OBJECTS_CSV}")

    # --------------------------------------------------------
    # SUMMARY CSV
    # --------------------------------------------------------

    summary_rows = []

    for size_label, _, _ in PERSON_SIZE_BINS:

        for density_label, _, _ in DENSITY_BINS:

            data = combinations[
                (
                    size_label,
                    density_label,
                )
            ]

            gt = data["gt"]
            tp = data["tp"]
            fn = data["fn"]

            recall = calculate_recall(tp, gt)

            summary_rows.append(
                {
                    "person_size_bin": size_label,
                    "density_bin": density_label,
                    "gt": gt,
                    "tp": tp,
                    "fn": fn,
                    "recall": round(recall, 6),
                    "percentage_of_person_gt": round(
                        (gt / total_person_gt * 100)
                        if total_person_gt
                        else 0,
                        4,
                    ),
                }
            )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "person_size_bin",
                "density_bin",
                "gt",
                "tp",
                "fn",
                "recall",
                "percentage_of_person_gt",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] {SUMMARY_CSV}")

    # --------------------------------------------------------
    # STATISTICS CSV
    # --------------------------------------------------------

    statistics_rows = []

    for size_label, _, _ in PERSON_SIZE_BINS:

        data = size_statistics[size_label]

        areas = data["areas"]

        if areas:

            areas_sorted = sorted(areas)

            median = areas_sorted[
                len(areas_sorted) // 2
            ]

            mean = sum(areas) / len(areas)

            p90_index = min(
                len(areas_sorted) - 1,
                int(
                    math.ceil(
                        0.90 * len(areas_sorted)
                    )
                ) - 1,
            )

            p90 = areas_sorted[p90_index]

            min_area = min(areas)
            max_area = max(areas)

        else:

            mean = 0
            median = 0
            p90 = 0
            min_area = 0
            max_area = 0

        gt = data["gt"]
        tp = data["tp"]
        fn = data["fn"]

        statistics_rows.append(
            {
                "person_size_bin": size_label,
                "images": len(data["images"]),
                "gt": gt,
                "tp": tp,
                "fn": fn,
                "recall": round(
                    calculate_recall(tp, gt),
                    6,
                ),
                "mean_area": round(mean, 4),
                "median_area": round(median, 4),
                "p90_area": round(p90, 4),
                "min_area": round(min_area, 4),
                "max_area": round(max_area, 4),
            }
        )

    # --------------------------------------------------------
    # DENSITY STATS
    # --------------------------------------------------------

    for density_label, _, _ in DENSITY_BINS:

        data = density_statistics[density_label]

        gt = data["gt"]
        tp = data["tp"]
        fn = data["fn"]

        statistics_rows.append(
            {
                "person_size_bin": "ALL",
                "images": len(data["images"]),
                "gt": gt,
                "tp": tp,
                "fn": fn,
                "recall": round(
                    calculate_recall(tp, gt),
                    6,
                ),
                "mean_area": "",
                "median_area": "",
                "p90_area": "",
                "min_area": "",
                "max_area": "",
            }
        )

    with STATISTICS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "person_size_bin",
                "images",
                "gt",
                "tp",
                "fn",
                "recall",
                "mean_area",
                "median_area",
                "p90_area",
                "min_area",
                "max_area",
            ],
        )

        writer.writeheader()
        writer.writerows(statistics_rows)

    print(f"[OK] {STATISTICS_CSV}")

    # ========================================================
    # TXT SUMMARY
    # ========================================================

    with SUMMARY_TXT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON RECALL BY PERSON SIZE "
            "AND DENSITY ANALYSIS V1\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(f"Dataset: {DATASET_DIR}\n")
        f.write(f"Modelo: {MODEL_PATH}\n")
        f.write(f"Test: {TEST_IMAGES_DIR}\n\n")

        f.write("RESULTADO GENERAL\n")
        f.write("-" * 72 + "\n")

        f.write(
            f"Imágenes: {len(image_paths):,}\n"
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
            f"PERSON Recall: {overall_recall:.4f}\n\n"
        )

        f.write(
            "RECALL POR TAMAÑO DE PERSONA Y DENSIDAD\n"
        )

        f.write("=" * 72 + "\n\n")

        for size_label, _, _ in PERSON_SIZE_BINS:

            f.write(
                f"\nPERSON SIZE: {size_label}\n"
            )

            f.write("-" * 72 + "\n")

            for density_label, _, _ in DENSITY_BINS:

                data = combinations[
                    (
                        size_label,
                        density_label,
                    )
                ]

                gt = data["gt"]
                tp = data["tp"]
                fn = data["fn"]

                recall = calculate_recall(
                    tp,
                    gt,
                )

                percentage = (
                    gt / total_person_gt * 100
                    if total_person_gt
                    else 0
                )

                f.write(
                    f"{density_label:>8} "
                    f"GT={gt:6d} "
                    f"TP={tp:6d} "
                    f"FN={fn:6d} "
                    f"Recall={recall:.4f} "
                    f"({percentage:6.2f} %)\n"
                )

        f.write("\n")
        f.write("=" * 72 + "\n")
        f.write(
            "IMPORTANTE: el dataset NO ha sido modificado.\n"
        )

    print(f"[OK] {SUMMARY_TXT}")

    # ========================================================
    # CONSOLA
    # ========================================================

    print()
    print("=" * 72)
    print("# RESULTADO PERSON RECALL BY PERSON SIZE AND DENSITY V1")
    print("=" * 72)

    print()
    print(f"Imágenes:              {len(image_paths):,}")
    print(f"PERSON GT:             {total_person_gt:,}")
    print(f"PERSON TP:             {total_person_tp:,}")
    print(f"PERSON FN:             {total_person_fn:,}")
    print(f"PERSON Recall:         {overall_recall:.4f}")

    print()
    print("RECALL POR TAMAÑO DE PERSONA + DENSIDAD")
    print()

    for size_label, _, _ in PERSON_SIZE_BINS:

        print()
        print(f"PERSON SIZE: {size_label}")

        for density_label, _, _ in DENSITY_BINS:

            data = combinations[
                (
                    size_label,
                    density_label,
                )
            ]

            gt = data["gt"]
            tp = data["tp"]
            fn = data["fn"]

            recall = calculate_recall(
                tp,
                gt,
            )

            if gt > 0:

                print(
                    f"{density_label:>8} "
                    f"GT={gt:6d} "
                    f"TP={tp:6d} "
                    f"FN={fn:6d} "
                    f"Recall={recall:.4f}"
                )

    print()
    print("[OK] Reports generados.")

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()