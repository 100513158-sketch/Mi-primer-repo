from pathlib import Path
from collections import defaultdict, Counter
import csv
import math

from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"\baseline\training\runs\baseline_v1\weights\best.pt"
)

TEST_IMAGES = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS = DATASET_ROOT / "test_dev" / "labels"

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"\baseline\evaluation\dataset_analysis"
    r"\detection_failure_analysis"
    r"\person_false_negatives"
    r"\analyze_person_false_negatives_v1"
)

REPORTS_DIR = OUTPUT_ROOT / "reports"

# Clases YOLO
PERSON_CLASS = 0
VEHICLE_CLASS = 1

# Debe coincidir con detection_failure_analysis_v1
IOU_THRESHOLD = 0.50

# Confianza mínima para considerar una predicción
CONF_THRESHOLD = 0.25

# Intervalos de área en píxeles cuadrados
AREA_BINS = [
    ("<16", 0, 16),
    ("16-32", 16, 32),
    ("32-64", 32, 64),
    ("64-100", 64, 100),
    (">=100", 100, float("inf")),
]

# Densidad
DENSITY_BINS = [
    ("<50", 0, 50),
    ("50-99", 50, 100),
    ("100-199", 100, 200),
    ("200-299", 200, 300),
    ("300-499", 300, 500),
    (">=500", 500, float("inf")),
]

# Distancia mínima al borde como porcentaje del ancho/alto
BORDER_THRESHOLD = 0.02


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


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


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def iou(a, b):
    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


def area_bin(area):
    for name, low, high in AREA_BINS:
        if low <= area < high:
            return name
    return "UNKNOWN"


def density_bin(objects):
    for name, low, high in DENSITY_BINS:
        if low <= objects < high:
            return name
    return "UNKNOWN"


def border_distance(box, img_w, img_h):
    x1, y1, x2, y2 = box

    distances = [
        x1 / img_w,
        y1 / img_h,
        (img_w - x2) / img_w,
        (img_h - y2) / img_h,
    ]

    return min(distances)


def is_near_border(box, img_w, img_h):
    return border_distance(box, img_w, img_h) <= BORDER_THRESHOLD


def is_partial_bbox(box, img_w, img_h):
    x1, y1, x2, y2 = box

    return (
        x1 < 0
        or y1 < 0
        or x2 > img_w
        or y2 > img_h
    )


def read_yolo_labels(label_path, img_w, img_h):
    objects = []

    if not label_path.exists():
        return objects

    with label_path.open("r", encoding="utf-8") as f:
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
                x, y, w, h, img_w, img_h
            )

            objects.append(
                {
                    "class": cls,
                    "box": box,
                    "area": box_area(box),
                    "line": line_number,
                }
            )

    return objects


def match_predictions(gt_objects, predictions):
    """
    Greedy one-to-one matching por IoU.

    Primero busca la mejor predicción para cada GT.
    Solo se acepta si:
        IoU >= IOU_THRESHOLD
    """

    candidates = []

    for gt_idx, gt in enumerate(gt_objects):
        for pred_idx, pred in enumerate(predictions):
            score = iou(gt["box"], pred["box"])

            if score >= IOU_THRESHOLD:
                candidates.append(
                    (score, gt_idx, pred_idx)
                )

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for score, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

        matches.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "iou": score,
            }
        )

    return matches


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("# SAR YOLO26 - PERSON FALSE NEGATIVES ANALYSIS V1")
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

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATASET_ROOT}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"Test images no encontrado: {TEST_IMAGES}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        [
            p
            for p in TEST_IMAGES.iterdir()
            if p.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )

    print(f"Imágenes encontradas: {len(image_paths)}")
    print()

    print("Cargando modelo YOLO26s...")

    model = YOLO(str(MODEL_PATH))

    print("[OK] Modelo cargado.")
    print()

    # ========================================================
    # ACUMULADORES
    # ========================================================

    fn_objects = []
    image_statistics = []

    area_stats = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    density_stats = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    border_stats = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    confusion_counter = Counter()

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    total_person_predictions = 0
    total_vehicle_predictions = 0

    fn_areas = []
    tp_areas = []

    # ========================================================
    # PROCESAMIENTO
    # ========================================================

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        # ----------------------------------------------------
        # Obtener dimensiones mediante OpenCV
        # ----------------------------------------------------

        import cv2

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        label_path = (
            TEST_LABELS
            / f"{image_path.stem}.txt"
        )

        gt_objects = read_yolo_labels(
            label_path,
            img_w,
            img_h,
        )

        person_gt = [
            obj
            for obj in gt_objects
            if obj["class"] == PERSON_CLASS
        ]

        vehicle_gt = [
            obj
            for obj in gt_objects
            if obj["class"] == VEHICLE_CLASS
        ]

        total_objects = len(gt_objects)

        # ----------------------------------------------------
        # Predicciones
        # ----------------------------------------------------

        result = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        predictions = []

        if result.boxes is not None:

            for box in result.boxes:

                cls = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coords = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                predictions.append(
                    {
                        "class": cls,
                        "confidence": confidence,
                        "box": tuple(coords),
                    }
                )

        person_predictions = [
            p
            for p in predictions
            if p["class"] == PERSON_CLASS
        ]

        vehicle_predictions = [
            p
            for p in predictions
            if p["class"] == VEHICLE_CLASS
        ]

        total_person_predictions += len(
            person_predictions
        )

        total_vehicle_predictions += len(
            vehicle_predictions
        )

        # ----------------------------------------------------
        # Match PERSON
        # ----------------------------------------------------

        matches = match_predictions(
            person_gt,
            person_predictions,
        )

        matched_gt = {
            m["gt_idx"]
            for m in matches
        }

        matched_pred = {
            m["pred_idx"]
            for m in matches
        }

        total_person_gt += len(person_gt)
        total_person_tp += len(matches)

        total_person_fn += (
            len(person_gt)
            - len(matches)
        )

        # ----------------------------------------------------
        # Analizar cada PERSON GT
        # ----------------------------------------------------

        image_fn = 0
        image_tp = 0

        for gt_idx, gt in enumerate(
            person_gt
        ):

            area = gt["area"]
            a_bin = area_bin(area)

            near_border = is_near_border(
                gt["box"],
                img_w,
                img_h,
            )

            partial = is_partial_bbox(
                gt["box"],
                img_w,
                img_h,
            )

            d_bin = density_bin(
                total_objects
            )

            area_stats[a_bin]["gt"] += 1
            density_stats[d_bin]["gt"] += 1

            border_category = (
                "border"
                if near_border
                else "not_border"
            )

            border_stats[
                border_category
            ]["gt"] += 1

            matched = gt_idx in matched_gt

            if matched:

                image_tp += 1

                area_stats[a_bin]["tp"] += 1
                density_stats[d_bin]["tp"] += 1
                border_stats[
                    border_category
                ]["tp"] += 1

                tp_areas.append(area)

            else:

                image_fn += 1

                area_stats[a_bin]["fn"] += 1
                density_stats[d_bin]["fn"] += 1
                border_stats[
                    border_category
                ]["fn"] += 1

                fn_areas.append(area)

                # ------------------------------------------------
                # Buscar posible confusión PERSON -> VEHICLE
                # ------------------------------------------------

                best_vehicle_iou = 0.0
                best_vehicle_conf = 0.0

                for pred in vehicle_predictions:

                    score = iou(
                        gt["box"],
                        pred["box"],
                    )

                    if score > best_vehicle_iou:
                        best_vehicle_iou = score
                        best_vehicle_conf = (
                            pred["confidence"]
                        )

                if best_vehicle_iou >= IOU_THRESHOLD:
                    confusion_type = (
                        "PERSON_TO_VEHICLE"
                    )
                    confusion_counter[
                        "PERSON_TO_VEHICLE"
                    ] += 1

                elif best_vehicle_iou >= 0.25:
                    confusion_type = (
                        "PARTIAL_VEHICLE_OVERLAP"
                    )
                    confusion_counter[
                        "PARTIAL_VEHICLE_OVERLAP"
                    ] += 1

                else:
                    confusion_type = (
                        "NO_VEHICLE_MATCH"
                    )
                    confusion_counter[
                        "NO_VEHICLE_MATCH"
                    ] += 1

                fn_objects.append(
                    {
                        "image": image_path.name,
                        "image_path": str(
                            image_path
                        ),
                        "split": "test_dev",
                        "gt_class": "PERSON",
                        "gt_class_id": PERSON_CLASS,
                        "area_px2": round(
                            area,
                            4,
                        ),
                        "area_bin": a_bin,
                        "width_px": round(
                            gt["box"][2]
                            - gt["box"][0],
                            4,
                        ),
                        "height_px": round(
                            gt["box"][3]
                            - gt["box"][1],
                            4,
                        ),
                        "objects_in_image":
                            total_objects,
                        "density_bin": d_bin,
                        "near_border": int(
                            near_border
                        ),
                        "partial_bbox": int(
                            partial
                        ),
                        "border_distance_pct":
                            round(
                                border_distance(
                                    gt["box"],
                                    img_w,
                                    img_h,
                                )
                                * 100,
                                4,
                            ),
                        "best_vehicle_iou":
                            round(
                                best_vehicle_iou,
                                4,
                            ),
                        "best_vehicle_conf":
                            round(
                                best_vehicle_conf,
                                4,
                            ),
                        "failure_type":
                            confusion_type,
                        "image_width": img_w,
                        "image_height": img_h,
                    }
                )

        image_statistics.append(
            {
                "image": image_path.name,
                "image_path": str(image_path),
                "person_gt": len(person_gt),
                "person_tp": image_tp,
                "person_fn": image_fn,
                "objects_total": total_objects,
                "person_recall": (
                    image_tp / len(person_gt)
                    if person_gt
                    else 0.0
                ),
            }
        )

        if (
            index % 100 == 0
            or index == len(image_paths)
        ):
            print(
                f"Analizadas: "
                f"{index:,}/{len(image_paths):,}"
            )

    # ========================================================
    # GENERAR REPORTES
    # ========================================================

    print()
    print("=" * 72)
    print("# RESULTADO PERSON FALSE NEGATIVES ANALYSIS V1")
    print("=" * 72)
    print()

    global_recall = (
        total_person_tp / total_person_gt
        if total_person_gt
        else 0.0
    )

    print(
        f"PERSON GT:              "
        f"{total_person_gt:,}"
    )

    print(
        f"PERSON TP:              "
        f"{total_person_tp:,}"
    )

    print(
        f"PERSON FN:              "
        f"{total_person_fn:,}"
    )

    print(
        f"PERSON Recall:          "
        f"{global_recall:.4f}"
    )

    print()

    print("FN POR TIPO")

    for key in [
        "NO_VEHICLE_MATCH",
        "PARTIAL_VEHICLE_OVERLAP",
        "PERSON_TO_VEHICLE",
    ]:

        value = confusion_counter[key]

        pct = (
            value / total_person_fn * 100
            if total_person_fn
            else 0.0
        )

        print(
            f"{key:28s}: "
            f"{value:7,} "
            f"({pct:6.2f} %)"
        )

    # ========================================================
    # CSV 1 - FN OBJECTS
    # ========================================================

    write_csv(
        REPORTS_DIR
        / "person_fn_objects_v1.csv",
        fn_objects,
        [
            "image",
            "image_path",
            "split",
            "gt_class",
            "gt_class_id",
            "area_px2",
            "area_bin",
            "width_px",
            "height_px",
            "objects_in_image",
            "density_bin",
            "near_border",
            "partial_bbox",
            "border_distance_pct",
            "best_vehicle_iou",
            "best_vehicle_conf",
            "failure_type",
            "image_width",
            "image_height",
        ],
    )

    # ========================================================
    # CSV 2 - FN IMAGES
    # ========================================================

    fn_images = [
        row
        for row in image_statistics
        if row["person_fn"] > 0
    ]

    write_csv(
        REPORTS_DIR
        / "person_fn_images_v1.csv",
        fn_images,
        [
            "image",
            "image_path",
            "person_gt",
            "person_tp",
            "person_fn",
            "objects_total",
            "person_recall",
        ],
    )

    # ========================================================
    # CSV 3 - RECALL BY AREA
    # ========================================================

    area_rows = []

    for name, _, _ in AREA_BINS:

        data = area_stats[name]

        recall = (
            data["tp"] / data["gt"]
            if data["gt"]
            else 0.0
        )

        area_rows.append(
            {
                "area_bin": name,
                "gt": data["gt"],
                "tp": data["tp"],
                "fn": data["fn"],
                "recall": round(
                    recall,
                    6,
                ),
                "fn_pct": round(
                    data["fn"]
                    / data["gt"]
                    * 100
                    if data["gt"]
                    else 0,
                    4,
                ),
            }
        )

    write_csv(
        REPORTS_DIR
        / "person_recall_by_area_v1.csv",
        area_rows,
        [
            "area_bin",
            "gt",
            "tp",
            "fn",
            "recall",
            "fn_pct",
        ],
    )

    # ========================================================
    # CSV 4 - RECALL BY DENSITY
    # ========================================================

    density_rows = []

    for name, _, _ in DENSITY_BINS:

        data = density_stats[name]

        recall = (
            data["tp"] / data["gt"]
            if data["gt"]
            else 0.0
        )

        density_rows.append(
            {
                "density_bin": name,
                "gt": data["gt"],
                "tp": data["tp"],
                "fn": data["fn"],
                "recall": round(
                    recall,
                    6,
                ),
                "fn_pct": round(
                    data["fn"]
                    / data["gt"]
                    * 100
                    if data["gt"]
                    else 0,
                    4,
                ),
            }
        )

    write_csv(
        REPORTS_DIR
        / "person_recall_by_density_v1.csv",
        density_rows,
        [
            "density_bin",
            "gt",
            "tp",
            "fn",
            "recall",
            "fn_pct",
        ],
    )

    # ========================================================
    # CSV 5 - RECALL BY BORDER
    # ========================================================

    border_rows = []

    for name in [
        "border",
        "not_border",
    ]:

        data = border_stats[name]

        recall = (
            data["tp"] / data["gt"]
            if data["gt"]
            else 0.0
        )

        border_rows.append(
            {
                "border_category": name,
                "gt": data["gt"],
                "tp": data["tp"],
                "fn": data["fn"],
                "recall": round(
                    recall,
                    6,
                ),
                "fn_pct": round(
                    data["fn"]
                    / data["gt"]
                    * 100
                    if data["gt"]
                    else 0,
                    4,
                ),
            }
        )

    write_csv(
        REPORTS_DIR
        / "person_recall_by_border_v1.csv",
        border_rows,
        [
            "border_category",
            "gt",
            "tp",
            "fn",
            "recall",
            "fn_pct",
        ],
    )

    # ========================================================
    # CSV 6 - CLASS CONFUSION
    # ========================================================

    confusion_rows = []

    for key in [
        "NO_VEHICLE_MATCH",
        "PARTIAL_VEHICLE_OVERLAP",
        "PERSON_TO_VEHICLE",
    ]:

        value = confusion_counter[key]

        confusion_rows.append(
            {
                "failure_type": key,
                "count": value,
                "percentage": round(
                    value
                    / total_person_fn
                    * 100
                    if total_person_fn
                    else 0,
                    4,
                ),
            }
        )

    write_csv(
        REPORTS_DIR
        / "person_class_confusion_v1.csv",
        confusion_rows,
        [
            "failure_type",
            "count",
            "percentage",
        ],
    )

    # ========================================================
    # SUMMARY TXT
    # ========================================================

    summary_path = (
        REPORTS_DIR
        / "PERSON_FALSE_NEGATIVES_V1_SUMMARY.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON FALSE "
            "NEGATIVES ANALYSIS V1\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            f"Dataset:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Modelo:\n{MODEL_PATH}\n\n"
        )

        f.write(
            f"Test:\n{TEST_IMAGES}\n\n"
        )

        f.write(
            "CONFIGURACION\n"
        )

        f.write(
            f"IoU threshold: {IOU_THRESHOLD}\n"
        )

        f.write(
            f"Confidence threshold: "
            f"{CONF_THRESHOLD}\n\n"
        )

        f.write(
            "RESULTADO GLOBAL PERSON\n"
        )

        f.write(
            f"GT:       {total_person_gt:,}\n"
        )

        f.write(
            f"TP:       {total_person_tp:,}\n"
        )

        f.write(
            f"FN:       {total_person_fn:,}\n"
        )

        f.write(
            f"Recall:   {global_recall:.4f}\n\n"
        )

        f.write(
            "FN POR AREA\n"
        )

        for row in area_rows:

            f.write(
                f"{row['area_bin']:10s} "
                f"GT={row['gt']:7,} "
                f"TP={row['tp']:7,} "
                f"FN={row['fn']:7,} "
                f"Recall={row['recall']:.4f}\n"
            )

        f.write("\n")

        f.write(
            "FN POR DENSIDAD\n"
        )

        for row in density_rows:

            f.write(
                f"{row['density_bin']:10s} "
                f"GT={row['gt']:7,} "
                f"TP={row['tp']:7,} "
                f"FN={row['fn']:7,} "
                f"Recall={row['recall']:.4f}\n"
            )

        f.write("\n")

        f.write(
            "FN POR BORDE\n"
        )

        for row in border_rows:

            f.write(
                f"{row['border_category']:12s} "
                f"GT={row['gt']:7,} "
                f"TP={row['tp']:7,} "
                f"FN={row['fn']:7,} "
                f"Recall={row['recall']:.4f}\n"
            )

        f.write("\n")

        f.write(
            "TIPO DE FN\n"
        )

        for row in confusion_rows:

            f.write(
                f"{row['failure_type']:28s} "
                f"{row['count']:7,} "
                f"({row['percentage']:.2f} %)\n"
            )

        f.write("\n")

        f.write(
            "ESTADISTICAS DE AREA\n"
        )

        f.write(
            f"TP area median: "
            f"{percentile(tp_areas, 0.50):.2f} px2\n"
        )

        f.write(
            f"TP area P90:    "
            f"{percentile(tp_areas, 0.90):.2f} px2\n"
        )

        f.write(
            f"FN area median: "
            f"{percentile(fn_areas, 0.50):.2f} px2\n"
        )

        f.write(
            f"FN area P90:    "
            f"{percentile(fn_areas, 0.90):.2f} px2\n"
        )

        f.write("\n")

        f.write(
            "INTERPRETACION\n"
        )

        f.write(
            "Este informe es diagnostico. "
            "No modifica imagenes, labels ni el dataset.\n"
        )

        f.write(
            "El objetivo es determinar si los FN de PERSON "
            "se concentran en objetos pequenos, escenas "
            "densas, zonas de borde o posibles confusiones "
            "PERSON/VEHICLE.\n"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        f"[OK] {REPORTS_DIR / 'person_fn_objects_v1.csv'}"
    )

    print(
        f"[OK] {REPORTS_DIR / 'person_fn_images_v1.csv'}"
    )

    print(
        f"[OK] {REPORTS_DIR / 'person_recall_by_area_v1.csv'}"
    )

    print(
        f"[OK] {REPORTS_DIR / 'person_recall_by_density_v1.csv'}"
    )

    print(
        f"[OK] {REPORTS_DIR / 'person_recall_by_border_v1.csv'}"
    )

    print(
        f"[OK] {REPORTS_DIR / 'person_class_confusion_v1.csv'}"
    )

    print(
        f"[OK] {summary_path}"
    )

    print()
    print(
        "IMPORTANTE: el dataset NO ha sido modificado."
    )
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()