from pathlib import Path
from collections import defaultdict
import csv
import math

from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

MODEL_PATH = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\training\runs\baseline_v1\weights\best.pt"
)

TEST_IMAGES = DATASET / "test_dev" / "images"

OUTPUT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\detection_failure_analysis"
    r"\person_failure_hotspots"
    r"\analyze_person_failure_hotspots_v1"
)

REPORTS = OUTPUT / "reports"

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
VEHICLE_CLASS = 1

# Mismo criterio de IoU utilizado en los análisis anteriores.
IOU_THRESHOLD = 0.50

# Umbrales de escala PERSON
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

# Densidad total de objetos de la imagen
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

def ensure_output():
    REPORTS.mkdir(parents=True, exist_ok=True)


def find_images():
    return sorted(
        [
            p
            for p in TEST_IMAGES.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def label_path(image_path):
    return (
        image_path.parent.parent
        / "labels"
        / f"{image_path.stem}.txt"
    )


def read_labels(path):
    objects = []

    if not path.exists():
        return objects

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

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

            if w <= 0 or h <= 0:
                continue

            objects.append(
                {
                    "class": cls,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                    "line": line_number,
                }
            )

    return objects


def normalized_to_xyxy(obj, width, height):
    xc = obj["xc"] * width
    yc = obj["yc"] * height
    bw = obj["w"] * width
    bh = obj["h"] * height

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return [x1, y1, x2, y2]


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


def get_scale(area):
    for name, low, high in SCALE_BINS:
        if low <= area < high:
            return name

    return ">2048"


def get_density(count):
    for name, low, high in DENSITY_BINS:
        if low <= count < high:
            return name

    return ">=500"


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return values[f] * (c - k) + values[c] * (k - f)


def safe_ratio(a, b):
    return a / b if b else 0.0


# ============================================================
# MATCHING
# ============================================================

def match_person_ground_truth(gt_persons, predictions):
    """
    Matching 1:1 entre PERSON GT y predicciones PERSON.

    Devuelve:
        matched_gt
        matched_predictions
    """

    candidates = []

    for gi, gt in enumerate(gt_persons):
        for pi, pred in enumerate(predictions):

            if pred["class"] != PERSON_CLASS:
                continue

            value = iou(gt["box"], pred["box"])

            if value >= IOU_THRESHOLD:
                candidates.append((value, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_predictions = set()

    for value, gi, pi in candidates:

        if gi in matched_gt:
            continue

        if pi in matched_predictions:
            continue

        matched_gt.add(gi)
        matched_predictions.add(pi)

    return matched_gt, matched_predictions


# ============================================================
# PROCESAMIENTO
# ============================================================

def main():

    print()
    print("=" * 72)
    print("# SAR YOLO26 - PERSON FAILURE HOTSPOTS ANALYSIS V1")
    print("=" * 72)
    print()

    print("Dataset:")
    print(DATASET)
    print()

    print("Modelo:")
    print(MODEL_PATH)
    print()

    print("Test:")
    print(TEST_IMAGES)
    print()

    print("Output:")
    print(OUTPUT)
    print()

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATASET}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(f"Test no encontrado: {TEST_IMAGES}")

    ensure_output()

    images = find_images()

    print(f"Imágenes encontradas: {len(images)}")
    print()

    print("Cargando modelo YOLO26s...")
    model = YOLO(str(MODEL_PATH))
    print("[OK] Modelo cargado.")
    print()

    image_records = []
    object_records = []

    total_person_gt = 0
    total_person_tp = 0
    total_person_fn = 0

    for index, image_path in enumerate(images, start=1):

        # ----------------------------------------------------
        # Imagen
        # ----------------------------------------------------

        try:
            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size

        except Exception:
            continue

        labels = read_labels(label_path(image_path))

        if not labels:
            continue

        gt_objects = []

        for obj in labels:
            box = normalized_to_xyxy(obj, width, height)

            gt_objects.append(
                {
                    "class": obj["class"],
                    "box": box,
                    "area": box_area(box),
                }
            )

        gt_persons = [
            x for x in gt_objects
            if x["class"] == PERSON_CLASS
        ]

        total_objects = len(gt_objects)

        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------

        result = model.predict(
            source=str(image_path),
            verbose=False,
            conf=0.001,
            iou=0.50,
        )[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):

                predictions.append(
                    {
                        "class": int(cls),
                        "box": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ],
                    }
                )

        # ----------------------------------------------------
        # Matching PERSON
        # ----------------------------------------------------

        matched_gt, matched_predictions = match_person_ground_truth(
            gt_persons,
            predictions,
        )

        person_gt_count = len(gt_persons)
        person_tp_count = len(matched_gt)
        person_fn_count = person_gt_count - person_tp_count

        total_person_gt += person_gt_count
        total_person_tp += person_tp_count
        total_person_fn += person_fn_count

        density_bin = get_density(total_objects)

        person_areas = [
            p["area"]
            for p in gt_persons
        ]

        small16 = sum(
            1 for area in person_areas
            if area < 16
        )

        small32 = sum(
            1 for area in person_areas
            if area < 32
        )

        small64 = sum(
            1 for area in person_areas
            if area < 64
        )

        # ----------------------------------------------------
        # Objetos PERSON individuales
        # ----------------------------------------------------

        for gi, person in enumerate(gt_persons):

            area = person["area"]
            scale_bin = get_scale(area)

            detected = gi in matched_gt

            x1, y1, x2, y2 = person["box"]

            partial_bbox = (
                x1 < 0
                or y1 < 0
                or x2 > width
                or y2 > height
            )

            border_margin = min(
                x1,
                y1,
                width - x2,
                height - y2,
            )

            near_border = (
                border_margin <= 10
            )

            object_records.append(
                {
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "image_area": width * height,
                    "total_objects": total_objects,
                    "density_bin": density_bin,
                    "person_area": round(area, 4),
                    "person_scale_bin": scale_bin,
                    "detected": int(detected),
                    "tp": int(detected),
                    "fn": int(not detected),
                    "partial_bbox": int(partial_bbox),
                    "near_border": int(near_border),
                }
            )

        # ----------------------------------------------------
        # Métricas de imagen
        # ----------------------------------------------------

        recall = safe_ratio(
            person_tp_count,
            person_gt_count,
        )

        fn_ratio = safe_ratio(
            person_fn_count,
            person_gt_count,
        )

        mean_person_area = (
            sum(person_areas) / len(person_areas)
            if person_areas
            else 0.0
        )

        median_person_area = (
            percentile(person_areas, 0.50)
            if person_areas
            else 0.0
        )

        p10_person_area = (
            percentile(person_areas, 0.10)
            if person_areas
            else 0.0
        )

        p90_person_area = (
            percentile(person_areas, 0.90)
            if person_areas
            else 0.0
        )

        image_records.append(
            {
                "image": image_path.name,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "image_area": width * height,
                "total_objects": total_objects,
                "density_bin": density_bin,
                "person_gt": person_gt_count,
                "person_tp": person_tp_count,
                "person_fn": person_fn_count,
                "person_recall": round(recall, 6),
                "person_fn_ratio": round(fn_ratio, 6),
                "person_small16": small16,
                "person_small32": small32,
                "person_small64": small64,
                "person_mean_area": round(mean_person_area, 4),
                "person_median_area": round(median_person_area, 4),
                "person_p10_area": round(p10_person_area, 4),
                "person_p90_area": round(p90_person_area, 4),
            }
        )

        if (
            index % 100 == 0
            or index == len(images)
        ):
            print(
                f"Analizadas: {index:,}/{len(images):,}"
            )

    print()

    # ========================================================
    # RANKING HOTSPOTS
    # ========================================================

    # Prioridad:
    #   1. mayor FN
    #   2. menor recall
    #   3. mayor cantidad de PERSON GT

    hotspots = sorted(
        image_records,
        key=lambda x: (
            x["person_fn"],
            -x["person_recall"],
            x["person_gt"],
        ),
        reverse=True,
    )

    # ========================================================
    # AGRUPACIÓN POR ESCALA
    # ========================================================

    scale_stats = defaultdict(
        lambda: {
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    for row in object_records:

        key = row["person_scale_bin"]

        scale_stats[key]["gt"] += 1
        scale_stats[key]["tp"] += row["tp"]
        scale_stats[key]["fn"] += row["fn"]

    # ========================================================
    # AGRUPACIÓN POR DENSIDAD
    # ========================================================

    density_stats = defaultdict(
        lambda: {
            "images": set(),
            "gt": 0,
            "tp": 0,
            "fn": 0,
        }
    )

    for row in object_records:

        key = row["density_bin"]

        density_stats[key]["images"].add(
            row["image"]
        )

        density_stats[key]["gt"] += 1
        density_stats[key]["tp"] += row["tp"]
        density_stats[key]["fn"] += row["fn"]

    # ========================================================
    # CSV 1 - HOTSPOTS
    # ========================================================

    hotspots_csv = REPORTS / (
        "person_failure_hotspots_v1.csv"
    )

    hotspot_fields = [
        "rank",
        "image",
        "image_path",
        "width",
        "height",
        "image_area",
        "total_objects",
        "density_bin",
        "person_gt",
        "person_tp",
        "person_fn",
        "person_recall",
        "person_fn_ratio",
        "person_small16",
        "person_small32",
        "person_small64",
        "person_mean_area",
        "person_median_area",
        "person_p10_area",
        "person_p90_area",
    ]

    with hotspots_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=hotspot_fields,
        )

        writer.writeheader()

        for rank, row in enumerate(
            hotspots,
            start=1,
        ):

            output = dict(row)
            output["rank"] = rank

            writer.writerow(output)

    print(f"[OK] {hotspots_csv}")

    # ========================================================
    # CSV 2 - POR IMAGEN
    # ========================================================

    by_image_csv = REPORTS / (
        "person_failure_hotspots_by_image_v1.csv"
    )

    with by_image_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=hotspot_fields[1:],
        )

        writer.writeheader()

        for row in image_records:
            writer.writerow(row)

    print(f"[OK] {by_image_csv}")

    # ========================================================
    # CSV 3 - POR ESCALA
    # ========================================================

    scale_csv = REPORTS / (
        "person_failure_hotspots_by_scale_v1.csv"
    )

    scale_fields = [
        "scale_bin",
        "gt",
        "tp",
        "fn",
        "recall",
        "fn_rate",
    ]

    with scale_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=scale_fields,
        )

        writer.writeheader()

        for name, _, _ in SCALE_BINS:

            stats = scale_stats[name]

            gt = stats["gt"]
            tp = stats["tp"]
            fn = stats["fn"]

            writer.writerow(
                {
                    "scale_bin": name,
                    "gt": gt,
                    "tp": tp,
                    "fn": fn,
                    "recall": round(
                        safe_ratio(tp, gt),
                        6,
                    ),
                    "fn_rate": round(
                        safe_ratio(fn, gt),
                        6,
                    ),
                }
            )

    print(f"[OK] {scale_csv}")

    # ========================================================
    # CSV 4 - POR DENSIDAD
    # ========================================================

    density_csv = REPORTS / (
        "person_failure_hotspots_by_density_v1.csv"
    )

    density_fields = [
        "density_bin",
        "images",
        "gt",
        "tp",
        "fn",
        "recall",
        "fn_rate",
    ]

    with density_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=density_fields,
        )

        writer.writeheader()

        for name, _, _ in DENSITY_BINS:

            stats = density_stats[name]

            gt = stats["gt"]
            tp = stats["tp"]
            fn = stats["fn"]

            writer.writerow(
                {
                    "density_bin": name,
                    "images": len(
                        stats["images"]
                    ),
                    "gt": gt,
                    "tp": tp,
                    "fn": fn,
                    "recall": round(
                        safe_ratio(tp, gt),
                        6,
                    ),
                    "fn_rate": round(
                        safe_ratio(fn, gt),
                        6,
                    ),
                }
            )

    print(f"[OK] {density_csv}")

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = REPORTS / (
        "PERSON_FAILURE_HOTSPOTS_V1_SUMMARY.txt"
    )

    global_recall = safe_ratio(
        total_person_tp,
        total_person_gt,
    )

    global_fn_rate = safe_ratio(
        total_person_fn,
        total_person_gt,
    )

    worst_images = hotspots[:20]

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - PERSON FAILURE HOTSPOTS ANALYSIS V1\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Modelo: {MODEL_PATH}\n")
        f.write(f"Test: {TEST_IMAGES}\n\n")

        f.write("RESULTADOS GLOBALES\n")
        f.write("-" * 72 + "\n")
        f.write(
            f"Imágenes:              {len(image_records):,}\n"
        )
        f.write(
            f"PERSON GT:             {total_person_gt:,}\n"
        )
        f.write(
            f"PERSON TP:             {total_person_tp:,}\n"
        )
        f.write(
            f"PERSON FN:             {total_person_fn:,}\n"
        )
        f.write(
            f"PERSON Recall:         {global_recall:.4f}\n"
        )
        f.write(
            f"PERSON FN rate:        {global_fn_rate:.4f}\n"
        )
        f.write("\n")

        f.write("TOP 20 FAILURE HOTSPOTS\n")
        f.write("-" * 72 + "\n")

        for rank, row in enumerate(
            worst_images,
            start=1,
        ):

            f.write(
                f"{rank:02d}. {row['image']}\n"
            )
            f.write(
                f"    PERSON GT:       {row['person_gt']}\n"
            )
            f.write(
                f"    PERSON TP:       {row['person_tp']}\n"
            )
            f.write(
                f"    PERSON FN:       {row['person_fn']}\n"
            )
            f.write(
                f"    Recall:          {row['person_recall']:.4f}\n"
            )
            f.write(
                f"    Objetos totales: {row['total_objects']}\n"
            )
            f.write(
                f"    Densidad:        {row['density_bin']}\n"
            )
            f.write(
                f"    PERSON <16:      {row['person_small16']}\n"
            )
            f.write(
                f"    PERSON <32:      {row['person_small32']}\n"
            )
            f.write(
                f"    PERSON <64:      {row['person_small64']}\n"
            )
            f.write(
                f"    Área mediana:    {row['person_median_area']:.2f}\n"
            )
            f.write(
                f"    Área P10:        {row['person_p10_area']:.2f}\n"
            )
            f.write(
                f"    Área P90:        {row['person_p90_area']:.2f}\n"
            )
            f.write(
                f"    Ruta:            {row['image_path']}\n"
            )
            f.write("\n")

        f.write("\n")
        f.write("INTERPRETACIÓN AUTOMÁTICA\n")
        f.write("-" * 72 + "\n\n")

        if global_recall < 0.40:
            f.write(
                "El Recall global de PERSON es bajo (<40%).\n"
            )

        if worst_images:

            worst = worst_images[0]

            f.write(
                "La imagen con mayor número absoluto de PERSON FN es:\n"
            )
            f.write(
                f"  {worst['image']}\n"
            )
            f.write(
                f"  FN={worst['person_fn']}\n"
            )
            f.write(
                f"  Recall={worst['person_recall']:.4f}\n"
            )
            f.write(
                f"  Densidad={worst['density_bin']}\n"
            )
            f.write(
                f"  PERSON <64={worst['person_small64']}\n"
            )

        f.write("\n")
        f.write(
            "Este análisis es exclusivamente diagnóstico.\n"
        )
        f.write(
            "No se modificó el dataset ni el modelo.\n"
        )

    print(f"[OK] {summary_path}")

    # ========================================================
    # CONSOLA
    # ========================================================

    print()
    print("=" * 72)
    print("# RESULTADO PERSON FAILURE HOTSPOTS V1")
    print("=" * 72)
    print()

    print(
        f"Imágenes:              {len(image_records):,}"
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

    print("TOP 20 FAILURE HOTSPOTS")
    print()

    for rank, row in enumerate(
        worst_images,
        start=1,
    ):

        print(
            f"{rank:02d}. {row['image']}"
        )

        print(
            f"    GT={row['person_gt']:5d} "
            f"TP={row['person_tp']:5d} "
            f"FN={row['person_fn']:5d} "
            f"Recall={row['person_recall']:.4f} "
            f"Objects={row['total_objects']:4d} "
            f"Density={row['density_bin']}"
        )

        print(
            f"    <16={row['person_small16']:4d} "
            f"<32={row['person_small32']:4d} "
            f"<64={row['person_small64']:4d} "
            f"MedianArea={row['person_median_area']:.1f}"
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