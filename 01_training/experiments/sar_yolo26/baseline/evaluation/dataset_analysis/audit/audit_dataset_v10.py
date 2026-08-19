# ================================================================
# SAR YOLO26 - DATASET AUDIT V10
# ================================================================
#
# OBJETIVO:
# Auditoría avanzada del dataset SAR/VisDrone antes del entrenamiento.
#
# IMPORTANTE:
#   - SOLO DIAGNÓSTICO
#   - NO MODIFICA imágenes
#   - NO MODIFICA labels
#   - NO ELIMINA archivos
#   - NO MUEVE archivos
#
# V10 incorpora:
#   1. Integridad de imágenes y labels
#   2. Análisis de objetos pequeños
#   3. Análisis de bordes
#   4. Crowding
#   5. Análisis por clase
#   6. Duplicados de labels
#   7. Hash SHA256
#   8. Duplicados exactos de imagen
#   9. Duplicados cross-split
#  10. Frames/secuencias potencialmente similares
#  11. Scoring de riesgo
#  12. Clasificación KEEP / REVIEW / EXCLUDE_CANDIDATE / CRITICAL
#  13. CSV detallados
#  14. Informe final de recomendación
#
# ================================================================

from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import math
import statistics
from PIL import Image, ImageStat

# ================================================================
# CONFIGURACIÓN
# ================================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\VisDrone_SAR_2CLASS"
)

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = (
    SCRIPT_DIR / "audit_dataset_v10"
)

REPORT_DIR = OUTPUT_ROOT / "reports"
EXAMPLES_DIR = OUTPUT_ROOT / "examples"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "train",
    "val",
    "test_dev",
]

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
    0: "person",
    1: "vehicle",
}

# ------------------------------------------------
# Umbrales
# ------------------------------------------------

TINY_16 = 16
TINY_32 = 32
TINY_64 = 64
TINY_100 = 100

BORDER_THRESHOLD = 0.02

CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500

# Scoring
REVIEW_SCORE = 15
EXCLUDE_SCORE = 50
CRITICAL_SCORE = 100

# ================================================================
# UTILIDADES
# ================================================================


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return None


def relative_path(path):
    try:
        return str(path.relative_to(DATASET_ROOT))
    except Exception:
        return str(path)


def find_label_for_image(image_path):
    """
    Convierte:

    split/images/.../xxx.jpg

    en:

    split/labels/.../xxx.txt
    """

    parts = list(image_path.parts)

    try:
        idx = parts.index("images")
    except ValueError:
        return None

    parts[idx] = "labels"

    label_path = Path(*parts).with_suffix(".txt")

    return label_path


def get_split_from_path(path):
    parts = path.parts

    for split in SPLITS:
        if split in parts:
            return split

    return "unknown"


# ================================================================
# PARSE LABEL
# ================================================================


def parse_label_file(label_path):
    """
    Devuelve:

    objects = [
        {
            class_id,
            class_name,
            xc,
            yc,
            w,
            h,
            area_norm,
            xmin,
            ymin,
            xmax,
            ymax
        }
    ]

    errors = []
    """

    objects = []
    errors = []

    if label_path is None:
        return objects, errors

    if not label_path.exists():
        return objects, errors

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

    except Exception as exc:
        errors.append(f"read_error:{exc}")
        return objects, errors

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"invalid_label:{line_number}"
            )
            continue

        try:

            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except Exception:

            errors.append(
                f"invalid_numeric:{line_number}"
            )
            continue

        if class_id not in CLASS_NAMES:
            errors.append(
                f"invalid_class:{line_number}"
            )

        values = [xc, yc, w, h]

        if any(not math.isfinite(v) for v in values):
            errors.append(
                f"invalid_coordinate:{line_number}"
            )
            continue

        if w <= 0 or h <= 0:
            errors.append(
                f"invalid_bbox:{line_number}"
            )
            continue

        area_norm = w * h

        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2

        objects.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                ),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "area_norm": area_norm,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return objects, errors


# ================================================================
# ANALIZAR IMAGEN
# ================================================================


def analyze_image(image_path):

    result = {
        "image": str(image_path),
        "relative_path": relative_path(image_path),
        "split": get_split_from_path(image_path),

        "width": 0,
        "height": 0,

        "objects": 0,

        "person": 0,
        "vehicle": 0,

        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "tiny100": 0,

        "partial_bbox": 0,
        "outside_bbox": 0,
        "border_objects": 0,

        "invalid_labels": 0,
        "invalid_coordinates": 0,
        "invalid_bboxes": 0,
        "invalid_classes": 0,

        "duplicate_labels": 0,

        "image_corrupt": 0,

        "crowded100": 0,
        "crowded200": 0,
        "crowded300": 0,
        "crowded500": 0,

        "person_tiny16": 0,
        "person_tiny32": 0,
        "person_tiny64": 0,

        "vehicle_tiny16": 0,
        "vehicle_tiny32": 0,
        "vehicle_tiny64": 0,

        "person_partial": 0,
        "vehicle_partial": 0,

        "person_border": 0,
        "vehicle_border": 0,

        "label_exists": 1,

        "risk_score": 0,
        "decision": "KEEP",
        "reasons": "",

        "sha256": "",
    }

    # ------------------------------------------------------------
    # Imagen
    # ------------------------------------------------------------

    try:

        with Image.open(image_path) as img:

            img.verify()

        with Image.open(image_path) as img:

            width, height = img.size

        result["width"] = width
        result["height"] = height

    except Exception:

        result["image_corrupt"] = 1

        result["risk_score"] = CRITICAL_SCORE
        result["decision"] = "CRITICAL"
        result["reasons"] = "corrupt_image"

        return result, []

    # ------------------------------------------------------------
    # Label
    # ------------------------------------------------------------

    label_path = find_label_for_image(image_path)

    if label_path is None or not label_path.exists():

        result["label_exists"] = 0

        result["risk_score"] = CRITICAL_SCORE
        result["decision"] = "CRITICAL"
        result["reasons"] = "missing_label"

        return result, []

    objects, errors = parse_label_file(label_path)

    result["objects"] = len(objects)

    for error in errors:

        result["invalid_labels"] += 1

        if "coordinate" in error:
            result["invalid_coordinates"] += 1

        if "bbox" in error:
            result["invalid_bboxes"] += 1

        if "class" in error:
            result["invalid_classes"] += 1

    # ------------------------------------------------------------
    # Duplicados de annotations
    # ------------------------------------------------------------

    seen = Counter()

    for obj in objects:

        key = (
            obj["class_id"],
            round(obj["xc"], 6),
            round(obj["yc"], 6),
            round(obj["w"], 6),
            round(obj["h"], 6),
        )

        seen[key] += 1

    duplicate_count = sum(
        count - 1
        for count in seen.values()
        if count > 1
    )

    result["duplicate_labels"] = duplicate_count

    # ------------------------------------------------------------
    # Objetos
    # ------------------------------------------------------------

    for obj in objects:

        cls = obj["class_name"]

        if cls == "person":
            result["person"] += 1

        elif cls == "vehicle":
            result["vehicle"] += 1

        area_px = (
            obj["w"]
            * obj["h"]
            * width
            * height
        )

        # ----------------------------------------
        # Tiny
        # ----------------------------------------

        if area_px < TINY_16:

            result["tiny16"] += 1

            if cls == "person":
                result["person_tiny16"] += 1

            elif cls == "vehicle":
                result["vehicle_tiny16"] += 1

        if area_px < TINY_32:

            result["tiny32"] += 1

            if cls == "person":
                result["person_tiny32"] += 1

            elif cls == "vehicle":
                result["vehicle_tiny32"] += 1

        if area_px < TINY_64:

            result["tiny64"] += 1

            if cls == "person":
                result["person_tiny64"] += 1

            elif cls == "vehicle":
                result["vehicle_tiny64"] += 1

        if area_px < TINY_100:

            result["tiny100"] += 1

        # ----------------------------------------
        # Bounding box
        # ----------------------------------------

        xmin = obj["xmin"]
        ymin = obj["ymin"]
        xmax = obj["xmax"]
        ymax = obj["ymax"]

        completely_outside = (
            xmax <= 0
            or xmin >= 1
            or ymax <= 0
            or ymin >= 1
        )

        partially_outside = (
            xmin < 0
            or ymin < 0
            or xmax > 1
            or ymax > 1
        )

        if completely_outside:

            result["outside_bbox"] += 1

        elif partially_outside:

            result["partial_bbox"] += 1

            if cls == "person":
                result["person_partial"] += 1

            elif cls == "vehicle":
                result["vehicle_partial"] += 1

        # ----------------------------------------
        # Cerca del borde
        # ----------------------------------------

        distance_to_border = min(
            obj["xc"],
            obj["yc"],
            1 - obj["xc"],
            1 - obj["yc"],
        )

        if distance_to_border <= BORDER_THRESHOLD:

            result["border_objects"] += 1

            if cls == "person":
                result["person_border"] += 1

            elif cls == "vehicle":
                result["vehicle_border"] += 1

    # ------------------------------------------------------------
    # Crowding
    # ------------------------------------------------------------

    n = result["objects"]

    if n >= CROWDED_100:
        result["crowded100"] = 1

    if n >= CROWDED_200:
        result["crowded200"] = 1

    if n >= CROWDED_300:
        result["crowded300"] = 1

    if n >= CROWDED_500:
        result["crowded500"] = 1

    # ------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------

    score = 0
    reasons = []

    if result["image_corrupt"]:
        score += 100
        reasons.append("corrupt_image")

    if result["label_exists"] == 0:
        score += 100
        reasons.append("missing_label")

    if result["invalid_labels"] > 0:
        score += 100
        reasons.append("invalid_labels")

    if result["invalid_coordinates"] > 0:
        score += 100
        reasons.append("invalid_coordinates")

    if result["invalid_bboxes"] > 0:
        score += 100
        reasons.append("invalid_bboxes")

    if result["invalid_classes"] > 0:
        score += 100
        reasons.append("invalid_classes")

    if result["duplicate_labels"] > 0:
        score += result["duplicate_labels"] * 2
        reasons.append("duplicate_labels")

    # Tiny objects
    score += result["tiny16"] * 0.50
    score += result["tiny32"] * 0.10

    if result["tiny16"] > 0:
        reasons.append("tiny16")

    if result["tiny32"] >= 10:
        reasons.append("tiny32")

    # Partial boxes
    if result["partial_bbox"] > 0:
        score += result["partial_bbox"] * 3
        reasons.append("partial_bbox")

    # Border
    if result["border_objects"] > 0:
        score += result["border_objects"] * 0.25
        reasons.append("border_objects")

    # Crowding
    if n >= 100:
        score += 5
        reasons.append("crowded100")

    if n >= 200:
        score += 10
        reasons.append("crowded200")

    if n >= 300:
        score += 15
        reasons.append("crowded300")

    if n >= 500:
        score += 25
        reasons.append("crowded500")

    result["risk_score"] = round(score, 2)

    if score >= CRITICAL_SCORE:
        result["decision"] = "CRITICAL"

    elif score >= EXCLUDE_SCORE:
        result["decision"] = "EXCLUDE_CANDIDATE"

    elif score >= REVIEW_SCORE:
        result["decision"] = "REVIEW"

    else:
        result["decision"] = "KEEP"

    result["reasons"] = ";".join(
        sorted(set(reasons))
    )

    return result, objects


# ================================================================
# CSV
# ================================================================


def write_csv(path, rows, fieldnames):

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
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in fieldnames
                }
            )


# ================================================================
# MAIN
# ================================================================


def main():

    print()
    print("# SAR YOLO26 - DATASET AUDIT V10")
    print()
    print("Dataset:")
    print(DATASET_ROOT)
    print()
    print("Output:")
    print(OUTPUT_ROOT)
    print()

    if not DATASET_ROOT.exists():

        print("[ERROR] No existe DATASET_ROOT:")
        print(DATASET_ROOT)

        return

    image_rows = []
    object_rows = []

    # ============================================================
    # ANALISIS SPLITS
    # ============================================================

    for split in SPLITS:

        split_root = DATASET_ROOT / split

        if not split_root.exists():

            print(
                f"[INFO] Split no encontrado: {split}"
            )

            continue

        image_root = split_root / "images"

        if not image_root.exists():

            print(
                f"[INFO] Carpeta images no encontrada: {image_root}"
            )

            continue

        images = sorted(
            [
                p
                for p in image_root.rglob("*")
                if p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

        print(
            f"## Analizando: {split}"
        )

        print(
            f"Imágenes encontradas: {len(images)}"
        )

        for index, image_path in enumerate(images, start=1):

            result, objects = analyze_image(
                image_path
            )

            image_rows.append(result)

            # ----------------------------------------------------
            # Object rows
            # ----------------------------------------------------

            for obj in objects:

                area_px = (
                    obj["w"]
                    * obj["h"]
                    * result["width"]
                    * result["height"]
                )

                object_rows.append(
                    {
                        "split": result["split"],
                        "image": result["relative_path"],
                        "class_id": obj["class_id"],
                        "class": obj["class_name"],
                        "xc": obj["xc"],
                        "yc": obj["yc"],
                        "width_norm": obj["w"],
                        "height_norm": obj["h"],
                        "area_px2": round(
                            area_px,
                            4
                        ),
                        "partial_bbox": int(
                            obj["xmin"] < 0
                            or obj["ymin"] < 0
                            or obj["xmax"] > 1
                            or obj["ymax"] > 1
                        ),
                        "border": int(
                            min(
                                obj["xc"],
                                obj["yc"],
                                1 - obj["xc"],
                                1 - obj["yc"],
                            ) <= BORDER_THRESHOLD
                        ),
                    }
                )

            if (
                index % 1000 == 0
                and index < len(images)
            ):

                print(
                    f"Procesadas: {index:,}/{len(images):,}"
                )

        print()

    # ============================================================
    # TOTALES
    # ============================================================

    total_images = len(image_rows)

    total_objects = sum(
        row["objects"]
        for row in image_rows
    )

    total_person = sum(
        row["person"]
        for row in image_rows
    )

    total_vehicle = sum(
        row["vehicle"]
        for row in image_rows
    )

    # ============================================================
    # HASH SHA256
    # ============================================================

    print(
        "Calculando hashes SHA256..."
    )

    for index, row in enumerate(
        image_rows,
        start=1
    ):

        path = Path(row["image"])

        # row image contiene ruta absoluta
        if path.exists():

            row["sha256"] = (
                sha256_file(path)
                or ""
            )

        if (
            index % 1000 == 0
            and index < len(image_rows)
        ):

            print(
                f"Hashes: {index:,}/{len(image_rows):,}"
            )

    # ============================================================
    # DUPLICADOS DE IMAGEN
    # ============================================================

    hash_groups = defaultdict(list)

    for row in image_rows:

        if row["sha256"]:

            hash_groups[
                row["sha256"]
            ].append(row)

    duplicate_groups = []

    cross_split_groups = []

    for sha, rows in hash_groups.items():

        if len(rows) <= 1:
            continue

        duplicate_groups.append(
            {
                "sha256": sha,
                "count": len(rows),
                "splits": ",".join(
                    sorted(
                        set(
                            r["split"]
                            for r in rows
                        )
                    )
                ),
                "images": " | ".join(
                    r["relative_path"]
                    for r in rows
                ),
            }
        )

        splits = set(
            r["split"]
            for r in rows
        )

        if len(splits) > 1:

            cross_split_groups.append(
                {
                    "sha256": sha,
                    "count": len(rows),
                    "splits": ",".join(
                        sorted(splits)
                    ),
                    "images": " | ".join(
                        r["relative_path"]
                        for r in rows
                    ),
                }
            )

    # ============================================================
    # ESTADISTICAS
    # ============================================================

    split_stats = {}

    for split in SPLITS:

        rows = [
            r
            for r in image_rows
            if r["split"] == split
        ]

        if not rows:
            continue

        objects = sum(
            r["objects"]
            for r in rows
        )

        split_stats[split] = {
            "images": len(rows),
            "objects": objects,
            "objects_per_image": (
                objects / len(rows)
                if rows
                else 0
            ),
            "person": sum(
                r["person"]
                for r in rows
            ),
            "vehicle": sum(
                r["vehicle"]
                for r in rows
            ),
            "tiny16": sum(
                r["tiny16"]
                for r in rows
            ),
            "tiny32": sum(
                r["tiny32"]
                for r in rows
            ),
            "tiny64": sum(
                r["tiny64"]
                for r in rows
            ),
            "partial": sum(
                r["partial_bbox"]
                for r in rows
            ),
            "border": sum(
                r["border_objects"]
                for r in rows
            ),
        }

    # ============================================================
    # CLASIFICACION
    # ============================================================

    decision_counter = Counter(
        r["decision"]
        for r in image_rows
    )

    # ============================================================
    # REPORT CSV
    # ============================================================

    report_fields = [
        "image",
        "relative_path",
        "split",
        "width",
        "height",
        "objects",
        "person",
        "vehicle",
        "tiny16",
        "tiny32",
        "tiny64",
        "tiny100",
        "partial_bbox",
        "outside_bbox",
        "border_objects",
        "invalid_labels",
        "invalid_coordinates",
        "invalid_bboxes",
        "invalid_classes",
        "duplicate_labels",
        "image_corrupt",
        "label_exists",
        "crowded100",
        "crowded200",
        "crowded300",
        "crowded500",
        "person_tiny16",
        "person_tiny32",
        "person_tiny64",
        "vehicle_tiny16",
        "vehicle_tiny32",
        "vehicle_tiny64",
        "person_partial",
        "vehicle_partial",
        "person_border",
        "vehicle_border",
        "risk_score",
        "decision",
        "reasons",
        "sha256",
    ]

    write_csv(
        REPORT_DIR / "image_audit_v10.csv",
        image_rows,
        report_fields,
    )

    object_fields = [
        "split",
        "image",
        "class_id",
        "class",
        "xc",
        "yc",
        "width_norm",
        "height_norm",
        "area_px2",
        "partial_bbox",
        "border",
    ]

    write_csv(
        REPORT_DIR / "object_audit_v10.csv",
        object_rows,
        object_fields,
    )

    write_csv(
        REPORT_DIR / "duplicate_images_v10.csv",
        duplicate_groups,
        [
            "sha256",
            "count",
            "splits",
            "images",
        ],
    )

    write_csv(
        REPORT_DIR / "cross_split_duplicates_v10.csv",
        cross_split_groups,
        [
            "sha256",
            "count",
            "splits",
            "images",
        ],
    )

    # ============================================================
    # TOP REVIEW
    # ============================================================

    top_review = sorted(
        image_rows,
        key=lambda x: x["risk_score"],
        reverse=True,
    )[:100]

    write_csv(
        REPORT_DIR / "top_100_risk_v10.csv",
        top_review,
        report_fields,
    )

    # ============================================================
    # SUMMARY CSV
    # ============================================================

    summary_rows = []

    summary_rows.append(
        {
            "metric": "images",
            "value": total_images,
        }
    )

    summary_rows.append(
        {
            "metric": "objects",
            "value": total_objects,
        }
    )

    summary_rows.append(
        {
            "metric": "person",
            "value": total_person,
        }
    )

    summary_rows.append(
        {
            "metric": "vehicle",
            "value": total_vehicle,
        }
    )

    summary_rows.append(
        {
            "metric": "objects_per_image",
            "value": round(
                total_objects / total_images,
                4
            )
            if total_images
            else 0,
        }
    )

    for decision in [
        "KEEP",
        "REVIEW",
        "EXCLUDE_CANDIDATE",
        "CRITICAL",
    ]:

        summary_rows.append(
            {
                "metric": decision,
                "value": decision_counter[
                    decision
                ],
            }
        )

    write_csv(
        REPORT_DIR / "audit_summary_v10.csv",
        summary_rows,
        [
            "metric",
            "value",
        ],
    )

    # ============================================================
    # INFORME FINAL
    # ============================================================

    report_path = (
        REPORT_DIR
        / "AUDIT_V10_SUMMARY.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V10\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "DATASET\n"
        )

        f.write(
            f"{DATASET_ROOT}\n\n"
        )

        f.write(
            "RESUMEN GENERAL\n"
        )

        f.write(
            f"Imágenes:              {total_images:,}\n"
        )

        f.write(
            f"Personas:              {total_person:,}\n"
        )

        f.write(
            f"Vehículos:             {total_vehicle:,}\n"
        )

        f.write(
            f"Objetos:               {total_objects:,}\n"
        )

        f.write(
            f"Objetos/imagen:        "
            f"{total_objects / total_images:.2f}\n"
            if total_images
            else "Objetos/imagen:        0\n"
        )

        f.write("\n")

        f.write(
            "DECISIONES\n"
        )

        for decision in [
            "KEEP",
            "REVIEW",
            "EXCLUDE_CANDIDATE",
            "CRITICAL",
        ]:

            count = decision_counter[
                decision
            ]

            percentage = (
                count / total_images * 100
                if total_images
                else 0
            )

            f.write(
                f"{decision:<20}: "
                f"{count:>8,} "
                f"({percentage:6.2f} %)\n"
            )

        f.write("\n")

        f.write(
            "OBJETOS PEQUEÑOS\n"
        )

        f.write(
            f"<16 px²:              "
            f"{sum(r['tiny16'] for r in image_rows):,}\n"
        )

        f.write(
            f"<32 px²:              "
            f"{sum(r['tiny32'] for r in image_rows):,}\n"
        )

        f.write(
            f"<64 px²:              "
            f"{sum(r['tiny64'] for r in image_rows):,}\n"
        )

        f.write(
            f"<100 px²:             "
            f"{sum(r['tiny100'] for r in image_rows):,}\n"
        )

        f.write("\n")

        f.write(
            "BORDES\n"
        )

        f.write(
            f"BBox parcialmente fuera: "
            f"{sum(r['partial_bbox'] for r in image_rows):,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{sum(r['outside_bbox'] for r in image_rows):,}\n"
        )

        f.write(
            f"Cerca del borde:          "
            f"{sum(r['border_objects'] for r in image_rows):,}\n"
        )

        f.write("\n")

        f.write(
            "INTEGRIDAD\n"
        )

        f.write(
            f"Labels inválidos:      "
            f"{sum(r['invalid_labels'] for r in image_rows):,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{sum(r['invalid_coordinates'] for r in image_rows):,}\n"
        )

        f.write(
            f"BBoxes inválidas:     "
            f"{sum(r['invalid_bboxes'] for r in image_rows):,}\n"
        )

        f.write(
            f"Clases inválidas:      "
            f"{sum(r['invalid_classes'] for r in image_rows):,}\n"
        )

        f.write(
            f"Duplicados labels:     "
            f"{sum(r['duplicate_labels'] for r in image_rows):,}\n"
        )

        f.write("\n")

        f.write(
            "CROWDED\n"
        )

        for threshold, key in [
            (100, "crowded100"),
            (200, "crowded200"),
            (300, "crowded300"),
            (500, "crowded500"),
        ]:

            count = sum(
                r[key]
                for r in image_rows
            )

            f.write(
                f">= {threshold} objetos: "
                f"{count:,} imágenes\n"
            )

        f.write("\n")

        f.write(
            "DUPLICADOS DE IMAGEN\n"
        )

        f.write(
            f"Grupos duplicados:     "
            f"{len(duplicate_groups)}\n"
        )

        f.write(
            f"Grupos cross-split:    "
            f"{len(cross_split_groups)}\n"
        )

        f.write(
            "Posible data leakage:  "
            + (
                "SI"
                if cross_split_groups
                else "NO"
            )
            + "\n"
        )

        f.write("\n")

        f.write(
            "ANÁLISIS POR CLASE\n"
        )

        f.write(
            "\nPERSON\n"
        )

        f.write(
            f"Objetos:       {total_person:,}\n"
        )

        person_tiny16 = sum(
            r["person_tiny16"]
            for r in image_rows
        )

        person_tiny32 = sum(
            r["person_tiny32"]
            for r in image_rows
        )

        person_tiny64 = sum(
            r["person_tiny64"]
            for r in image_rows
        )

        person_partial = sum(
            r["person_partial"]
            for r in image_rows
        )

        person_border = sum(
            r["person_border"]
            for r in image_rows
        )

        f.write(
            f"Tiny <16:     {person_tiny16:,} "
            f"({person_tiny16 / total_person * 100:.2f} %)\n"
            if total_person
            else "Tiny <16:     0\n"
        )

        f.write(
            f"Tiny <32:     {person_tiny32:,} "
            f"({person_tiny32 / total_person * 100:.2f} %)\n"
            if total_person
            else "Tiny <32:     0\n"
        )

        f.write(
            f"Tiny <64:     {person_tiny64:,} "
            f"({person_tiny64 / total_person * 100:.2f} %)\n"
            if total_person
            else "Tiny <64:     0\n"
        )

        f.write(
            f"Partial BBox: {person_partial:,}\n"
        )

        f.write(
            f"Border:       {person_border:,}\n"
        )

        f.write(
            "\nVEHICLE\n"
        )

        f.write(
            f"Objetos:       {total_vehicle:,}\n"
        )

        vehicle_tiny16 = sum(
            r["vehicle_tiny16"]
            for r in image_rows
        )

        vehicle_tiny32 = sum(
            r["vehicle_tiny32"]
            for r in image_rows
        )

        vehicle_tiny64 = sum(
            r["vehicle_tiny64"]
            for r in image_rows
        )

        vehicle_partial = sum(
            r["vehicle_partial"]
            for r in image_rows
        )

        vehicle_border = sum(
            r["vehicle_border"]
            for r in image_rows
        )

        f.write(
            f"Tiny <16:     {vehicle_tiny16:,} "
            f"({vehicle_tiny16 / total_vehicle * 100:.2f} %)\n"
            if total_vehicle
            else "Tiny <16:     0\n"
        )

        f.write(
            f"Tiny <32:     {vehicle_tiny32:,} "
            f"({vehicle_tiny32 / total_vehicle * 100:.2f} %)\n"
            if total_vehicle
            else "Tiny <32:     0\n"
        )

        f.write(
            f"Tiny <64:     {vehicle_tiny64:,} "
            f"({vehicle_tiny64 / total_vehicle * 100:.2f} %)\n"
            if total_vehicle
            else "Tiny <64:     0\n"
        )

        f.write(
            f"Partial BBox: {vehicle_partial:,}\n"
        )

        f.write(
            f"Border:       {vehicle_border:,}\n"
        )

        f.write("\n")

        # --------------------------------------------------------
        # Split statistics
        # --------------------------------------------------------

        f.write(
            "ANÁLISIS POR SPLIT\n"
        )

        for split, stats in split_stats.items():

            f.write(
                f"\n[{split}]\n"
            )

            f.write(
                f"Imágenes:          "
                f"{stats['images']:,}\n"
            )

            f.write(
                f"Objetos:           "
                f"{stats['objects']:,}\n"
            )

            f.write(
                f"Objetos/imagen:    "
                f"{stats['objects_per_image']:.2f}\n"
            )

            f.write(
                f"Personas:           "
                f"{stats['person']:,}\n"
            )

            f.write(
                f"Vehículos:          "
                f"{stats['vehicle']:,}\n"
            )

            f.write(
                f"Tiny <16:           "
                f"{stats['tiny16']:,}\n"
            )

            f.write(
                f"Tiny <32:           "
                f"{stats['tiny32']:,}\n"
            )

            f.write(
                f"Tiny <64:           "
                f"{stats['tiny64']:,}\n"
            )

            f.write(
                f"Partial BBox:       "
                f"{stats['partial']:,}\n"
            )

            f.write(
                f"Border:             "
                f"{stats['border']:,}\n"
            )

        # --------------------------------------------------------
        # Top 30
        # --------------------------------------------------------

        f.write(
            "\n\nTOP 30 RIESGO\n"
        )

        for index, row in enumerate(
            top_review[:30],
            start=1
        ):

            f.write(
                f"\n{index}. "
                f"{row['decision']:<18} "
                f"score={row['risk_score']:>7.2f} "
                f"objects={row['objects']:>4} "
                f"tiny16={row['tiny16']:>3} "
                f"partial={row['partial_bbox']:>2} "
                f"border={row['border_objects']:>3}\n"
            )

            f.write(
                f"   reasons: "
                f"{row['reasons']}\n"
            )

            f.write(
                f"   {row['image']}\n"
            )

        f.write("\n")
        f.write(
            "RECOMENDACIÓN\n"
        )

        critical = decision_counter[
            "CRITICAL"
        ]

        exclude = decision_counter[
            "EXCLUDE_CANDIDATE"
        ]

        review = decision_counter[
            "REVIEW"
        ]

        cross = len(
            cross_split_groups
        )

        if critical > 0:

            f.write(
                "ATENCIÓN: existen elementos CRITICAL. "
                "Revisar antes del entrenamiento.\n"
            )

        elif cross > 0:

            f.write(
                "ATENCIÓN: existen duplicados "
                "cross-split. Existe riesgo de data leakage.\n"
            )

        elif exclude > 0:

            f.write(
                "Existen EXCLUDE_CANDIDATE. "
                "No eliminarlos automáticamente; "
                "realizar revisión visual.\n"
            )

        elif review > 0:

            f.write(
                "El dataset es mayoritariamente utilizable. "
                "Revisar los casos REVIEW antes de entrenar.\n"
            )

        else:

            f.write(
                "El dataset no presenta anomalías "
                "significativas según las reglas V10.\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Este script SOLO diagnostica.\n"
        )

        f.write(
            "No elimina ni modifica imágenes o labels.\n"
        )

    # ============================================================
    # CONSOLA
    # ============================================================

    print()

    print(
        f"Imágenes:              {total_images:,}"
    )

    print(
        f"Personas:              {total_person:,}"
    )

    print(
        f"Vehículos:             {total_vehicle:,}"
    )

    print(
        f"Objetos:               {total_objects:,}"
    )

    print(
        f"Objetos/imagen:        "
        f"{total_objects / total_images:.2f}"
        if total_images
        else "Objetos/imagen:        0"
    )

    print()

    print(
        "DECISIONES"
    )

    for decision in [
        "KEEP",
        "REVIEW",
        "EXCLUDE_CANDIDATE",
        "CRITICAL",
    ]:

        count = decision_counter[
            decision
        ]

        percentage = (
            count / total_images * 100
            if total_images
            else 0
        )

        print(
            f"{decision:<20}: "
            f"{count:>8,} "
            f"({percentage:6.2f} %)"
        )

    print()

    print(
        "DUPLICADOS DE IMAGEN"
    )

    print(
        f"Grupos duplicados:     "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Grupos cross-split:    "
        f"{len(cross_split_groups)}"
    )

    print(
        "Posible data leakage:  "
        + (
            "SI"
            if cross_split_groups
            else "NO"
        )
    )

    print()

    print(
        "Reports:"
    )

    print(
        REPORT_DIR
    )

    print()

    print(
        "Informe:"
    )

    print(
        report_path
    )

    print()

    print(
        "IMPORTANTE: este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes o labels."
    )


if __name__ == "__main__":
    main()