# ============================================================
# SAR YOLO26 - DATASET AUDIT V11
# ============================================================
#
# Objetivo:
#   Auditoría avanzada orientada a entrenamiento YOLO26.
#
# V11 añade:
#   - Distribución por clase y split
#   - Balance PERSON / VEHICLE
#   - Tamaño de objetos por clase
#   - Crowding
#   - Tiny-object concentration
#   - BBoxes parcialmente fuera por clase
#   - Ranking de imágenes sospechosas
#   - Comparación train / val / test_dev
#   - Recomendación automática
#
# IMPORTANTE:
#   Este script SOLO DIAGNOSTICA.
#   NO elimina ni modifica imágenes ni labels.
#
# ============================================================

from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import math
import statistics


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\VisDrone_SAR_2CLASS"
)

WORK_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

OUTPUT_ROOT = (
    WORK_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "audit"
    / "audit_dataset_v11"
)

REPORTS_DIR = OUTPUT_ROOT / "reports"
EXAMPLES_DIR = OUTPUT_ROOT / "examples"

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
    ".webp",
}

# Clases del dataset
CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}

# Umbrales
TINY16 = 16.0
TINY32 = 32.0
TINY64 = 64.0

BORDER_MARGIN = 0.02

CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500

# Concentración de tiny objects
TINY_RATIO_REVIEW = 0.25
TINY_RATIO_EXCLUDE = 0.50

# BBoxes parciales
PARTIAL_RATIO_REVIEW = 0.10
PARTIAL_RATIO_EXCLUDE = 0.30


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def image_files(split):
    image_dir = DATASET_ROOT / split / "images"

    if not image_dir.exists():
        return []

    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path):
    split_dir = DATASET_ROOT / image_path.parts[
        image_path.parts.index(DATASET_ROOT.name) + 1
    ]

    relative = image_path.relative_to(split_dir / "images")

    return split_dir / "labels" / relative.with_suffix(".txt")


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def bbox_area_pixels(parts, width, height):
    if width <= 0 or height <= 0:
        return 0.0

    _, _, bw, bh = parts

    return (
        max(0.0, bw * width)
        * max(0.0, bh * height)
    )


def parse_image_size(image_path):
    """
    Intenta obtener dimensiones mediante PIL.
    Si PIL no está disponible, devuelve None.
    """

    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return im.width, im.height

    except Exception:
        return None


# ============================================================
# LECTURA LABELS
# ============================================================

def parse_label_file(label_path):

    result = {
        "objects": [],
        "invalid_lines": 0,
        "invalid_coordinates": 0,
        "invalid_boxes": 0,
        "invalid_classes": 0,
        "duplicate_lines": 0,
    }

    if not label_path.exists():
        return result

    seen = set()

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception:
        return result

    for raw in lines:

        line = raw.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            result["invalid_lines"] += 1
            continue

        try:
            cls = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            result["invalid_lines"] += 1
            continue

        key = tuple(parts)

        if key in seen:
            result["duplicate_lines"] += 1

        seen.add(key)

        if cls not in CLASS_NAMES:
            result["invalid_classes"] += 1

        coords = [xc, yc, bw, bh]

        if any(not math.isfinite(x) for x in coords):
            result["invalid_coordinates"] += 1
            continue

        if any(x < 0 or x > 1 for x in [xc, yc]):
            result["invalid_coordinates"] += 1

        if bw <= 0 or bh <= 0:
            result["invalid_boxes"] += 1

        result["objects"].append(
            {
                "class": cls,
                "xc": xc,
                "yc": yc,
                "bw": bw,
                "bh": bh,
            }
        )

    return result


# ============================================================
# ANÁLISIS DE UNA IMAGEN
# ============================================================

def analyze_image(image_path, split):

    label_path = (
        DATASET_ROOT
        / split
        / "labels"
        / image_path.relative_to(
            DATASET_ROOT / split / "images"
        ).with_suffix(".txt")
    )

    parsed = parse_label_file(label_path)

    dimensions = parse_image_size(image_path)

    if dimensions:
        width, height = dimensions
    else:
        width, height = 1, 1

    objects = parsed["objects"]

    class_counts = Counter()
    class_tiny16 = Counter()
    class_tiny32 = Counter()
    class_tiny64 = Counter()
    class_partial = Counter()
    class_border = Counter()

    tiny16 = 0
    tiny32 = 0
    tiny64 = 0

    partial = 0
    border = 0
    outside = 0

    areas = []

    for obj in objects:

        cls = obj["class"]

        if cls in CLASS_NAMES:
            class_counts[cls] += 1

        area = bbox_area_pixels(
            (
                obj["xc"],
                obj["yc"],
                obj["bw"],
                obj["bh"],
            ),
            width,
            height,
        )

        areas.append(area)

        if area < TINY16:
            tiny16 += 1
            class_tiny16[cls] += 1

        if area < TINY32:
            tiny32 += 1
            class_tiny32[cls] += 1

        if area < TINY64:
            tiny64 += 1
            class_tiny64[cls] += 1

        x1 = obj["xc"] - obj["bw"] / 2
        y1 = obj["yc"] - obj["bh"] / 2
        x2 = obj["xc"] + obj["bw"] / 2
        y2 = obj["yc"] + obj["bh"] / 2

        is_outside = (
            x2 < 0
            or x1 > 1
            or y2 < 0
            or y1 > 1
        )

        is_partial = (
            not is_outside
            and (
                x1 < 0
                or y1 < 0
                or x2 > 1
                or y2 > 1
            )
        )

        if is_outside:
            outside += 1

        elif is_partial:
            partial += 1
            class_partial[cls] += 1

        margin = BORDER_MARGIN

        near_border = (
            obj["xc"] - obj["bw"] / 2 < margin
            or obj["yc"] - obj["bh"] / 2 < margin
            or obj["xc"] + obj["bw"] / 2 > 1 - margin
            or obj["yc"] + obj["bh"] / 2 > 1 - margin
        )

        if near_border:
            border += 1
            class_border[cls] += 1

    total = len(objects)

    tiny_ratio = tiny16 / total if total else 0
    partial_ratio = partial / total if total else 0

    return {
        "split": split,
        "image": str(image_path),
        "label": str(label_path),

        "objects": total,

        "person": class_counts.get(0, 0),
        "vehicle": class_counts.get(1, 0),

        "tiny16": tiny16,
        "tiny32": tiny32,
        "tiny64": tiny64,

        "tiny16_ratio": tiny16 / total if total else 0,
        "tiny32_ratio": tiny32 / total if total else 0,
        "tiny64_ratio": tiny64 / total if total else 0,

        "partial": partial,
        "outside": outside,
        "border": border,

        "partial_ratio": partial_ratio,
        "border_ratio": border / total if total else 0,

        "invalid_lines": parsed["invalid_lines"],
        "invalid_coordinates": parsed["invalid_coordinates"],
        "invalid_boxes": parsed["invalid_boxes"],
        "invalid_classes": parsed["invalid_classes"],
        "duplicate_lines": parsed["duplicate_lines"],

        "person_tiny16": class_tiny16.get(0, 0),
        "person_tiny32": class_tiny32.get(0, 0),
        "person_tiny64": class_tiny64.get(0, 0),

        "vehicle_tiny16": class_tiny16.get(1, 0),
        "vehicle_tiny32": class_tiny32.get(1, 0),
        "vehicle_tiny64": class_tiny64.get(1, 0),

        "person_partial": class_partial.get(0, 0),
        "vehicle_partial": class_partial.get(1, 0),

        "person_border": class_border.get(0, 0),
        "vehicle_border": class_border.get(1, 0),

        "area_min": min(areas) if areas else 0,
        "area_max": max(areas) if areas else 0,
        "area_mean": statistics.mean(areas) if areas else 0,
        "area_median": statistics.median(areas) if areas else 0,
    }


# ============================================================
# SCORING
# ============================================================

def classify_image(row):

    score = 0.0
    reasons = []

    objects = row["objects"]

    # --------------------------------------------------------
    # Tiny objects
    # --------------------------------------------------------

    if row["tiny16"] > 0:
        score += row["tiny16"] * 0.75
        reasons.append(f"tiny16={row['tiny16']}")

    if row["tiny32"] >= 20:
        score += 5
        reasons.append(f"tiny32={row['tiny32']}")

    # --------------------------------------------------------
    # Concentración tiny
    # --------------------------------------------------------

    if row["tiny16_ratio"] >= TINY_RATIO_REVIEW:
        score += 15
        reasons.append("high_tiny_ratio")

    if row["tiny16_ratio"] >= TINY_RATIO_EXCLUDE:
        score += 35
        reasons.append("extreme_tiny_ratio")

    # --------------------------------------------------------
    # BBoxes parciales
    # --------------------------------------------------------

    if row["partial"] > 0:
        score += row["partial"] * 3
        reasons.append("partial_bbox")

    if row["partial_ratio"] >= PARTIAL_RATIO_REVIEW:
        score += 15
        reasons.append("high_partial_ratio")

    if row["partial_ratio"] >= PARTIAL_RATIO_EXCLUDE:
        score += 30
        reasons.append("extreme_partial_ratio")

    # --------------------------------------------------------
    # Borde
    # --------------------------------------------------------

    if row["border"] > 0:
        score += min(row["border"] * 0.25, 20)
        reasons.append("border_objects")

    # --------------------------------------------------------
    # Crowded
    # --------------------------------------------------------

    if objects >= 100:
        score += 3
        reasons.append("crowded100")

    if objects >= 200:
        score += 5
        reasons.append("crowded200")

    if objects >= 300:
        score += 8
        reasons.append("crowded300")

    if objects >= 500:
        score += 12
        reasons.append("crowded500")

    # --------------------------------------------------------
    # Integridad
    # --------------------------------------------------------

    integrity_errors = (
        row["invalid_lines"]
        + row["invalid_coordinates"]
        + row["invalid_boxes"]
        + row["invalid_classes"]
    )

    if integrity_errors > 0:
        score += 100
        reasons.append("integrity_error")

    if row["duplicate_lines"] > 0:
        score += 20
        reasons.append("duplicate_labels")

    # --------------------------------------------------------
    # Clasificación
    # --------------------------------------------------------

    if integrity_errors > 0:
        decision = "CRITICAL"

    elif score >= 100:
        decision = "EXCLUDE_CANDIDATE"

    elif score >= 50:
        decision = "REVIEW"

    else:
        decision = "KEEP"

    return decision, round(score, 2), ";".join(reasons)


# ============================================================
# ANÁLISIS DE SPLITS
# ============================================================

def analyze_split(split):

    print(f"\n## Analizando: {split}")

    images = image_files(split)

    if not images:
        print(f"[INFO] Split no encontrado: {split}")
        return []

    print(f"Imágenes encontradas: {len(images)}")

    rows = []

    for i, image_path in enumerate(images, 1):

        row = analyze_image(image_path, split)

        decision, score, reasons = classify_image(row)

        row["decision"] = decision
        row["score"] = score
        row["reasons"] = reasons

        rows.append(row)

        if i % 1000 == 0:
            print(f"Procesadas: {i:,}/{len(images):,}")

    return rows


# ============================================================
# CSV
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    fields = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# RESUMEN GLOBAL
# ============================================================

def global_summary(rows):

    decisions = Counter(
        row["decision"]
        for row in rows
    )

    persons = sum(
        row["person"]
        for row in rows
    )

    vehicles = sum(
        row["vehicle"]
        for row in rows
    )

    objects = sum(
        row["objects"]
        for row in rows
    )

    tiny16 = sum(
        row["tiny16"]
        for row in rows
    )

    tiny32 = sum(
        row["tiny32"]
        for row in rows
    )

    tiny64 = sum(
        row["tiny64"]
        for row in rows
    )

    partial = sum(
        row["partial"]
        for row in rows
    )

    outside = sum(
        row["outside"]
        for row in rows
    )

    border = sum(
        row["border"]
        for row in rows
    )

    duplicates = sum(
        row["duplicate_lines"]
        for row in rows
    )

    return {
        "images": len(rows),
        "persons": persons,
        "vehicles": vehicles,
        "objects": objects,

        "objects_per_image":
            objects / len(rows)
            if rows else 0,

        "keep": decisions["KEEP"],
        "review": decisions["REVIEW"],
        "exclude": decisions["EXCLUDE_CANDIDATE"],
        "critical": decisions["CRITICAL"],

        "tiny16": tiny16,
        "tiny32": tiny32,
        "tiny64": tiny64,

        "partial": partial,
        "outside": outside,
        "border": border,

        "duplicates": duplicates,
    }


# ============================================================
# SPLIT SUMMARY
# ============================================================

def split_summary(rows):

    result = []

    by_split = defaultdict(list)

    for row in rows:
        by_split[row["split"]].append(row)

    for split, split_rows in by_split.items():

        s = global_summary(split_rows)

        s["split"] = split

        result.append(s)

    return result


# ============================================================
# GENERAR INFORME
# ============================================================

def write_report(rows):

    summary = global_summary(rows)
    split_stats = split_summary(rows)

    report = REPORTS_DIR / "AUDIT_V11_SUMMARY.txt"

    with open(
        report,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# SAR YOLO26 - DATASET AUDIT V11\n\n"
        )

        f.write(
            "## DATASET\n\n"
        )

        f.write(
            f"{DATASET_ROOT}\n\n"
        )

        f.write(
            "## RESUMEN GLOBAL\n\n"
        )

        f.write(
            f"Imágenes:              {summary['images']:,}\n"
        )

        f.write(
            f"Personas:              {summary['persons']:,}\n"
        )

        f.write(
            f"Vehículos:             {summary['vehicles']:,}\n"
        )

        f.write(
            f"Objetos:               {summary['objects']:,}\n"
        )

        f.write(
            f"Objetos/imagen:        "
            f"{summary['objects_per_image']:.2f}\n\n"
        )

        f.write("## DECISIONES\n\n")

        total = summary["images"]

        for key, label in [
            ("keep", "KEEP"),
            ("review", "REVIEW"),
            ("exclude", "EXCLUDE_CANDIDATE"),
            ("critical", "CRITICAL"),
        ]:

            value = summary[key]

            pct = (
                value / total * 100
                if total
                else 0
            )

            f.write(
                f"{label:20}: "
                f"{value:6,} "
                f"({pct:6.2f} %)\n"
            )

        f.write("\n## OBJETOS PEQUEÑOS\n\n")

        f.write(
            f"<16 px²:              "
            f"{summary['tiny16']:,}\n"
        )

        f.write(
            f"<32 px²:              "
            f"{summary['tiny32']:,}\n"
        )

        f.write(
            f"<64 px²:              "
            f"{summary['tiny64']:,}\n"
        )

        f.write("\n## BORDES\n\n")

        f.write(
            f"BBox parcialmente fuera: "
            f"{summary['partial']:,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{summary['outside']:,}\n"
        )

        f.write(
            f"Cerca del borde:          "
            f"{summary['border']:,}\n"
        )

        f.write("\n## INTEGRIDAD\n\n")

        f.write(
            f"Duplicados:            "
            f"{summary['duplicates']:,}\n"
        )

        f.write("\n## DISTRIBUCIÓN POR SPLIT\n\n")

        for s in split_stats:

            f.write(
                f"{s['split']}\n"
            )

            f.write(
                f"  Imágenes:       "
                f"{s['images']:,}\n"
            )

            f.write(
                f"  Personas:       "
                f"{s['persons']:,}\n"
            )

            f.write(
                f"  Vehículos:      "
                f"{s['vehicles']:,}\n"
            )

            f.write(
                f"  Objetos:        "
                f"{s['objects']:,}\n"
            )

            f.write(
                f"  Obj/imagen:     "
                f"{s['objects_per_image']:.2f}\n"
            )

            f.write(
                f"  KEEP:           "
                f"{s['keep']:,}\n"
            )

            f.write(
                f"  REVIEW:         "
                f"{s['review']:,}\n"
            )

            f.write(
                f"  EXCLUDE:        "
                f"{s['exclude']:,}\n"
            )

            f.write("\n")

        f.write(
            "\n## TOP 50 IMÁGENES PARA REVISIÓN\n\n"
        )

        ranked = sorted(
            rows,
            key=lambda x: x["score"],
            reverse=True,
        )

        for i, row in enumerate(
            ranked[:50],
            1
        ):

            f.write(
                f"{i:02d}. "
                f"{row['decision']:18} "
                f"score={row['score']:7.2f} "
                f"objects={row['objects']:4d} "
                f"tiny16={row['tiny16']:3d} "
                f"partial={row['partial']:3d} "
                f"border={row['border']:3d}\n"
            )

            f.write(
                f"    reasons: "
                f"{row['reasons']}\n"
            )

            f.write(
                f"    {row['image']}\n\n"
            )

        f.write(
            "## RECOMENDACIÓN\n\n"
        )

        if summary["critical"] > 0:

            f.write(
                "CRITICAL: existen imágenes con "
                "problemas de integridad. "
                "Deben revisarse antes del entrenamiento.\n"
            )

        elif summary["exclude"] > 0:

            f.write(
                "REVIEW REQUIRED: existen imágenes "
                "candidatas a exclusión. "
                "No deben eliminarse automáticamente; "
                "deben inspeccionarse visualmente.\n"
            )

        elif summary["review"] > 0:

            f.write(
                "DATASET USABLE: existen imágenes "
                "que requieren revisión manual.\n"
            )

        else:

            f.write(
                "DATASET CLEAN: no se detectaron "
                "anomalías relevantes.\n"
            )

        f.write(
            "\nIMPORTANTE:\n"
            "Este informe es diagnóstico.\n"
            "El script NO modifica imágenes ni labels.\n"
        )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n# SAR YOLO26 - DATASET AUDIT V11\n"
    )

    print(
        f"Dataset:\n{DATASET_ROOT}\n"
    )

    print(
        f"Output:\n{OUTPUT_ROOT}\n"
    )

    if not DATASET_ROOT.exists():

        print(
            "\n[ERROR] No existe DATASET_ROOT:"
        )

        print(DATASET_ROOT)

        return

    ensure_dirs()

    all_rows = []

    for split in SPLITS:

        rows = analyze_split(split)

        all_rows.extend(rows)

    if not all_rows:

        print(
            "\n[ERROR] No se encontraron imágenes."
        )

        return

    # --------------------------------------------------------
    # CSV principal
    # --------------------------------------------------------

    write_csv(
        REPORTS_DIR / "image_audit_v11.csv",
        all_rows,
    )

    # --------------------------------------------------------
    # Resumen por split
    # --------------------------------------------------------

    split_rows = split_summary(all_rows)

    write_csv(
        REPORTS_DIR / "split_summary_v11.csv",
        split_rows,
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked = sorted(
        all_rows,
        key=lambda x: x["score"],
        reverse=True,
    )

    write_csv(
        REPORTS_DIR / "review_ranking_v11.csv",
        ranked,
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    report = write_report(
        all_rows
    )

    summary = global_summary(
        all_rows
    )

    # --------------------------------------------------------
    # CONSOLA
    # --------------------------------------------------------

    print(
        "\nImágenes:              "
        f"{summary['images']:,}"
    )

    print(
        "Personas:              "
        f"{summary['persons']:,}"
    )

    print(
        "Vehículos:             "
        f"{summary['vehicles']:,}"
    )

    print(
        "Objetos:               "
        f"{summary['objects']:,}"
    )

    print(
        "Objetos/imagen:        "
        f"{summary['objects_per_image']:.2f}"
    )

    print("\nDECISIONES")

    total = summary["images"]

    for key, label in [
        ("keep", "KEEP"),
        ("review", "REVIEW"),
        ("exclude", "EXCLUDE_CANDIDATE"),
        ("critical", "CRITICAL"),
    ]:

        value = summary[key]

        pct = (
            value / total * 100
            if total
            else 0
        )

        print(
            f"{label:20}: "
            f"{value:6,} "
            f"({pct:6.2f} %)"
        )

    print("\nOBJETOS PEQUEÑOS")

    print(
        f"<16 px²:              "
        f"{summary['tiny16']:,}"
    )

    print(
        f"<32 px²:              "
        f"{summary['tiny32']:,}"
    )

    print(
        f"<64 px²:              "
        f"{summary['tiny64']:,}"
    )

    print("\nBORDES")

    print(
        f"BBox parcialmente fuera: "
        f"{summary['partial']:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{summary['outside']:,}"
    )

    print(
        f"Cerca del borde:          "
        f"{summary['border']:,}"
    )

    print("\nINTEGRIDAD")

    print(
        f"Duplicados:            "
        f"{summary['duplicates']:,}"
    )

    print(
        "\nReports:"
    )

    print(REPORTS_DIR)

    print(
        "\nInforme:"
    )

    print(report)

    print(
        "\nIMPORTANTE: este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes o labels."
    )


if __name__ == "__main__":
    main()