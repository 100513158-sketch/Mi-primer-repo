from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv
import math
import shutil
from PIL import Image


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
    / "audit_dataset_v9"
)

REPORTS_DIR = OUTPUT_ROOT / "reports"
EXAMPLES_DIR = OUTPUT_ROOT / "examples"

SPLITS = [
    "train",
    "val",
    "test",
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

# ------------------------------------------------------------
# Umbrales
# ------------------------------------------------------------

TINY_16 = 16.0
TINY_32 = 32.0
TINY_64 = 64.0

BORDER_THRESHOLD = 0.02

CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500

# Los mismos criterios generales que V8/V7.
# No se eliminan imágenes automáticamente.
#
# La puntuación sirve SOLO para priorizar revisión.

WEIGHTS = {
    "tiny16": 1.5,
    "tiny32": 0.50,
    "partial_bbox": 4.0,
    "border_objects": 0.25,
    "crowded100": 2.0,
    "crowded200": 5.0,
    "crowded300": 10.0,
    "crowded500": 20.0,
}


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    print()
    print("# SAR YOLO26 - DATASET AUDIT V9")
    print()
    print("Dataset:")
    print(DATASET_ROOT)
    print()
    print("Output:")
    print(OUTPUT_ROOT)
    print()
    print("-" * 70)


def find_images(split_dir):
    images_dir = split_dir / "images"

    if not images_dir.exists():
        return []

    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for_image(image_path):
    """
    Busca el label manteniendo la estructura relativa
    images/... -> labels/...
    """
    parts = list(image_path.parts)

    try:
        idx = parts.index("images")
    except ValueError:
        return None

    parts[idx] = "labels"

    label_path = Path(*parts).with_suffix(".txt")

    return label_path


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None, None


def area_from_yolo(w_norm, h_norm, width, height):
    return (
        w_norm * width
    ) * (
        h_norm * height
    )


def bbox_edges(x, y, w, h):
    left = x - w / 2
    right = x + w / 2
    top = y - h / 2
    bottom = y + h / 2

    return left, top, right, bottom


# ============================================================
# ANÁLISIS DE LABEL
# ============================================================

def analyze_label(image_path, split):
    label_path = label_path_for_image(image_path)

    result = {
        "image_path": str(image_path),
        "split": split,
        "label_path": str(label_path) if label_path else "",
        "objects": 0,
        "persons": 0,
        "vehicles": 0,
        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "partial_bbox": 0,
        "outside_bbox": 0,
        "border_objects": 0,
        "invalid_lines": 0,
        "invalid_coordinates": 0,
        "invalid_bbox": 0,
        "invalid_class": 0,
        "duplicate_annotations": 0,
        "label_exists": 0,
        "corrupt_image": 0,
        "width": 0,
        "height": 0,
    }

    if label_path is None or not label_path.exists():
        return result

    result["label_exists"] = 1

    width, height = get_image_size(image_path)

    if width is None or height is None:
        result["corrupt_image"] = 1
        return result

    result["width"] = width
    result["height"] = height

    annotations = []

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()
    except Exception:
        result["invalid_lines"] += 1
        return result

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            result["invalid_lines"] += 1
            continue

        cls = None
        values = []

        try:
            cls = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except Exception:
            result["invalid_lines"] += 1
            continue

        if cls not in CLASS_NAMES:
            result["invalid_class"] += 1

        if len(values) != 4:
            result["invalid_lines"] += 1
            continue

        x, y, w, h = values

        if not all(math.isfinite(v) for v in values):
            result["invalid_coordinates"] += 1
            continue

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 <= w <= 1
            and 0 <= h <= 1
        ):
            result["invalid_coordinates"] += 1

        if w <= 0 or h <= 0:
            result["invalid_bbox"] += 1

        result["objects"] += 1

        if cls == 0:
            result["persons"] += 1
        elif cls == 1:
            result["vehicles"] += 1

        area = area_from_yolo(
            w,
            h,
            width,
            height
        )

        if area < TINY_16:
            result["tiny16"] += 1

        if area < TINY_32:
            result["tiny32"] += 1

        if area < TINY_64:
            result["tiny64"] += 1

        left, top, right, bottom = bbox_edges(
            x,
            y,
            w,
            h
        )

        outside = (
            right < 0
            or left > 1
            or bottom < 0
            or top > 1
        )

        partial = (
            left < 0
            or right > 1
            or top < 0
            or bottom > 1
        )

        if outside:
            result["outside_bbox"] += 1
        elif partial:
            result["partial_bbox"] += 1

        border = (
            x - w / 2 <= BORDER_THRESHOLD
            or x + w / 2 >= 1 - BORDER_THRESHOLD
            or y - h / 2 <= BORDER_THRESHOLD
            or y + h / 2 >= 1 - BORDER_THRESHOLD
        )

        if border:
            result["border_objects"] += 1

        annotations.append(
            (
                cls,
                round(x, 8),
                round(y, 8),
                round(w, 8),
                round(h, 8),
            )
        )

    counts = Counter(annotations)

    for _, count in counts.items():
        if count > 1:
            result["duplicate_annotations"] += count - 1

    return result


# ============================================================
# SCORE
# ============================================================

def calculate_score(row):

    score = 0.0
    reasons = []

    tiny16 = row["tiny16"]
    tiny32 = row["tiny32"]
    partial = row["partial_bbox"]
    border = row["border_objects"]
    objects = row["objects"]

    if tiny16 > 0:
        score += tiny16 * WEIGHTS["tiny16"]
        reasons.append(f"tiny16={tiny16}")

    if tiny32 > 0:
        score += tiny32 * WEIGHTS["tiny32"]
        reasons.append(f"tiny32={tiny32}")

    if partial > 0:
        score += partial * WEIGHTS["partial_bbox"]
        reasons.append("partial_bbox")

    if border > 0:
        score += border * WEIGHTS["border_objects"]
        reasons.append("border_objects")

    if objects >= CROWDED_100:
        score += WEIGHTS["crowded100"]
        reasons.append("crowded100")

    if objects >= CROWDED_200:
        score += WEIGHTS["crowded200"]
        reasons.append("crowded200")

    if objects >= CROWDED_300:
        score += WEIGHTS["crowded300"]
        reasons.append("crowded300")

    if objects >= CROWDED_500:
        score += WEIGHTS["crowded500"]
        reasons.append("crowded500")

    return score, reasons


def classify_decision(row, score, duplicate_cross_split=False):

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if row["corrupt_image"]:
        return "CRITICAL"

    if row["invalid_lines"] > 0:
        return "CRITICAL"

    if row["invalid_coordinates"] > 0:
        return "CRITICAL"

    if row["invalid_bbox"] > 0:
        return "CRITICAL"

    if row["invalid_class"] > 0:
        return "CRITICAL"

    if duplicate_cross_split:
        return "CRITICAL"

    # --------------------------------------------------------
    # EXCLUDE CANDIDATE
    #
    # IMPORTANTE:
    # candidato ≠ eliminación automática.
    # --------------------------------------------------------

    if (
        row["tiny16"] >= 50
        and row["objects"] >= 100
    ):
        return "EXCLUDE_CANDIDATE"

    if (
        row["tiny32"] >= 120
        and row["objects"] >= 200
    ):
        return "EXCLUDE_CANDIDATE"

    if (
        score >= 100
    ):
        return "EXCLUDE_CANDIDATE"

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    if score >= 15:
        return "REVIEW"

    if row["partial_bbox"] >= 5:
        return "REVIEW"

    if row["border_objects"] >= 20:
        return "REVIEW"

    if row["objects"] >= 300:
        return "REVIEW"

    return "KEEP"


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()
    print_header()

    if not DATASET_ROOT.exists():
        print()
        print("[ERROR] No existe DATASET_ROOT:")
        print(DATASET_ROOT)
        return

    all_rows = []

    split_counts = {}

    # --------------------------------------------------------
    # ANALIZAR SPLITS
    # --------------------------------------------------------

    for split in SPLITS:

        split_dir = DATASET_ROOT / split

        if not split_dir.exists():
            print(f"[INFO] Split no encontrado: {split}")
            continue

        images = find_images(split_dir)

        print()
        print(f"## Analizando: {split}")
        print()
        print(f"Imágenes encontradas: {len(images)}")

        split_counts[split] = len(images)

        for idx, image_path in enumerate(images, 1):

            row = analyze_label(
                image_path,
                split
            )

            all_rows.append(row)

            if idx % 1000 == 0:
                print(
                    f"Procesadas: {idx:,}/{len(images):,}"
                )

    if not all_rows:
        print("[ERROR] No se encontraron imágenes.")
        return

    # --------------------------------------------------------
    # HASHES
    # --------------------------------------------------------

    print()
    print("Calculando hashes SHA256...")

    hash_map = defaultdict(list)

    for idx, row in enumerate(all_rows, 1):

        image_path = Path(row["image_path"])

        try:
            digest = sha256_file(image_path)
        except Exception:
            digest = ""

        row["sha256"] = digest

        if digest:
            hash_map[digest].append(row)

        if idx % 1000 == 0:
            print(
                f"Hashes: {idx:,}/{len(all_rows):,}"
            )

    # --------------------------------------------------------
    # DUPLICADOS
    # --------------------------------------------------------

    duplicate_groups = []

    for digest, rows in hash_map.items():

        if len(rows) > 1:

            splits_present = sorted(
                set(r["split"] for r in rows)
            )

            duplicate_groups.append(
                {
                    "sha256": digest,
                    "count": len(rows),
                    "splits": splits_present,
                    "rows": rows,
                }
            )

    # --------------------------------------------------------
    # DATA LEAKAGE
    # --------------------------------------------------------

    cross_split_groups = []

    for group in duplicate_groups:

        splits_present = group["splits"]

        if len(splits_present) > 1:

            cross_split_groups.append(group)

    cross_split_paths = set()

    for group in cross_split_groups:

        for row in group["rows"]:
            cross_split_paths.add(
                row["image_path"]
            )

    # --------------------------------------------------------
    # SCORES Y DECISIONES
    # --------------------------------------------------------

    for row in all_rows:

        score, reasons = calculate_score(row)

        cross_split = (
            row["image_path"]
            in cross_split_paths
        )

        decision = classify_decision(
            row,
            score,
            duplicate_cross_split=cross_split
        )

        row["score"] = round(score, 2)
        row["decision"] = decision
        row["reasons"] = ";".join(reasons)

    # --------------------------------------------------------
    # ESTADÍSTICAS
    # --------------------------------------------------------

    total_images = len(all_rows)
    total_objects = sum(
        r["objects"] for r in all_rows
    )

    total_persons = sum(
        r["persons"] for r in all_rows
    )

    total_vehicles = sum(
        r["vehicles"] for r in all_rows
    )

    total_tiny16 = sum(
        r["tiny16"] for r in all_rows
    )

    total_tiny32 = sum(
        r["tiny32"] for r in all_rows
    )

    total_tiny64 = sum(
        r["tiny64"] for r in all_rows
    )

    total_partial = sum(
        r["partial_bbox"] for r in all_rows
    )

    total_outside = sum(
        r["outside_bbox"] for r in all_rows
    )

    total_border = sum(
        r["border_objects"] for r in all_rows
    )

    total_invalid_lines = sum(
        r["invalid_lines"] for r in all_rows
    )

    total_invalid_coordinates = sum(
        r["invalid_coordinates"] for r in all_rows
    )

    total_invalid_bbox = sum(
        r["invalid_bbox"] for r in all_rows
    )

    total_invalid_class = sum(
        r["invalid_class"] for r in all_rows
    )

    total_duplicates = sum(
        r["duplicate_annotations"] for r in all_rows
    )

    decision_counts = Counter(
        r["decision"] for r in all_rows
    )

    # --------------------------------------------------------
    # CSV: FINAL DECISIONS
    # --------------------------------------------------------

    final_csv = REPORTS_DIR / "final_decisions.csv"

    fields = [
        "split",
        "image_path",
        "label_path",
        "sha256",
        "objects",
        "persons",
        "vehicles",
        "tiny16",
        "tiny32",
        "tiny64",
        "partial_bbox",
        "outside_bbox",
        "border_objects",
        "invalid_lines",
        "invalid_coordinates",
        "invalid_bbox",
        "invalid_class",
        "duplicate_annotations",
        "score",
        "decision",
        "reasons",
    ]

    with final_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in sorted(
            all_rows,
            key=lambda x: (
                -x["score"],
                x["split"],
                x["image_path"]
            )
        ):

            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })

    # --------------------------------------------------------
    # CSV: DUPLICADOS
    # --------------------------------------------------------

    duplicates_csv = REPORTS_DIR / "image_duplicates.csv"

    with duplicates_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sha256",
            "group_size",
            "split",
            "image_path",
        ])

        for group in duplicate_groups:

            for row in group["rows"]:

                writer.writerow([
                    group["sha256"],
                    group["count"],
                    row["split"],
                    row["image_path"],
                ])

    # --------------------------------------------------------
    # CSV: CROSS SPLIT
    # --------------------------------------------------------

    leakage_csv = (
        REPORTS_DIR
        / "cross_split_duplicates.csv"
    )

    with leakage_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sha256",
            "group_size",
            "splits",
            "image_path",
        ])

        for group in cross_split_groups:

            splits_text = "|".join(
                group["splits"]
            )

            for row in group["rows"]:

                writer.writerow([
                    group["sha256"],
                    group["count"],
                    splits_text,
                    row["image_path"],
                ])

    # --------------------------------------------------------
    # CSV: CRITICAL
    # --------------------------------------------------------

    critical_csv = REPORTS_DIR / "critical_images.csv"

    with critical_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "image_path",
            "score",
            "objects",
            "reasons",
        ])

        for row in sorted(
            all_rows,
            key=lambda x: -x["score"]
        ):

            if row["decision"] == "CRITICAL":

                writer.writerow([
                    row["split"],
                    row["image_path"],
                    row["score"],
                    row["objects"],
                    row["reasons"],
                ])

    # --------------------------------------------------------
    # CSV: EXCLUDE
    # --------------------------------------------------------

    exclude_csv = (
        REPORTS_DIR
        / "exclude_candidates.csv"
    )

    with exclude_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "image_path",
            "score",
            "objects",
            "tiny16",
            "tiny32",
            "partial_bbox",
            "border_objects",
            "reasons",
        ])

        for row in sorted(
            all_rows,
            key=lambda x: -x["score"]
        ):

            if row["decision"] == "EXCLUDE_CANDIDATE":

                writer.writerow([
                    row["split"],
                    row["image_path"],
                    row["score"],
                    row["objects"],
                    row["tiny16"],
                    row["tiny32"],
                    row["partial_bbox"],
                    row["border_objects"],
                    row["reasons"],
                ])

    # --------------------------------------------------------
    # CSV: REVIEW
    # --------------------------------------------------------

    review_csv = REPORTS_DIR / "review_images.csv"

    with review_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "split",
            "image_path",
            "score",
            "objects",
            "tiny16",
            "tiny32",
            "partial_bbox",
            "border_objects",
            "reasons",
        ])

        review_rows = [
            r
            for r in all_rows
            if r["decision"] == "REVIEW"
        ]

        review_rows.sort(
            key=lambda x: -x["score"]
        )

        for rank, row in enumerate(
            review_rows,
            1
        ):

            writer.writerow([
                rank,
                row["split"],
                row["image_path"],
                row["score"],
                row["objects"],
                row["tiny16"],
                row["tiny32"],
                row["partial_bbox"],
                row["border_objects"],
                row["reasons"],
            ])

    # --------------------------------------------------------
    # CSV: SPLIT INTEGRITY
    # --------------------------------------------------------

    split_csv = REPORTS_DIR / "split_integrity.csv"

    with split_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "images",
            "objects",
            "persons",
            "vehicles",
        ])

        for split in SPLITS:

            rows = [
                r
                for r in all_rows
                if r["split"] == split
            ]

            if not rows:
                continue

            writer.writerow([
                split,
                len(rows),
                sum(r["objects"] for r in rows),
                sum(r["persons"] for r in rows),
                sum(r["vehicles"] for r in rows),
            ])

    # --------------------------------------------------------
    # TOP 50
    # --------------------------------------------------------

    top50_csv = REPORTS_DIR / "top50_review.csv"

    ranked = sorted(
        all_rows,
        key=lambda x: -x["score"]
    )[:50]

    with top50_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "decision",
            "score",
            "split",
            "objects",
            "tiny16",
            "tiny32",
            "partial_bbox",
            "border_objects",
            "reasons",
            "image_path",
        ])

        for rank, row in enumerate(
            ranked,
            1
        ):

            writer.writerow([
                rank,
                row["decision"],
                row["score"],
                row["split"],
                row["objects"],
                row["tiny16"],
                row["tiny32"],
                row["partial_bbox"],
                row["border_objects"],
                row["reasons"],
                row["image_path"],
            ])

    # --------------------------------------------------------
    # INFORME TXT
    # --------------------------------------------------------

    report_txt = (
        REPORTS_DIR
        / "AUDIT_V9_SUMMARY.txt"
    )

    with report_txt.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V9\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            "OBJETIVO\n"
        )
        f.write(
            "Auditoría final de integridad, "
            "duplicados y data leakage.\n"
        )
        f.write(
            "Este informe NO modifica el dataset.\n\n"
        )

        f.write(
            "DATASET\n"
        )
        f.write(
            f"{DATASET_ROOT}\n\n"
        )

        f.write(
            "RESUMEN\n"
        )
        f.write(
            f"Imágenes:              {total_images:,}\n"
        )
        f.write(
            f"Personas:              {total_persons:,}\n"
        )
        f.write(
            f"Vehículos:             {total_vehicles:,}\n"
        )
        f.write(
            f"Objetos:               {total_objects:,}\n"
        )

        if total_images:
            f.write(
                f"Objetos/imagen:        "
                f"{total_objects / total_images:.2f}\n"
            )

        f.write("\nDECISIONES\n")

        for decision in [
            "KEEP",
            "REVIEW",
            "EXCLUDE_CANDIDATE",
            "CRITICAL",
        ]:

            count = decision_counts[decision]

            pct = (
                count / total_images * 100
                if total_images
                else 0
            )

            f.write(
                f"{decision:20}: "
                f"{count:6,} "
                f"({pct:6.2f} %)\n"
            )

        f.write("\nOBJETOS PEQUEÑOS\n")

        f.write(
            f"<16 px²:              "
            f"{total_tiny16:,}\n"
        )

        f.write(
            f"<32 px²:              "
            f"{total_tiny32:,}\n"
        )

        f.write(
            f"<64 px²:              "
            f"{total_tiny64:,}\n"
        )

        f.write("\nBORDES\n")

        f.write(
            f"BBox parcialmente fuera: "
            f"{total_partial:,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{total_outside:,}\n"
        )

        f.write(
            f"Cerca del borde:          "
            f"{total_border:,}\n"
        )

        f.write("\nINTEGRIDAD\n")

        f.write(
            f"Labels inválidos:      "
            f"{total_invalid_lines:,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{total_invalid_coordinates:,}\n"
        )

        f.write(
            f"BBoxes inválidas:      "
            f"{total_invalid_bbox:,}\n"
        )

        f.write(
            f"Clases inválidas:      "
            f"{total_invalid_class:,}\n"
        )

        f.write(
            f"Duplicados labels:     "
            f"{total_duplicates:,}\n"
        )

        f.write("\nDUPLICADOS DE IMAGEN\n")

        f.write(
            f"Grupos duplicados:     "
            f"{len(duplicate_groups)}\n"
        )

        f.write(
            f"Grupos cross-split:    "
            f"{len(cross_split_groups)}\n"
        )

        f.write(
            f"Posible data leakage:  "
            f"{'SI' if cross_split_groups else 'NO'}\n"
        )

        f.write("\n")

        if cross_split_groups:

            f.write(
                "ATENCIÓN: se detectaron "
                "imágenes idénticas entre splits.\n"
            )

            f.write(
                "Estas imágenes deben revisarse "
                "antes del entrenamiento final.\n\n"
            )

        else:

            f.write(
                "No se detectaron imágenes "
                "idénticas entre splits.\n"
            )

        f.write("\nTOP 30\n")
        f.write("-" * 70 + "\n")

        for rank, row in enumerate(
            ranked[:30],
            1
        ):

            f.write(
                f"{rank:2}. "
                f"{row['decision']:18} "
                f"score={row['score']:7.2f} "
                f"objects={row['objects']:4} "
                f"tiny16={row['tiny16']:3} "
                f"partial={row['partial_bbox']:3} "
                f"border={row['border_objects']:3}\n"
            )

            f.write(
                f"    reasons: "
                f"{row['reasons']}\n"
            )

            f.write(
                f"    {row['image_path']}\n"
            )

        f.write("\n")
        f.write("=" * 70 + "\n")

        if cross_split_groups:

            f.write(
                "RECOMENDACIÓN: NO ENTRENAR todavía "
                "con los splits actuales hasta revisar "
                "los duplicados cross-split.\n"
            )

        elif decision_counts["CRITICAL"] > 0:

            f.write(
                "RECOMENDACIÓN: revisar primero "
                "las imágenes CRITICAL.\n"
            )

        elif decision_counts["EXCLUDE_CANDIDATE"] > 0:

            f.write(
                "RECOMENDACIÓN: revisar visualmente "
                "los EXCLUDE_CANDIDATE antes de "
                "crear una versión limpia.\n"
            )

        else:

            f.write(
                "RECOMENDACIÓN: dataset apto para "
                "pasar a la fase de preparación "
                "final del entrenamiento.\n"
            )

        f.write(
            "\nIMPORTANTE: este script SOLO diagnostica.\n"
        )

        f.write(
            "No elimina ni modifica imágenes o labels.\n"
        )

    # --------------------------------------------------------
    # CONSOLA
    # --------------------------------------------------------

    print()
    print("Imágenes:              {:,}".format(total_images))
    print("Personas:              {:,}".format(total_persons))
    print("Vehículos:             {:,}".format(total_vehicles))
    print("Objetos:               {:,}".format(total_objects))

    if total_images:
        print(
            "Objetos/imagen:        {:.2f}".format(
                total_objects / total_images
            )
        )

    print()
    print("DECISIONES")

    for decision in [
        "KEEP",
        "REVIEW",
        "EXCLUDE_CANDIDATE",
        "CRITICAL",
    ]:

        count = decision_counts[decision]

        pct = (
            count / total_images * 100
            if total_images
            else 0
        )

        print(
            f"{decision:20}: "
            f"{count:6,} "
            f"({pct:6.2f} %)"
        )

    print()
    print("OBJETOS PEQUEÑOS")

    print(
        f"<16 px²:              {total_tiny16:,}"
    )

    print(
        f"<32 px²:              {total_tiny32:,}"
    )

    print(
        f"<64 px²:              {total_tiny64:,}"
    )

    print()
    print("BORDES")

    print(
        f"BBox parcialmente fuera: "
        f"{total_partial:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{total_outside:,}"
    )

    print(
        f"Cerca del borde:          "
        f"{total_border:,}"
    )

    print()
    print("INTEGRIDAD")

    print(
        f"Labels inválidos:      "
        f"{total_invalid_lines:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{total_invalid_coordinates:,}"
    )

    print(
        f"BBoxes inválidas:      "
        f"{total_invalid_bbox:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{total_invalid_class:,}"
    )

    print(
        f"Duplicados labels:     "
        f"{total_duplicates:,}"
    )

    print()
    print("DUPLICADOS DE IMAGEN")

    print(
        f"Grupos duplicados:     "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Grupos cross-split:    "
        f"{len(cross_split_groups)}"
    )

    print(
        f"Posible data leakage:  "
        f"{'SI' if cross_split_groups else 'NO'}"
    )

    print()
    print("Reports:")
    print(REPORTS_DIR)

    print()
    print("Informe:")
    print(report_txt)

    print()
    print(
        "IMPORTANTE: este script SOLO diagnostica."
    )
    print(
        "No elimina ni modifica imágenes o labels."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()