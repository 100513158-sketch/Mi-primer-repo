from pathlib import Path
from collections import Counter
import csv
import hashlib
import math

# ============================================================
# SAR YOLO26 - DATASET AUDIT V5
#
# OBJETIVO:
#   Clasificar imágenes del dataset según riesgo de anotación.
#
# IMPORTANTE:
#   ESTE SCRIPT SOLO DIAGNOSTICA.
#   NO ELIMINA NI MODIFICA IMÁGENES.
#   NO MODIFICA LABELS.
#
# CATEGORÍAS:
#   KEEP
#   REVIEW
#   EXCLUDE_CANDIDATE
#   CRITICAL
#
# ============================================================


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO"
    r"\processed\sar\VisDrone_SAR_2CLASS"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"\baseline\evaluation\dataset_analysis\audit\audit_dataset_v5"
)

SPLITS = [
    "train",
    "val",
    "test",
    "test_dev",
]

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}


# ============================================================
# UMBRALES
# ============================================================

# Área bbox en píxeles.
# Se calcula usando dimensiones de imagen.
TINY_AREA = 16
VERY_SMALL_AREA = 32
SMALL_AREA = 64
SMALL_100_AREA = 100

# Distancia mínima al borde.
BORDER_MARGIN_PX = 5

# Densidad.
# IMPORTANTE:
# crowded NO genera exclusión por sí misma.
CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500

# Porcentaje de objetos problemáticos
# dentro de una imagen.
HIGH_TINY_RATIO = 0.25
EXTREME_TINY_RATIO = 0.50

HIGH_BORDER_RATIO = 0.30
EXTREME_BORDER_RATIO = 0.60

HIGH_PARTIAL_RATIO = 0.20
EXTREME_PARTIAL_RATIO = 0.50

# Score.
REVIEW_SCORE = 4
EXCLUDE_SCORE = 8
CRITICAL_SCORE = 12


# ============================================================
# UTILIDADES
# ============================================================

def find_images(images_dir):
    extensions = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    )

    files = []

    for ext in extensions:
        files.extend(images_dir.rglob(ext))

    return sorted(files)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def safe_ratio(value, total):
    if total <= 0:
        return 0.0

    return value / total


def bbox_iou(a, b):
    """
    IoU entre dos bounding boxes:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def annotation_hash(line):
    """
    Hash normalizado de una anotación.
    Permite detectar duplicados exactos.
    """

    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        class_id = int(parts[0])

        values = [
            round(float(x), 8)
            for x in parts[1:5]
        ]

        normalized = (
            class_id,
            *values,
        )

        return hashlib.md5(
            repr(normalized).encode(
                "utf-8"
            )
        ).hexdigest()

    except Exception:
        return None


# ============================================================
# ANÁLISIS DE UNA IMAGEN
# ============================================================

def analyze_image(
    image_path,
    label_path,
    split,
):
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size

    except Exception as exc:

        return {
            "split": split,
            "image": str(image_path),
            "width": 0,
            "height": 0,
            "objects": 0,
            "persons": 0,
            "vehicles": 0,
            "tiny16": 0,
            "tiny32": 0,
            "tiny64": 0,
            "tiny100": 0,
            "border_objects": 0,
            "partial_objects": 0,
            "outside_objects": 0,
            "invalid_labels": 0,
            "invalid_coordinates": 0,
            "invalid_bbox": 0,
            "invalid_classes": 0,
            "duplicate_annotations": 0,
            "missing_label": not label_path.exists(),
            "corrupt_image": True,
            "crowded": False,
            "risk_score": CRITICAL_SCORE,
            "decision": "CRITICAL",
            "reasons": f"Imagen corrupta: {exc}",
        }

    objects = 0
    persons = 0
    vehicles = 0

    tiny16 = 0
    tiny32 = 0
    tiny64 = 0
    tiny100 = 0

    border_objects = 0
    partial_objects = 0
    outside_objects = 0

    invalid_labels = 0
    invalid_coordinates = 0
    invalid_bbox = 0
    invalid_classes = 0
    duplicate_annotations = 0

    reasons = []

    annotations = []
    annotation_hashes = Counter()

    if not label_path.exists():

        return {
            "split": split,
            "image": str(image_path),
            "width": width,
            "height": height,
            "objects": 0,
            "persons": 0,
            "vehicles": 0,
            "tiny16": 0,
            "tiny32": 0,
            "tiny64": 0,
            "tiny100": 0,
            "border_objects": 0,
            "partial_objects": 0,
            "outside_objects": 0,
            "invalid_labels": 0,
            "invalid_coordinates": 0,
            "invalid_bbox": 0,
            "invalid_classes": 0,
            "duplicate_annotations": 0,
            "missing_label": True,
            "corrupt_image": False,
            "crowded": False,
            "risk_score": CRITICAL_SCORE,
            "decision": "CRITICAL",
            "reasons": "Label inexistente",
        }

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception as exc:

        return {
            "split": split,
            "image": str(image_path),
            "width": width,
            "height": height,
            "objects": 0,
            "persons": 0,
            "vehicles": 0,
            "tiny16": 0,
            "tiny32": 0,
            "tiny64": 0,
            "tiny100": 0,
            "border_objects": 0,
            "partial_objects": 0,
            "outside_objects": 0,
            "invalid_labels": 1,
            "invalid_coordinates": 0,
            "invalid_bbox": 0,
            "invalid_classes": 0,
            "duplicate_annotations": 0,
            "missing_label": False,
            "corrupt_image": False,
            "crowded": False,
            "risk_score": CRITICAL_SCORE,
            "decision": "CRITICAL",
            "reasons": f"Label ilegible: {exc}",
        }

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):

        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:

            invalid_labels += 1

            continue

        try:

            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

        except Exception:

            invalid_labels += 1

            continue

        # ----------------------------------------------------
        # CLASE
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            invalid_classes += 1

        else:

            if class_id == 0:
                persons += 1

            elif class_id == 1:
                vehicles += 1

        # ----------------------------------------------------
        # COORDENADAS NORMALIZADAS
        # ----------------------------------------------------

        coordinates_ok = (
            math.isfinite(xc)
            and math.isfinite(yc)
            and math.isfinite(bw)
            and math.isfinite(bh)
        )

        if not coordinates_ok:

            invalid_coordinates += 1

            continue

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        if bw <= 0 or bh <= 0:

            invalid_bbox += 1

            continue

        # ----------------------------------------------------
        # COORDENADAS EN PIXELES
        # ----------------------------------------------------

        x_center = xc * width
        y_center = yc * height

        box_width = bw * width
        box_height = bh * height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        # ----------------------------------------------------
        # ÁREA
        # ----------------------------------------------------

        area = box_width * box_height

        objects += 1

        if area < TINY_AREA:
            tiny16 += 1

        if area < VERY_SMALL_AREA:
            tiny32 += 1

        if area < SMALL_AREA:
            tiny64 += 1

        if area < SMALL_100_AREA:
            tiny100 += 1

        # ----------------------------------------------------
        # BORDES
        # ----------------------------------------------------

        completely_outside = (
            x2 <= 0
            or y2 <= 0
            or x1 >= width
            or y1 >= height
        )

        partially_outside = (
            x1 < 0
            or y1 < 0
            or x2 > width
            or y2 > height
        )

        if completely_outside:

            outside_objects += 1

        elif partially_outside:

            partial_objects += 1

        # ----------------------------------------------------
        # CERCA DEL BORDE
        # ----------------------------------------------------

        near_border = (
            x1 <= BORDER_MARGIN_PX
            or y1 <= BORDER_MARGIN_PX
            or x2 >= width - BORDER_MARGIN_PX
            or y2 >= height - BORDER_MARGIN_PX
        )

        if near_border:

            border_objects += 1

        # ----------------------------------------------------
        # DUPLICADOS
        # ----------------------------------------------------

        h = annotation_hash(line)

        if h is not None:

            annotation_hashes[h] += 1

        annotations.append(
            (
                class_id,
                [x1, y1, x2, y2],
            )
        )

    # ========================================================
    # DUPLICADOS
    # ========================================================

    for count in annotation_hashes.values():

        if count > 1:

            duplicate_annotations += (
                count - 1
            )

    # ========================================================
    # RATIOS
    # ========================================================

    tiny_ratio = safe_ratio(
        tiny16,
        objects,
    )

    border_ratio = safe_ratio(
        border_objects,
        objects,
    )

    partial_ratio = safe_ratio(
        partial_objects,
        objects,
    )

    # ========================================================
    # CROWDED
    # ========================================================

    crowded = objects >= CROWDED_100

    # ========================================================
    # SCORE
    # ========================================================

    score = 0

    # --------------------------------------------------------
    # Integridad
    # --------------------------------------------------------

    if invalid_labels > 0:
        score += 8
        reasons.append(
            "labels_invalidos"
        )

    if invalid_coordinates > 0:
        score += 8
        reasons.append(
            "coordenadas_invalidas"
        )

    if invalid_bbox > 0:
        score += 8
        reasons.append(
            "bbox_invalida"
        )

    if invalid_classes > 0:
        score += 8
        reasons.append(
            "clase_invalida"
        )

    if duplicate_annotations > 0:
        score += min(
            4,
            duplicate_annotations,
        )

        reasons.append(
            "duplicados"
        )

    # --------------------------------------------------------
    # Objetos diminutos
    # --------------------------------------------------------

    if tiny_ratio >= EXTREME_TINY_RATIO:

        score += 5

        reasons.append(
            "muchos_objetos_extremos"
        )

    elif tiny_ratio >= HIGH_TINY_RATIO:

        score += 3

        reasons.append(
            "muchos_objetos_tiny"
        )

    elif tiny16 > 0:

        score += 1

        reasons.append(
            "objetos_tiny"
        )

    # --------------------------------------------------------
    # BBOX parcialmente fuera
    # --------------------------------------------------------

    if partial_ratio >= EXTREME_PARTIAL_RATIO:

        score += 5

        reasons.append(
            "muchas_bbox_parciales"
        )

    elif partial_ratio >= HIGH_PARTIAL_RATIO:

        score += 3

        reasons.append(
            "bbox_parciales"
        )

    elif partial_objects > 0:

        score += 1

        reasons.append(
            "bbox_parcial"
        )

    # --------------------------------------------------------
    # Borde
    # --------------------------------------------------------

    if border_ratio >= EXTREME_BORDER_RATIO:

        score += 4

        reasons.append(
            "muchos_objetos_borde"
        )

    elif border_ratio >= HIGH_BORDER_RATIO:

        score += 2

        reasons.append(
            "objetos_borde"
        )

    # --------------------------------------------------------
    # DENSIDAD
    #
    # NO penalizamos crowded.
    # Solo lo registramos.
    # --------------------------------------------------------

    if objects >= CROWDED_500:

        reasons.append(
            "escena_extremadamente_densa"
        )

    elif objects >= CROWDED_300:

        reasons.append(
            "escena_muy_densa"
        )

    elif objects >= CROWDED_200:

        reasons.append(
            "escena_densa"
        )

    # ========================================================
    # DECISIÓN
    # ========================================================

    if score >= CRITICAL_SCORE:

        decision = "CRITICAL"

    elif score >= EXCLUDE_SCORE:

        decision = "EXCLUDE_CANDIDATE"

    elif score >= REVIEW_SCORE:

        decision = "REVIEW"

    else:

        decision = "KEEP"

    # --------------------------------------------------------
    # Excepción:
    #
    # Una escena crowded válida no debe convertirse
    # automáticamente en EXCLUDE.
    # --------------------------------------------------------

    reason_text = (
        ";".join(reasons)
        if reasons
        else "sin_anomalias_significativas"
    )

    return {
        "split": split,
        "image": str(image_path),
        "width": width,
        "height": height,
        "objects": objects,
        "persons": persons,
        "vehicles": vehicles,
        "tiny16": tiny16,
        "tiny32": tiny32,
        "tiny64": tiny64,
        "tiny100": tiny100,
        "tiny16_ratio": tiny_ratio,
        "border_objects": border_objects,
        "border_ratio": border_ratio,
        "partial_objects": partial_objects,
        "partial_ratio": partial_ratio,
        "outside_objects": outside_objects,
        "invalid_labels": invalid_labels,
        "invalid_coordinates": invalid_coordinates,
        "invalid_bbox": invalid_bbox,
        "invalid_classes": invalid_classes,
        "duplicate_annotations": duplicate_annotations,
        "missing_label": False,
        "corrupt_image": False,
        "crowded": crowded,
        "risk_score": score,
        "decision": decision,
        "reasons": reason_text,
    }


# ============================================================
# CSV
# ============================================================

def write_csv(
    path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAR YOLO26 - DATASET AUDIT V5")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    print()

    if not DATASET_ROOT.exists():

        print(
            "[ERROR] No existe DATASET_ROOT:"
        )

        print(DATASET_ROOT)

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir = (
        OUTPUT_ROOT / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    # ========================================================
    # SPLITS
    # ========================================================

    for split in SPLITS:

        split_dir = (
            DATASET_ROOT / split
        )

        if not split_dir.exists():

            print(
                f"[INFO] Split no encontrado: "
                f"{split}"
            )

            continue

        images_dir = (
            split_dir / "images"
        )

        labels_dir = (
            split_dir / "labels"
        )

        if not images_dir.exists():

            print(
                f"[WARN] No existe: "
                f"{images_dir}"
            )

            continue

        if not labels_dir.exists():

            print(
                f"[WARN] No existe: "
                f"{labels_dir}"
            )

            continue

        print()
        print(
            f"## Analizando: {split}"
        )

        image_files = find_images(
            images_dir
        )

        print(
            f"Imágenes encontradas: "
            f"{len(image_files):,}"
        )

        for index, image_path in enumerate(
            image_files,
            start=1,
        ):

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            row = analyze_image(
                image_path,
                label_path,
                split,
            )

            all_rows.append(row)

            if index % 1000 == 0:

                print(
                    f"Procesadas: "
                    f"{index:,}/"
                    f"{len(image_files):,}"
                )

    # ========================================================
    # RESUMEN
    # ========================================================

    total_images = len(all_rows)

    total_objects = sum(
        r["objects"]
        for r in all_rows
    )

    total_persons = sum(
        r["persons"]
        for r in all_rows
    )

    total_vehicles = sum(
        r["vehicles"]
        for r in all_rows
    )

    # ========================================================
    # DECISIONES
    # ========================================================

    decision_counts = Counter(
        r["decision"]
        for r in all_rows
    )

    # ========================================================
    # OBJETOS
    # ========================================================

    tiny16 = sum(
        r["tiny16"]
        for r in all_rows
    )

    tiny32 = sum(
        r["tiny32"]
        for r in all_rows
    )

    tiny64 = sum(
        r["tiny64"]
        for r in all_rows
    )

    tiny100 = sum(
        r["tiny100"]
        for r in all_rows
    )

    partial = sum(
        r["partial_objects"]
        for r in all_rows
    )

    outside = sum(
        r["outside_objects"]
        for r in all_rows
    )

    border = sum(
        r["border_objects"]
        for r in all_rows
    )

    invalid_labels = sum(
        r["invalid_labels"]
        for r in all_rows
    )

    invalid_coordinates = sum(
        r["invalid_coordinates"]
        for r in all_rows
    )

    invalid_bbox = sum(
        r["invalid_bbox"]
        for r in all_rows
    )

    invalid_classes = sum(
        r["invalid_classes"]
        for r in all_rows
    )

    duplicates = sum(
        r["duplicate_annotations"]
        for r in all_rows
    )

    # ========================================================
    # CROWDED
    # ========================================================

    crowded_100 = sum(
        1
        for r in all_rows
        if r["objects"] >= 100
    )

    crowded_200 = sum(
        1
        for r in all_rows
        if r["objects"] >= 200
    )

    crowded_300 = sum(
        1
        for r in all_rows
        if r["objects"] >= 300
    )

    crowded_500 = sum(
        1
        for r in all_rows
        if r["objects"] >= 500
    )

    # ========================================================
    # CSV GENERAL
    # ========================================================

    write_csv(
        reports_dir
        / "image_decisions.csv",
        all_rows,
    )

    # ========================================================
    # KEEP
    # ========================================================

    keep_rows = [
        r
        for r in all_rows
        if r["decision"] == "KEEP"
    ]

    write_csv(
        reports_dir
        / "keep_images.csv",
        keep_rows,
    )

    # ========================================================
    # REVIEW
    # ========================================================

    review_rows = [
        r
        for r in all_rows
        if r["decision"] == "REVIEW"
    ]

    review_rows.sort(
        key=lambda x: x["risk_score"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "review_candidates.csv",
        review_rows,
    )

    # ========================================================
    # EXCLUDE CANDIDATE
    # ========================================================

    exclude_rows = [
        r
        for r in all_rows
        if r["decision"]
        == "EXCLUDE_CANDIDATE"
    ]

    exclude_rows.sort(
        key=lambda x: x["risk_score"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "exclude_candidates.csv",
        exclude_rows,
    )

    # ========================================================
    # CRITICAL
    # ========================================================

    critical_rows = [
        r
        for r in all_rows
        if r["decision"]
        == "CRITICAL"
    ]

    critical_rows.sort(
        key=lambda x: x["risk_score"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "critical_images.csv",
        critical_rows,
    )

    # ========================================================
    # TINY
    # ========================================================

    tiny_rows = [
        r
        for r in all_rows
        if r["tiny16"] > 0
    ]

    tiny_rows.sort(
        key=lambda x: x["tiny16"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "tiny_object_images.csv",
        tiny_rows,
    )

    # ========================================================
    # BORDER
    # ========================================================

    border_rows = [
        r
        for r in all_rows
        if r["border_objects"] > 0
    ]

    border_rows.sort(
        key=lambda x: x["border_objects"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "border_object_images.csv",
        border_rows,
    )

    # ========================================================
    # PARTIAL BBOX
    # ========================================================

    partial_rows = [
        r
        for r in all_rows
        if r["partial_objects"] > 0
    ]

    partial_rows.sort(
        key=lambda x: x["partial_objects"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "partial_bbox_images.csv",
        partial_rows,
    )

    # ========================================================
    # CROWDED
    # ========================================================

    crowded_rows = [
        r
        for r in all_rows
        if r["objects"] >= 100
    ]

    crowded_rows.sort(
        key=lambda x: x["objects"],
        reverse=True,
    )

    write_csv(
        reports_dir
        / "crowded_images.csv",
        crowded_rows,
    )

    # ========================================================
    # TOP REVIEW
    # ========================================================

    top_review = sorted(
        all_rows,
        key=lambda x: (
            x["risk_score"],
            x["tiny16"],
            x["partial_objects"],
            x["border_objects"],
        ),
        reverse=True,
    )

    write_csv(
        reports_dir
        / "top_review_images.csv",
        top_review[:200],
    )

    # ========================================================
    # RESUMEN TXT
    # ========================================================

    summary_path = (
        reports_dir
        / "AUDIT_V5_SUMMARY.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V5\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Dataset:\n"
        )

        f.write(
            f"{DATASET_ROOT}\n\n"
        )

        f.write(
            "RESULTADO GENERAL\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Imágenes: {total_images:,}\n"
        )

        f.write(
            f"Personas: {total_persons:,}\n"
        )

        f.write(
            f"Vehículos: {total_vehicles:,}\n"
        )

        f.write(
            f"Objetos: {total_objects:,}\n"
        )

        if total_images:

            f.write(
                "Objetos/imagen: "
                f"{total_objects / total_images:.2f}\n"
            )

        f.write("\n")

        f.write(
            "DECISIONES\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for decision in (
            "KEEP",
            "REVIEW",
            "EXCLUDE_CANDIDATE",
            "CRITICAL",
        ):

            count = decision_counts.get(
                decision,
                0,
            )

            percentage = (
                count / total_images * 100
                if total_images
                else 0
            )

            f.write(
                f"{decision:20} "
                f"{count:8,} "
                f"({percentage:6.2f} %)\n"
            )

        f.write("\n")

        f.write(
            "OBJETOS PEQUEÑOS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"<16 px² : {tiny16:,}\n"
        )

        f.write(
            f"<32 px² : {tiny32:,}\n"
        )

        f.write(
            f"<64 px² : {tiny64:,}\n"
        )

        f.write(
            f"<100 px²: {tiny100:,}\n"
        )

        f.write("\n")

        f.write(
            "BORDES\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Parcialmente fuera: "
            f"{partial:,}\n"
        )

        f.write(
            f"Completamente fuera: "
            f"{outside:,}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{border:,}\n"
        )

        f.write("\n")

        f.write(
            "INTEGRIDAD\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Labels inválidos: "
            f"{invalid_labels:,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{invalid_coordinates:,}\n"
        )

        f.write(
            f"BBoxes inválidas: "
            f"{invalid_bbox:,}\n"
        )

        f.write(
            f"Clases inválidas: "
            f"{invalid_classes:,}\n"
        )

        f.write(
            f"Duplicados: "
            f"{duplicates:,}\n"
        )

        f.write("\n")

        f.write(
            "ESCENAS DENSAS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f">=100 objetos: "
            f"{crowded_100:,}\n"
        )

        f.write(
            f">=200 objetos: "
            f"{crowded_200:,}\n"
        )

        f.write(
            f">=300 objetos: "
            f"{crowded_300:,}\n"
        )

        f.write(
            f">=500 objetos: "
            f"{crowded_500:,}\n"
        )

        f.write("\n")

        f.write(
            "IMPORTANTE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Las escenas densas NO se consideran "
            "anomalías automáticamente.\n"
        )

        f.write(
            "Este script SOLO diagnostica.\n"
        )

        f.write(
            "No elimina ni modifica imágenes "
            "o labels.\n"
        )

    # ========================================================
    # CONSOLA
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADO GENERAL")
    print("=" * 70)

    print(
        f"Imágenes:              {total_images:,}"
    )

    print(
        f"Personas:              {total_persons:,}"
    )

    print(
        f"Vehículos:             {total_vehicles:,}"
    )

    print(
        f"Objetos:               {total_objects:,}"
    )

    if total_images:

        print(
            f"Objetos/imagen:        "
            f"{total_objects / total_images:.2f}"
        )

    print()
    print("DECISIONES")
    print("-" * 70)

    for decision in (
        "KEEP",
        "REVIEW",
        "EXCLUDE_CANDIDATE",
        "CRITICAL",
    ):

        count = decision_counts.get(
            decision,
            0,
        )

        percentage = (
            count / total_images * 100
            if total_images
            else 0
        )

        print(
            f"{decision:20} "
            f"{count:8,} "
            f"({percentage:6.2f} %)"
        )

    print()
    print("OBJETOS PEQUEÑOS")
    print("-" * 70)

    print(
        f"<16 px²               : "
        f"{tiny16:,}"
    )

    print(
        f"<32 px²               : "
        f"{tiny32:,}"
    )

    print(
        f"<64 px²               : "
        f"{tiny64:,}"
    )

    print(
        f"<100 px²              : "
        f"{tiny100:,}"
    )

    print()
    print("BORDES")
    print("-" * 70)

    print(
        f"BBox parcialmente fuera: "
        f"{partial:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{outside:,}"
    )

    print(
        f"Cerca del borde: "
        f"{border:,}"
    )

    print()
    print("INTEGRIDAD")
    print("-" * 70)

    print(
        f"Labels inválidos:      "
        f"{invalid_labels:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{invalid_coordinates:,}"
    )

    print(
        f"BBoxes inválidas:      "
        f"{invalid_bbox:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{invalid_classes:,}"
    )

    print(
        f"Duplicados:            "
        f"{duplicates:,}"
    )

    print()
    print("CROWDED")
    print("-" * 70)

    print(
        f">= 100 objetos: "
        f"{crowded_100:,} imágenes"
    )

    print(
        f">= 200 objetos: "
        f"{crowded_200:,} imágenes"
    )

    print(
        f">= 300 objetos: "
        f"{crowded_300:,} imágenes"
    )

    print(
        f">= 500 objetos: "
        f"{crowded_500:,} imágenes"
    )

    print()
    print("Reports:")
    print(reports_dir)

    print()
    print(
        "IMPORTANTE: este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes o labels."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()