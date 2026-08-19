from pathlib import Path
from collections import Counter
import csv
import hashlib
import shutil
import math

# ============================================================
# SAR YOLO26 - DATASET AUDIT V6
#
# OBJETIVO:
#   Auditoría explicable del dataset.
#
# IMPORTANTE:
#   ESTE SCRIPT NO ELIMINA NI MODIFICA NADA.
#
#   Genera:
#       - decisión por imagen
#       - puntuación
#       - motivos
#       - auditoría por objeto
#       - candidatos REVIEW
#       - candidatos EXCLUDE_CANDIDATE
#       - ejemplos visuales
#       - resumen global
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
    r"\baseline\evaluation\dataset_analysis\audit\audit_dataset_v6"
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

# Área normalizada:
# width * height
#
# IMPORTANTE:
# Las áreas se calculan inicialmente sobre la imagen real
# para poder trabajar en píxeles cuadrados.

TINY_16 = 16
TINY_32 = 32
TINY_64 = 64
TINY_100 = 100


# Borde.
#
# Distancia en píxeles hasta el borde de la imagen.
BORDER_PIXELS = 5


# Crowded.
CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500


# ============================================================
# DECISIONES
# ============================================================

KEEP = "KEEP"
REVIEW = "REVIEW"
EXCLUDE = "EXCLUDE_CANDIDATE"
CRITICAL = "CRITICAL"


# ============================================================
# EJEMPLOS
# ============================================================

MAX_REVIEW_EXAMPLES = 50
MAX_EXCLUDE_EXAMPLES = 20
MAX_CROWDED_EXAMPLES = 20
MAX_TINY_EXAMPLES = 20
MAX_BORDER_EXAMPLES = 20


# ============================================================
# PESOS DEL SCORE
# ============================================================
#
# El objetivo NO es eliminar automáticamente.
#
# Un score alto significa:
#   "merece revisión"
#
# No significa:
#   "esta imagen es incorrecta".
#
# ============================================================

SCORE_TINY_16 = 1
SCORE_TINY_32 = 0.5

SCORE_PARTIAL_BBOX = 1

SCORE_BORDER = 0.25

SCORE_DUPLICATE = 3

SCORE_CROWDED_200 = 1
SCORE_CROWDED_300 = 2
SCORE_CROWDED_500 = 3


# ============================================================
# UTILIDADES
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(values[int(k)])

    return (
        values[f] * (c - k)
        + values[c] * (k - f)
    )


def write_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def image_hash(path):

    try:

        h = hashlib.md5()

        with path.open("rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception:

        return None


# ============================================================
# IMÁGENES
# ============================================================

def find_images(images_dir):

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]

    image_files = []

    for ext in extensions:
        image_files.extend(
            images_dir.rglob(ext)
        )

    return sorted(set(image_files))


# ============================================================
# LECTURA LABEL
# ============================================================

def read_label_file(label_path):

    records = []

    errors = []

    if not label_path.exists():
        return records, errors

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception as exc:

        errors.append(
            f"read_error:{exc}"
        )

        return records, errors

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:

            errors.append(
                f"invalid_label_fields:{line_number}"
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

        records.append({

            "line": line_number,

            "class_id": class_id,

            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h,

        })

    return records, errors


# ============================================================
# ANALIZAR OBJETO
# ============================================================

def analyze_object(
    record,
    image_width,
    image_height
):

    class_id = record["class_id"]

    xc = record["xc"]
    yc = record["yc"]

    w = record["w"]
    h = record["h"]

    result = {

        "class_id": class_id,

        "class_name":
            CLASS_NAMES.get(
                class_id,
                "INVALID"
            ),

        "xc": xc,
        "yc": yc,

        "w_norm": w,
        "h_norm": h,

        "valid_coordinates": True,
        "valid_bbox": True,

        "center_outside": False,

        "partial_outside": False,

        "fully_outside": False,

        "near_border": False,

        "area_px2": 0.0,

    }

    # --------------------------------------------------------
    # CLASE
    # --------------------------------------------------------

    if class_id not in CLASS_NAMES:

        result["valid_coordinates"] = False

    # --------------------------------------------------------
    # COORDENADAS
    # --------------------------------------------------------

    if not (
        0 <= xc <= 1
        and
        0 <= yc <= 1
    ):

        result["valid_coordinates"] = False

        result["center_outside"] = True

    # --------------------------------------------------------
    # DIMENSIONES
    # --------------------------------------------------------

    if not (
        0 < w <= 1
        and
        0 < h <= 1
    ):

        result["valid_bbox"] = False

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    width_px = w * image_width
    height_px = h * image_height

    area_px2 = width_px * height_px

    result["area_px2"] = area_px2

    # --------------------------------------------------------
    # COORDENADAS ABSOLUTAS
    # --------------------------------------------------------

    x1 = (
        xc - w / 2
    ) * image_width

    y1 = (
        yc - h / 2
    ) * image_height

    x2 = (
        xc + w / 2
    ) * image_width

    y2 = (
        yc + h / 2
    ) * image_height

    # --------------------------------------------------------
    # BBOX COMPLETAMENTE FUERA
    # --------------------------------------------------------

    if (
        x2 <= 0
        or
        y2 <= 0
        or
        x1 >= image_width
        or
        y1 >= image_height
    ):

        result["fully_outside"] = True

    # --------------------------------------------------------
    # BBOX PARCIALMENTE FUERA
    # --------------------------------------------------------

    elif (
        x1 < 0
        or
        y1 < 0
        or
        x2 > image_width
        or
        y2 > image_height
    ):

        result["partial_outside"] = True

    # --------------------------------------------------------
    # CERCA DEL BORDE
    # --------------------------------------------------------

    distance_left = abs(x1)

    distance_top = abs(y1)

    distance_right = abs(
        image_width - x2
    )

    distance_bottom = abs(
        image_height - y2
    )

    min_distance = min(
        distance_left,
        distance_top,
        distance_right,
        distance_bottom,
    )

    if min_distance <= BORDER_PIXELS:

        result["near_border"] = True

    return result


# ============================================================
# DECISION SCORE
# ============================================================

def calculate_decision(
    image_stats
):

    score = 0.0

    reasons = []

    # --------------------------------------------------------
    # OBJETOS EXTREMADAMENTE PEQUEÑOS
    # --------------------------------------------------------

    if image_stats["tiny16"] > 0:

        score += (
            image_stats["tiny16"]
            * SCORE_TINY_16
        )

        reasons.append(
            f"tiny16={image_stats['tiny16']}"
        )

    if image_stats["tiny32"] > 0:

        score += (
            image_stats["tiny32"]
            * SCORE_TINY_32
        )

        reasons.append(
            f"tiny32={image_stats['tiny32']}"
        )

    # --------------------------------------------------------
    # BBOX PARCIAL
    # --------------------------------------------------------

    if image_stats["partial_outside"] > 0:

        score += (
            image_stats["partial_outside"]
            * SCORE_PARTIAL_BBOX
        )

        reasons.append(
            "partial_bbox"
        )

    # --------------------------------------------------------
    # BORDE
    # --------------------------------------------------------

    if image_stats["near_border"] > 0:

        score += (
            image_stats["near_border"]
            * SCORE_BORDER
        )

        reasons.append(
            "border_objects"
        )

    # --------------------------------------------------------
    # DUPLICADOS
    # --------------------------------------------------------

    if image_stats["duplicates"] > 0:

        score += (
            image_stats["duplicates"]
            * SCORE_DUPLICATE
        )

        reasons.append(
            "duplicate_annotations"
        )

    # --------------------------------------------------------
    # CROWDED
    # --------------------------------------------------------

    objects = image_stats["objects"]

    if objects >= CROWDED_500:

        score += SCORE_CROWDED_500

        reasons.append(
            "crowded500"
        )

    elif objects >= CROWDED_300:

        score += SCORE_CROWDED_300

        reasons.append(
            "crowded300"
        )

    elif objects >= CROWDED_200:

        score += SCORE_CROWDED_200

        reasons.append(
            "crowded200"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    #
    # CRITICAL:
    # anomalías estructurales graves.
    #

    if (
        image_stats["invalid_labels"] > 0
        or
        image_stats["invalid_coordinates"] > 0
        or
        image_stats["invalid_bboxes"] > 0
        or
        image_stats["invalid_classes"] > 0
        or
        image_stats["fully_outside"] > 0
    ):

        decision = CRITICAL

    #
    # EXCLUDE CANDIDATE:
    # combinación fuerte de anomalías.
    #

    elif (
        score >= 20
        or
        (
            image_stats["objects"] == 0
            and
            image_stats["label_exists"]
        )
    ):

        decision = EXCLUDE

    #
    # REVIEW:
    # merece inspección humana.
    #

    elif score >= 5:

        decision = REVIEW

    else:

        decision = KEEP

    return (
        decision,
        round(score, 2),
        ";".join(reasons)
    )


# ============================================================
# ANALIZAR SPLIT
# ============================================================

def analyze_split(
    split_name,
    split_dir
):

    images_dir = (
        split_dir / "images"
    )

    labels_dir = (
        split_dir / "labels"
    )

    if not images_dir.exists():

        print(
            f"[INFO] No existe images: "
            f"{images_dir}"
        )

        return []

    if not labels_dir.exists():

        print(
            f"[INFO] No existe labels: "
            f"{labels_dir}"
        )

        return []

    image_files = find_images(
        images_dir
    )

    print(
        f"\n## Analizando: {split_name}"
    )

    print(
        f"Imágenes encontradas: "
        f"{len(image_files):,}"
    )

    rows = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Información imagen
        # ----------------------------------------------------

        try:

            from PIL import Image

            with Image.open(image_path) as im:

                image_width, image_height = (
                    im.size
                )

                im.verify()

        except Exception as exc:

            rows.append({

                "split": split_name,

                "image": str(image_path),

                "label": str(label_path),

                "decision": CRITICAL,

                "score": 100,

                "reasons":
                    f"corrupt_image:{exc}",

                "objects": 0,

                "tiny16": 0,
                "tiny32": 0,
                "tiny64": 0,

                "partial_outside": 0,
                "fully_outside": 0,
                "near_border": 0,

                "duplicates": 0,

                "invalid_labels": 0,
                "invalid_coordinates": 0,
                "invalid_bboxes": 0,
                "invalid_classes": 0,

                "label_exists":
                    label_path.exists(),

                "width": 0,
                "height": 0,

            })

            continue

        records, errors = read_label_file(
            label_path
        )

        # ----------------------------------------------------
        # Estadísticas imagen
        # ----------------------------------------------------

        image_stats = {

            "objects": 0,

            "tiny16": 0,
            "tiny32": 0,
            "tiny64": 0,

            "partial_outside": 0,
            "fully_outside": 0,
            "near_border": 0,

            "duplicates": 0,

            "invalid_labels":
                len(errors),

            "invalid_coordinates": 0,

            "invalid_bboxes": 0,

            "invalid_classes": 0,

            "label_exists":
                label_path.exists(),

        }

        # ----------------------------------------------------
        # Detección de duplicados
        # ----------------------------------------------------

        seen_annotations = set()

        object_details = []

        for record in records:

            obj = analyze_object(
                record,
                image_width,
                image_height
            )

            object_details.append(obj)

            image_stats["objects"] += 1

            area = obj["area_px2"]

            if area < TINY_16:

                image_stats["tiny16"] += 1

            if area < TINY_32:

                image_stats["tiny32"] += 1

            if area < TINY_64:

                image_stats["tiny64"] += 1

            if obj["partial_outside"]:

                image_stats[
                    "partial_outside"
                ] += 1

            if obj["fully_outside"]:

                image_stats[
                    "fully_outside"
                ] += 1

            if obj["near_border"]:

                image_stats[
                    "near_border"
                ] += 1

            if not obj[
                "valid_coordinates"
            ]:

                image_stats[
                    "invalid_coordinates"
                ] += 1

            if not obj[
                "valid_bbox"
            ]:

                image_stats[
                    "invalid_bboxes"
                ] += 1

            if obj["class_id"] not in CLASS_NAMES:

                image_stats[
                    "invalid_classes"
                ] += 1

            annotation_key = (

                obj["class_id"],

                round(
                    obj["xc"],
                    8
                ),

                round(
                    obj["yc"],
                    8
                ),

                round(
                    obj["w_norm"],
                    8
                ),

                round(
                    obj["h_norm"],
                    8
                ),

            )

            if annotation_key in seen_annotations:

                image_stats[
                    "duplicates"
                ] += 1

            else:

                seen_annotations.add(
                    annotation_key
                )

        # ----------------------------------------------------
        # DECISIÓN
        # ----------------------------------------------------

        (
            decision,
            score,
            reasons
        ) = calculate_decision(
            image_stats
        )

        # ----------------------------------------------------
        # ROW
        # ----------------------------------------------------

        rows.append({

            "split": split_name,

            "image": str(image_path),

            "label": str(label_path),

            "decision": decision,

            "score": score,

            "reasons": reasons,

            "objects":
                image_stats["objects"],

            "tiny16":
                image_stats["tiny16"],

            "tiny32":
                image_stats["tiny32"],

            "tiny64":
                image_stats["tiny64"],

            "partial_outside":
                image_stats[
                    "partial_outside"
                ],

            "fully_outside":
                image_stats[
                    "fully_outside"
                ],

            "near_border":
                image_stats[
                    "near_border"
                ],

            "duplicates":
                image_stats["duplicates"],

            "invalid_labels":
                image_stats[
                    "invalid_labels"
                ],

            "invalid_coordinates":
                image_stats[
                    "invalid_coordinates"
                ],

            "invalid_bboxes":
                image_stats[
                    "invalid_bboxes"
                ],

            "invalid_classes":
                image_stats[
                    "invalid_classes"
                ],

            "label_exists":
                image_stats[
                    "label_exists"
                ],

            "width":
                image_width,

            "height":
                image_height,

        })

        # ----------------------------------------------------
        # PROGRESO
        # ----------------------------------------------------

        if index % 1000 == 0:

            print(
                f"Procesadas: "
                f"{index:,}/"
                f"{len(image_files):,}"
            )

    return rows


# ============================================================
# COPIAR EJEMPLOS
# ============================================================

def copy_examples(
    rows,
    output_dir,
    decision,
    limit
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = [
        r
        for r in rows
        if r["decision"] == decision
    ]

    selected.sort(
        key=lambda x:
        float(x["score"]),
        reverse=True
    )

    copied = 0

    for row in selected:

        if copied >= limit:
            break

        source = Path(
            row["image"]
        )

        if not source.exists():
            continue

        destination = (
            output_dir
            / f"{copied + 1:03d}_"
            f"{source.name}"
        )

        try:

            shutil.copy2(
                source,
                destination
            )

            copied += 1

        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n# SAR YOLO26 - DATASET AUDIT V6\n"
    )

    print(
        "Dataset:"
    )

    print(
        DATASET_ROOT
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\n"
        + "-" * 70
    )

    if not DATASET_ROOT.exists():

        print(
            "\n[ERROR] No existe DATASET_ROOT:"
        )

        print(
            DATASET_ROOT
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    reports_dir = (
        OUTPUT_ROOT
        / "reports"
    )

    examples_dir = (
        OUTPUT_ROOT
        / "examples"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    examples_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # ANALIZAR TODO
    # ========================================================

    all_rows = []

    for split in SPLITS:

        split_dir = (
            DATASET_ROOT
            / split
        )

        if not split_dir.exists():

            print(
                f"\n[INFO] Split no encontrado: "
                f"{split}"
            )

            continue

        rows = analyze_split(
            split,
            split_dir
        )

        all_rows.extend(
            rows
        )

    # ========================================================
    # RESUMEN
    # ========================================================

    counter = Counter(
        r["decision"]
        for r in all_rows
    )

    total_images = len(
        all_rows
    )

    total_objects = sum(
        int(r["objects"])
        for r in all_rows
    )

    total_tiny16 = sum(
        int(r["tiny16"])
        for r in all_rows
    )

    total_tiny32 = sum(
        int(r["tiny32"])
        for r in all_rows
    )

    total_tiny64 = sum(
        int(r["tiny64"])
        for r in all_rows
    )

    total_partial = sum(
        int(r["partial_outside"])
        for r in all_rows
    )

    total_full = sum(
        int(r["fully_outside"])
        for r in all_rows
    )

    total_border = sum(
        int(r["near_border"])
        for r in all_rows
    )

    total_duplicates = sum(
        int(r["duplicates"])
        for r in all_rows
    )

    total_invalid_labels = sum(
        int(r["invalid_labels"])
        for r in all_rows
    )

    total_invalid_coordinates = sum(
        int(r["invalid_coordinates"])
        for r in all_rows
    )

    total_invalid_bboxes = sum(
        int(r["invalid_bboxes"])
        for r in all_rows
    )

    total_invalid_classes = sum(
        int(r["invalid_classes"])
        for r in all_rows
    )

    # ========================================================
    # CSV PRINCIPAL
    # ========================================================

    write_csv(
        reports_dir
        / "image_decisions.csv",
        all_rows
    )

    # ========================================================
    # REVIEW
    # ========================================================

    review_rows = [
        r
        for r in all_rows
        if r["decision"] == REVIEW
    ]

    review_rows.sort(
        key=lambda x:
        float(x["score"]),
        reverse=True
    )

    write_csv(
        reports_dir
        / "review_images.csv",
        review_rows
    )

    # ========================================================
    # EXCLUDE
    # ========================================================

    exclude_rows = [
        r
        for r in all_rows
        if r["decision"] == EXCLUDE
    ]

    exclude_rows.sort(
        key=lambda x:
        float(x["score"]),
        reverse=True
    )

    write_csv(
        reports_dir
        / "exclude_candidates.csv",
        exclude_rows
    )

    # ========================================================
    # CRITICAL
    # ========================================================

    critical_rows = [
        r
        for r in all_rows
        if r["decision"] == CRITICAL
    ]

    critical_rows.sort(
        key=lambda x:
        float(x["score"]),
        reverse=True
    )

    write_csv(
        reports_dir
        / "critical_images.csv",
        critical_rows
    )

    # ========================================================
    # TINY
    # ========================================================

    tiny_rows = [
        r
        for r in all_rows
        if int(r["tiny16"]) > 0
    ]

    tiny_rows.sort(
        key=lambda x:
        int(x["tiny16"]),
        reverse=True
    )

    write_csv(
        reports_dir
        / "tiny_objects_images.csv",
        tiny_rows
    )

    # ========================================================
    # BORDER
    # ========================================================

    border_rows = [
        r
        for r in all_rows
        if int(r["near_border"]) > 0
    ]

    border_rows.sort(
        key=lambda x:
        int(x["near_border"]),
        reverse=True
    )

    write_csv(
        reports_dir
        / "border_images.csv",
        border_rows
    )

    # ========================================================
    # CROWDED
    # ========================================================

    crowded_rows = [
        r
        for r in all_rows
        if int(r["objects"])
        >= CROWDED_200
    ]

    crowded_rows.sort(
        key=lambda x:
        int(x["objects"]),
        reverse=True
    )

    write_csv(
        reports_dir
        / "crowded_images.csv",
        crowded_rows
    )

    # ========================================================
    # TOP REVIEW
    # ========================================================

    write_csv(
        reports_dir
        / "top_50_review.csv",
        review_rows[:50]
    )

    # ========================================================
    # EJEMPLOS
    # ========================================================

    print(
        "\nGenerando ejemplos..."
    )

    copy_examples(
        all_rows,
        examples_dir / "review",
        REVIEW,
        MAX_REVIEW_EXAMPLES
    )

    copy_examples(
        all_rows,
        examples_dir / "exclude",
        EXCLUDE,
        MAX_EXCLUDE_EXAMPLES
    )

    # ========================================================
    # EJEMPLOS TINY
    # ========================================================

    tiny_examples_dir = (
        examples_dir
        / "tiny"
    )

    tiny_examples_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, row in enumerate(
        tiny_rows[:MAX_TINY_EXAMPLES],
        start=1
    ):

        source = Path(
            row["image"]
        )

        if source.exists():

            try:

                shutil.copy2(
                    source,
                    tiny_examples_dir
                    / f"{i:03d}_"
                    f"{source.name}"
                )

            except Exception:
                pass

    # ========================================================
    # EJEMPLOS BORDE
    # ========================================================

    border_examples_dir = (
        examples_dir
        / "border"
    )

    border_examples_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, row in enumerate(
        border_rows[:MAX_BORDER_EXAMPLES],
        start=1
    ):

        source = Path(
            row["image"]
        )

        if source.exists():

            try:

                shutil.copy2(
                    source,
                    border_examples_dir
                    / f"{i:03d}_"
                    f"{source.name}"
                )

            except Exception:
                pass

    # ========================================================
    # EJEMPLOS CROWDED
    # ========================================================

    crowded_examples_dir = (
        examples_dir
        / "crowded"
    )

    crowded_examples_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, row in enumerate(
        crowded_rows[:MAX_CROWDED_EXAMPLES],
        start=1
    ):

        source = Path(
            row["image"]
        )

        if source.exists():

            try:

                shutil.copy2(
                    source,
                    crowded_examples_dir
                    / f"{i:03d}_"
                    f"{source.name}"
                )

            except Exception:
                pass

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary_rows = [

        {
            "metric": "images",
            "value": total_images,
        },

        {
            "metric": "objects",
            "value": total_objects,
        },

        {
            "metric": "objects_per_image",
            "value":
                round(
                    total_objects
                    / total_images,
                    2
                )
                if total_images
                else 0,
        },

        {
            "metric": "KEEP",
            "value":
                counter[KEEP],
        },

        {
            "metric": "REVIEW",
            "value":
                counter[REVIEW],
        },

        {
            "metric": "EXCLUDE_CANDIDATE",
            "value":
                counter[EXCLUDE],
        },

        {
            "metric": "CRITICAL",
            "value":
                counter[CRITICAL],
        },

        {
            "metric": "tiny16",
            "value": total_tiny16,
        },

        {
            "metric": "tiny32",
            "value": total_tiny32,
        },

        {
            "metric": "tiny64",
            "value": total_tiny64,
        },

        {
            "metric": "partial_bbox",
            "value": total_partial,
        },

        {
            "metric": "full_bbox_outside",
            "value": total_full,
        },

        {
            "metric": "border_objects",
            "value": total_border,
        },

        {
            "metric": "duplicates",
            "value": total_duplicates,
        },

        {
            "metric": "invalid_labels",
            "value":
                total_invalid_labels,
        },

        {
            "metric": "invalid_coordinates",
            "value":
                total_invalid_coordinates,
        },

        {
            "metric": "invalid_bboxes",
            "value":
                total_invalid_bboxes,
        },

        {
            "metric": "invalid_classes",
            "value":
                total_invalid_classes,
        },

    ]

    write_csv(
        reports_dir
        / "audit_summary.csv",
        summary_rows
    )

    # ========================================================
    # INFORME TXT
    # ========================================================

    report_path = (
        reports_dir
        / "AUDIT_SUMMARY.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V6\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Dataset:\n"
            f"{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Imágenes: {total_images:,}\n"
        )

        f.write(
            f"Objetos: {total_objects:,}\n"
        )

        if total_images:

            f.write(
                "Objetos/imagen: "
                f"{total_objects / total_images:.2f}\n"
            )

        f.write(
            "\nDECISIONES\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for decision in [
            KEEP,
            REVIEW,
            EXCLUDE,
            CRITICAL,
        ]:

            count = counter[decision]

            percentage = (
                count
                / total_images
                * 100
                if total_images
                else 0
            )

            f.write(
                f"{decision:<20}"
                f"{count:>8,}"
                f" ({percentage:6.2f} %)\n"
            )

        f.write(
            "\nANOMALÍAS\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Objetos <16 px²: "
            f"{total_tiny16:,}\n"
        )

        f.write(
            f"Objetos <32 px²: "
            f"{total_tiny32:,}\n"
        )

        f.write(
            f"Objetos <64 px²: "
            f"{total_tiny64:,}\n"
        )

        f.write(
            f"BBox parcialmente fuera: "
            f"{total_partial:,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{total_full:,}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{total_border:,}\n"
        )

        f.write(
            f"Duplicados: "
            f"{total_duplicates:,}\n"
        )

        f.write(
            "\nINTEGRIDAD\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Labels inválidos: "
            f"{total_invalid_labels:,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{total_invalid_coordinates:,}\n"
        )

        f.write(
            f"BBoxes inválidas: "
            f"{total_invalid_bboxes:,}\n"
        )

        f.write(
            f"Clases inválidas: "
            f"{total_invalid_classes:,}\n"
        )

        f.write(
            "\nTOP REVIEW\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for i, row in enumerate(
            review_rows[:50],
            start=1
        ):

            f.write(
                f"{i:02d}. "
                f"{row['decision']} "
                f"score={row['score']} "
                f"objects={row['objects']} "
                f"tiny16={row['tiny16']} "
                f"partial={row['partial_outside']} "
                f"border={row['near_border']} "
                f"reasons={row['reasons']}\n"
            )

            f.write(
                f"    {row['image']}\n"
            )

        f.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Este script SOLO diagnostica.\n"
        )

        f.write(
            "No elimina ni modifica imágenes o labels.\n"
        )

    # ========================================================
    # CONSOLA
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESULTADO AUDIT V6"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImágenes:              "
        f"{total_images:,}"
    )

    print(
        f"Objetos:               "
        f"{total_objects:,}"
    )

    if total_images:

        print(
            f"Objetos/imagen:        "
            f"{total_objects / total_images:.2f}"
        )

    print(
        "\nDECISIONES"
    )

    print(
        f"KEEP:                  "
        f"{counter[KEEP]:,}"
    )

    print(
        f"REVIEW:                "
        f"{counter[REVIEW]:,}"
    )

    print(
        f"EXCLUDE_CANDIDATE:     "
        f"{counter[EXCLUDE]:,}"
    )

    print(
        f"CRITICAL:              "
        f"{counter[CRITICAL]:,}"
    )

    print(
        "\nOBJETOS PEQUEÑOS"
    )

    print(
        f"<16 px²:               "
        f"{total_tiny16:,}"
    )

    print(
        f"<32 px²:               "
        f"{total_tiny32:,}"
    )

    print(
        f"<64 px²:               "
        f"{total_tiny64:,}"
    )

    print(
        "\nBORDES"
    )

    print(
        f"BBox parcialmente fuera: "
        f"{total_partial:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{total_full:,}"
    )

    print(
        f"Cerca del borde:       "
        f"{total_border:,}"
    )

    print(
        "\nINTEGRIDAD"
    )

    print(
        f"Labels inválidos:      "
        f"{total_invalid_labels:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{total_invalid_coordinates:,}"
    )

    print(
        f"BBoxes inválidas:      "
        f"{total_invalid_bboxes:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{total_invalid_classes:,}"
    )

    print(
        f"Duplicados:            "
        f"{total_duplicates:,}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOP 20 REVIEW"
    )

    print(
        "=" * 70
    )

    for i, row in enumerate(
        review_rows[:20],
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{row['decision']:<18}"
            f"score={row['score']:>5} "
            f"objects={row['objects']:>4} "
            f"tiny16={row['tiny16']:>3} "
            f"partial={row['partial_outside']:>3} "
            f"border={row['near_border']:>3}"
        )

        print(
            f"    reasons: "
            f"{row['reasons']}"
        )

        print(
            f"    {row['image']}"
        )

    print(
        "\nReports:"
    )

    print(
        reports_dir
    )

    print(
        "\nExamples:"
    )

    print(
        examples_dir
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes o labels."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()