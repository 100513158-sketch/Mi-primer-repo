from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import hashlib


# ============================================================
# SAR YOLO26 - DATASET AUDIT V4
#
# OBJETIVO:
#   Auditoría semántica y estructural avanzada del dataset.
#
# IMPORTANTE:
#   ESTE SCRIPT ES 100 % NO DESTRUCTIVO.
#   NO modifica imágenes.
#   NO modifica labels.
#   NO elimina archivos.
#   NO corrige anotaciones.
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
    r"\baseline\evaluation\dataset_analysis\audit_v4"
)


SPLITS = [
    "train",
    "val",
    "test_dev",
]


CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}


IMAGE_EXTENSIONS = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
)


# ============================================================
# UMBRALES
# ============================================================

# Área de bbox en píxeles²
TINY_THRESHOLDS = [
    16,
    32,
    64,
    100,
]


# Objetos por imagen
CROWDED_THRESHOLDS = [
    100,
    150,
    200,
    300,
    400,
    500,
]


# Porcentaje de objetos pequeños que dispara revisión
SMALL_OBJECT_RATIO_REVIEW = 0.25
SMALL_OBJECT_RATIO_WARNING = 0.50
SMALL_OBJECT_RATIO_CRITICAL = 0.75


# Porcentaje de bboxes parcialmente fuera
PARTIAL_BBOX_RATIO_REVIEW = 0.05
PARTIAL_BBOX_RATIO_WARNING = 0.15
PARTIAL_BBOX_RATIO_CRITICAL = 0.30


# Cantidad absoluta de objetos parcialmente fuera
PARTIAL_BBOX_COUNT_WARNING = 5
PARTIAL_BBOX_COUNT_CRITICAL = 15


# Objetos por imagen
CROWDED_REVIEW = 100
CROWDED_WARNING = 200
CROWDED_CRITICAL = 300


# Objetos extremadamente pequeños
EXTREME_TINY_REVIEW = 3
EXTREME_TINY_WARNING = 10
EXTREME_TINY_CRITICAL = 25


# Objetos cerca del borde
NEAR_BORDER_RATIO_WARNING = 0.20
NEAR_BORDER_RATIO_CRITICAL = 0.40


# Diferencia extrema entre clases
CLASS_RATIO_EXTREME = 0.95


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def bbox_area_pixels(x_norm, y_norm, w_norm, h_norm, img_w, img_h):
    """
    Convierte bbox YOLO normalizada a área en píxeles².
    """

    width_px = abs(w_norm) * img_w
    height_px = abs(h_norm) * img_h

    return width_px * height_px


def bbox_dimensions_pixels(w_norm, h_norm, img_w, img_h):
    width_px = abs(w_norm) * img_w
    height_px = abs(h_norm) * img_h

    return width_px, height_px


def calculate_bbox_bounds(
    x_center,
    y_center,
    width,
    height
):
    """
    Devuelve:

        x1
        y1
        x2
        y2

    en coordenadas normalizadas.
    """

    x1 = x_center - width / 2
    y1 = y_center - height / 2

    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return x1, y1, x2, y2


def classify_bbox_position(
    x1,
    y1,
    x2,
    y2
):
    """
    Clasifica la posición de la bbox.

    normal
    partial
    outside

    Además calcula cuánto se sale.
    """

    outside_left = max(0.0, -x1)
    outside_top = max(0.0, -y1)
    outside_right = max(0.0, x2 - 1.0)
    outside_bottom = max(0.0, y2 - 1.0)

    outside_amount = (
        outside_left
        + outside_top
        + outside_right
        + outside_bottom
    )

    completely_outside = (
        x2 <= 0
        or y2 <= 0
        or x1 >= 1
        or y1 >= 1
    )

    partially_outside = (
        not completely_outside
        and (
            x1 < 0
            or y1 < 0
            or x2 > 1
            or y2 > 1
        )
    )

    if completely_outside:
        position = "outside"

    elif partially_outside:
        position = "partial"

    else:
        position = "inside"

    return (
        position,
        outside_amount,
        outside_left,
        outside_top,
        outside_right,
        outside_bottom,
    )


def is_near_border(
    x1,
    y1,
    x2,
    y2,
    margin=0.02
):
    """
    Determina si la bbox está a <=2% del borde.
    """

    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= 1.0 - margin
        or y2 >= 1.0 - margin
    )


def calculate_class_ratio(
    persons,
    vehicles
):
    total = persons + vehicles

    if total == 0:
        return 0.0

    return max(persons, vehicles) / total


def calculate_risk(
    total_objects,
    tiny_16,
    tiny_32,
    partial_bbox,
    near_border,
    duplicate_labels,
    invalid_labels,
    invalid_coordinates,
    invalid_bboxes,
):
    """
    Clasificación conservadora.

    OK
    REVIEW
    WARNING
    CRITICAL
    """

    score = 0

    # --------------------------------------------------------
    # Errores estructurales
    # --------------------------------------------------------

    if invalid_labels > 0:
        score += 10

    if invalid_coordinates > 0:
        score += 10

    if invalid_bboxes > 0:
        score += 10

    # --------------------------------------------------------
    # Duplicados
    # --------------------------------------------------------

    if duplicate_labels > 0:
        score += 5

    # --------------------------------------------------------
    # Tiny objects
    # --------------------------------------------------------

    if total_objects > 0:

        tiny_ratio = tiny_16 / total_objects

        if tiny_ratio >= SMALL_OBJECT_RATIO_CRITICAL:
            score += 8

        elif tiny_ratio >= SMALL_OBJECT_RATIO_WARNING:
            score += 5

        elif tiny_ratio >= SMALL_OBJECT_RATIO_REVIEW:
            score += 2

    if tiny_16 >= EXTREME_TINY_CRITICAL:
        score += 5

    elif tiny_16 >= EXTREME_TINY_WARNING:
        score += 3

    elif tiny_16 >= EXTREME_TINY_REVIEW:
        score += 1

    # --------------------------------------------------------
    # Partial BBoxes
    # --------------------------------------------------------

    if total_objects > 0:

        partial_ratio = (
            partial_bbox / total_objects
        )

        if partial_ratio >= PARTIAL_BBOX_RATIO_CRITICAL:
            score += 8

        elif partial_ratio >= PARTIAL_BBOX_RATIO_WARNING:
            score += 5

        elif partial_ratio >= PARTIAL_BBOX_RATIO_REVIEW:
            score += 2

    if partial_bbox >= PARTIAL_BBOX_COUNT_CRITICAL:
        score += 5

    elif partial_bbox >= PARTIAL_BBOX_COUNT_WARNING:
        score += 2

    # --------------------------------------------------------
    # Crowded
    # --------------------------------------------------------

    if total_objects >= CROWDED_CRITICAL:
        score += 5

    elif total_objects >= CROWDED_WARNING:
        score += 3

    elif total_objects >= CROWDED_REVIEW:
        score += 1

    # --------------------------------------------------------
    # Near border
    # --------------------------------------------------------

    if total_objects > 0:

        border_ratio = (
            near_border / total_objects
        )

        if border_ratio >= NEAR_BORDER_RATIO_CRITICAL:
            score += 4

        elif border_ratio >= NEAR_BORDER_RATIO_WARNING:
            score += 2

    # --------------------------------------------------------
    # Clasificación
    # --------------------------------------------------------

    if score >= 15:
        return "CRITICAL"

    if score >= 8:
        return "WARNING"

    if score >= 3:
        return "REVIEW"

    return "OK"


def calculate_annotation_signature(
    class_id,
    x,
    y,
    w,
    h
):
    """
    Firma exacta de una anotación.
    """

    return (
        class_id,
        round(x, 8),
        round(y, 8),
        round(w, 8),
        round(h, 8),
    )


def calculate_file_hash(path):
    """
    Hash MD5 de un fichero.

    Se utiliza únicamente para diagnóstico.
    """

    md5 = hashlib.md5()

    try:

        with path.open("rb") as f:

            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):
                md5.update(chunk)

        return md5.hexdigest()

    except Exception:

        return ""


# ============================================================
# OBTENER DIMENSIONES DE IMAGEN
# ============================================================

def get_image_size(image_path):
    """
    Obtiene dimensiones usando PIL si está disponible.
    """

    try:

        from PIL import Image

        with Image.open(image_path) as img:

            return img.width, img.height

    except Exception:

        return None, None


# ============================================================
# ANALIZAR IMAGEN
# ============================================================

def analyze_image(
    image_path,
    labels_dir,
    split
):

    label_path = (
        labels_dir
        / f"{image_path.stem}.txt"
    )

    img_w, img_h = get_image_size(
        image_path
    )

    result = {

        "split": split,

        "image": str(image_path),

        "label": str(label_path),

        "image_width": img_w or 0,

        "image_height": img_h or 0,

        "label_exists": label_path.exists(),

        "objects": 0,

        "persons": 0,

        "vehicles": 0,

        "tiny_16": 0,

        "tiny_32": 0,

        "tiny_64": 0,

        "tiny_100": 0,

        "partial_bbox": 0,

        "outside_bbox": 0,

        "near_border": 0,

        "duplicate_labels": 0,

        "invalid_labels": 0,

        "invalid_coordinates": 0,

        "invalid_bboxes": 0,

        "invalid_classes": 0,

        "max_bbox_area": 0.0,

        "min_bbox_area": 0.0,

        "mean_bbox_area": 0.0,

        "person_area_mean": 0.0,

        "vehicle_area_mean": 0.0,

        "person_ratio": 0.0,

        "vehicle_ratio": 0.0,

        "class_dominance": 0.0,

        "risk": "OK",

        "risk_score": 0,

        "notes": "",

    }

    # --------------------------------------------------------
    # Imagen inexistente/corrupta
    # --------------------------------------------------------

    if (
        img_w is None
        or img_h is None
        or img_w <= 0
        or img_h <= 0
    ):

        result["notes"] = "imagen_corrupta"

        return result, []

    # --------------------------------------------------------
    # Label inexistente
    # --------------------------------------------------------

    if not label_path.exists():

        result["notes"] = "label_inexistente"

        return result, []

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception as exc:

        result["invalid_labels"] = 1

        result["notes"] = (
            f"error_lectura_label:{exc}"
        )

        return result, []

    annotation_signatures = Counter()

    annotation_rows = []

    person_areas = []
    vehicle_areas = []

    all_areas = []

    # --------------------------------------------------------
    # Procesar labels
    # --------------------------------------------------------

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # ----------------------------------------------------
        # Estructura YOLO
        # ----------------------------------------------------

        if len(parts) != 5:

            result["invalid_labels"] += 1

            continue

        class_id = safe_int(parts[0])
        x = safe_float(parts[1])
        y = safe_float(parts[2])
        w = safe_float(parts[3])
        h = safe_float(parts[4])

        if (
            class_id is None
            or x is None
            or y is None
            or w is None
            or h is None
        ):

            result["invalid_labels"] += 1

            continue

        # ----------------------------------------------------
        # Clase
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            result["invalid_classes"] += 1

        # ----------------------------------------------------
        # Coordenadas
        # ----------------------------------------------------

        coordinate_invalid = (
            x < 0
            or x > 1
            or y < 0
            or y > 1
        )

        if coordinate_invalid:

            result["invalid_coordinates"] += 1

        # ----------------------------------------------------
        # BBox
        # ----------------------------------------------------

        bbox_invalid = (
            w <= 0
            or h <= 0
        )

        if bbox_invalid:

            result["invalid_bboxes"] += 1

            continue

        # ----------------------------------------------------
        # Firma duplicada
        # ----------------------------------------------------

        signature = (
            calculate_annotation_signature(
                class_id,
                x,
                y,
                w,
                h
            )
        )

        annotation_signatures[
            signature
        ] += 1

        # ----------------------------------------------------
        # Coordenadas bbox
        # ----------------------------------------------------

        (
            x1,
            y1,
            x2,
            y2
        ) = calculate_bbox_bounds(
            x,
            y,
            w,
            h
        )

        (
            position,
            outside_amount,
            outside_left,
            outside_top,
            outside_right,
            outside_bottom,
        ) = classify_bbox_position(
            x1,
            y1,
            x2,
            y2
        )

        if position == "partial":

            result["partial_bbox"] += 1

        elif position == "outside":

            result["outside_bbox"] += 1

        # ----------------------------------------------------
        # Cerca del borde
        # ----------------------------------------------------

        if is_near_border(
            x1,
            y1,
            x2,
            y2
        ):

            result["near_border"] += 1

        # ----------------------------------------------------
        # Área
        # ----------------------------------------------------

        area = bbox_area_pixels(
            x,
            y,
            w,
            h,
            img_w,
            img_h
        )

        width_px, height_px = (
            bbox_dimensions_pixels(
                w,
                h,
                img_w,
                img_h
            )
        )

        all_areas.append(area)

        # ----------------------------------------------------
        # Tiny
        # ----------------------------------------------------

        if area < 16:
            result["tiny_16"] += 1

        if area < 32:
            result["tiny_32"] += 1

        if area < 64:
            result["tiny_64"] += 1

        if area < 100:
            result["tiny_100"] += 1

        # ----------------------------------------------------
        # Clases
        # ----------------------------------------------------

        if class_id == 0:

            result["persons"] += 1

            person_areas.append(area)

        elif class_id == 1:

            result["vehicles"] += 1

            vehicle_areas.append(area)

        result["objects"] += 1

        # ----------------------------------------------------
        # Annotation detail
        # ----------------------------------------------------

        annotation_rows.append({

            "split": split,

            "image": str(image_path),

            "label": str(label_path),

            "line": line_number,

            "class_id": class_id,

            "class_name": CLASS_NAMES.get(
                class_id,
                "INVALID"
            ),

            "x": x,

            "y": y,

            "w": w,

            "h": h,

            "width_px": width_px,

            "height_px": height_px,

            "area_px2": area,

            "bbox_position": position,

            "outside_amount": outside_amount,

            "outside_left": outside_left,

            "outside_top": outside_top,

            "outside_right": outside_right,

            "outside_bottom": outside_bottom,

            "near_border": (
                1
                if is_near_border(
                    x1,
                    y1,
                    x2,
                    y2
                )
                else 0
            ),

            "tiny_16": (
                1 if area < 16 else 0
            ),

            "tiny_32": (
                1 if area < 32 else 0
            ),

            "tiny_64": (
                1 if area < 64 else 0
            ),

            "tiny_100": (
                1 if area < 100 else 0
            ),

        })

    # --------------------------------------------------------
    # Duplicados
    # --------------------------------------------------------

    duplicate_count = 0

    for signature, count in (
        annotation_signatures.items()
    ):

        if count > 1:

            duplicate_count += count - 1

    result["duplicate_labels"] = (
        duplicate_count
    )

    # --------------------------------------------------------
    # Estadísticas
    # --------------------------------------------------------

    if all_areas:

        result["max_bbox_area"] = max(
            all_areas
        )

        result["min_bbox_area"] = min(
            all_areas
        )

        result["mean_bbox_area"] = (
            sum(all_areas)
            / len(all_areas)
        )

    if person_areas:

        result["person_area_mean"] = (
            sum(person_areas)
            / len(person_areas)
        )

    if vehicle_areas:

        result["vehicle_area_mean"] = (
            sum(vehicle_areas)
            / len(vehicle_areas)
        )

    total_classes = (
        result["persons"]
        + result["vehicles"]
    )

    if total_classes > 0:

        result["person_ratio"] = (
            result["persons"]
            / total_classes
        )

        result["vehicle_ratio"] = (
            result["vehicles"]
            / total_classes
        )

        result["class_dominance"] = (
            calculate_class_ratio(
                result["persons"],
                result["vehicles"]
            )
        )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    score = 0

    total = result["objects"]

    # errores
    score += (
        result["invalid_labels"]
        * 10
    )

    score += (
        result["invalid_coordinates"]
        * 10
    )

    score += (
        result["invalid_bboxes"]
        * 10
    )

    score += (
        result["invalid_classes"]
        * 10
    )

    # duplicados
    if result["duplicate_labels"] > 0:
        score += 5

    # tiny
    if total > 0:

        tiny_ratio = (
            result["tiny_16"]
            / total
        )

        if (
            tiny_ratio
            >= SMALL_OBJECT_RATIO_CRITICAL
        ):

            score += 8

        elif (
            tiny_ratio
            >= SMALL_OBJECT_RATIO_WARNING
        ):

            score += 5

        elif (
            tiny_ratio
            >= SMALL_OBJECT_RATIO_REVIEW
        ):

            score += 2

    # partial
    if total > 0:

        partial_ratio = (
            result["partial_bbox"]
            / total
        )

        if (
            partial_ratio
            >= PARTIAL_BBOX_RATIO_CRITICAL
        ):

            score += 8

        elif (
            partial_ratio
            >= PARTIAL_BBOX_RATIO_WARNING
        ):

            score += 5

        elif (
            partial_ratio
            >= PARTIAL_BBOX_RATIO_REVIEW
        ):

            score += 2

    if (
        result["partial_bbox"]
        >= PARTIAL_BBOX_COUNT_CRITICAL
    ):

        score += 5

    elif (
        result["partial_bbox"]
        >= PARTIAL_BBOX_COUNT_WARNING
    ):

        score += 2

    # crowded
    if total >= CROWDED_CRITICAL:
        score += 5

    elif total >= CROWDED_WARNING:
        score += 3

    elif total >= CROWDED_REVIEW:
        score += 1

    # extreme tiny
    if (
        result["tiny_16"]
        >= EXTREME_TINY_CRITICAL
    ):

        score += 5

    elif (
        result["tiny_16"]
        >= EXTREME_TINY_WARNING
    ):

        score += 3

    elif (
        result["tiny_16"]
        >= EXTREME_TINY_REVIEW
    ):

        score += 1

    # near border
    if total > 0:

        border_ratio = (
            result["near_border"]
            / total
        )

        if (
            border_ratio
            >= NEAR_BORDER_RATIO_CRITICAL
        ):

            score += 4

        elif (
            border_ratio
            >= NEAR_BORDER_RATIO_WARNING
        ):

            score += 2

    result["risk_score"] = score

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if score >= 15:

        result["risk"] = "CRITICAL"

    elif score >= 8:

        result["risk"] = "WARNING"

    elif score >= 3:

        result["risk"] = "REVIEW"

    else:

        result["risk"] = "OK"

    # --------------------------------------------------------
    # Notas
    # --------------------------------------------------------

    notes = []

    if result["tiny_16"] > 0:
        notes.append("tiny")

    if result["partial_bbox"] > 0:
        notes.append("partial_bbox")

    if result["near_border"] > 0:
        notes.append("near_border")

    if result["duplicate_labels"] > 0:
        notes.append("duplicates")

    if total >= CROWDED_REVIEW:
        notes.append("crowded")

    if result["invalid_labels"] > 0:
        notes.append("invalid_labels")

    if result["invalid_coordinates"] > 0:
        notes.append("invalid_coordinates")

    if result["invalid_bboxes"] > 0:
        notes.append("invalid_bbox")

    result["notes"] = ",".join(
        notes
    )

    return result, annotation_rows


# ============================================================
# ANALIZAR SPLIT
# ============================================================

def analyze_split(split):

    split_dir = (
        DATASET_ROOT
        / split
    )

    images_dir = (
        split_dir
        / "images"
    )

    labels_dir = (
        split_dir
        / "labels"
    )

    if not split_dir.exists():

        print(
            f"[INFO] Split no encontrado: "
            f"{split}"
        )

        return [], []

    if not images_dir.exists():

        print(
            f"[WARN] No existe: "
            f"{images_dir}"
        )

        return [], []

    if not labels_dir.exists():

        print(
            f"[WARN] No existe: "
            f"{labels_dir}"
        )

        return [], []

    image_files = []

    for ext in IMAGE_EXTENSIONS:

        image_files.extend(
            images_dir.rglob(ext)
        )

    image_files = sorted(
        set(image_files)
    )

    print(
        f"\n## Analizando: {split}"
    )

    print(
        f"Imágenes encontradas: "
        f"{len(image_files):,}"
    )

    image_results = []
    annotation_results = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        result, annotations = (
            analyze_image(
                image_path,
                labels_dir,
                split
            )
        )

        image_results.append(result)

        annotation_results.extend(
            annotations
        )

        if index % 1000 == 0:

            print(
                f"Procesadas: "
                f"{index:,}/"
                f"{len(image_files):,}"
            )

    return (
        image_results,
        annotation_results
    )


# ============================================================
# CSV
# ============================================================

def write_csv(
    path,
    rows
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:

        return

    fieldnames = list(
        rows[0].keys()
    )

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


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAR YOLO26 - DATASET AUDIT V4"
    )

    print(
        "=" * 70
    )

    print(
        "\nDataset:"
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

    # --------------------------------------------------------
    # Validar dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print(
            "\n[ERROR] No existe DATASET_ROOT:"
        )

        print(
            DATASET_ROOT
        )

        return

    # --------------------------------------------------------
    # Crear carpetas
    # --------------------------------------------------------

    reports_dir = (
        OUTPUT_ROOT
        / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Analizar splits
    # --------------------------------------------------------

    all_images = []
    all_annotations = []

    for split in SPLITS:

        (
            image_results,
            annotation_results
        ) = analyze_split(split)

        all_images.extend(
            image_results
        )

        all_annotations.extend(
            annotation_results
        )

    # --------------------------------------------------------
    # Estadísticas generales
    # --------------------------------------------------------

    total_images = len(
        all_images
    )

    total_objects = sum(
        row["objects"]
        for row in all_images
    )

    total_persons = sum(
        row["persons"]
        for row in all_images
    )

    total_vehicles = sum(
        row["vehicles"]
        for row in all_images
    )

    tiny_16 = sum(
        row["tiny_16"]
        for row in all_images
    )

    tiny_32 = sum(
        row["tiny_32"]
        for row in all_images
    )

    tiny_64 = sum(
        row["tiny_64"]
        for row in all_images
    )

    tiny_100 = sum(
        row["tiny_100"]
        for row in all_images
    )

    partial_bbox = sum(
        row["partial_bbox"]
        for row in all_images
    )

    outside_bbox = sum(
        row["outside_bbox"]
        for row in all_images
    )

    near_border = sum(
        row["near_border"]
        for row in all_images
    )

    duplicates = sum(
        row["duplicate_labels"]
        for row in all_images
    )

    invalid_labels = sum(
        row["invalid_labels"]
        for row in all_images
    )

    invalid_coordinates = sum(
        row["invalid_coordinates"]
        for row in all_images
    )

    invalid_bboxes = sum(
        row["invalid_bboxes"]
        for row in all_images
    )

    invalid_classes = sum(
        row["invalid_classes"]
        for row in all_images
    )

    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    risk_counter = Counter(
        row["risk"]
        for row in all_images
    )

    # --------------------------------------------------------
    # Crowded
    # --------------------------------------------------------

    crowded_counts = {}

    for threshold in CROWDED_THRESHOLDS:

        crowded_counts[
            threshold
        ] = sum(
            1
            for row in all_images
            if row["objects"]
            >= threshold
        )

    # --------------------------------------------------------
    # Clase por split
    # --------------------------------------------------------

    split_summary = []

    for split in SPLITS:

        rows = [
            row
            for row in all_images
            if row["split"] == split
        ]

        if not rows:
            continue

        split_summary.append({

            "split": split,

            "images": len(rows),

            "persons": sum(
                row["persons"]
                for row in rows
            ),

            "vehicles": sum(
                row["vehicles"]
                for row in rows
            ),

            "objects": sum(
                row["objects"]
                for row in rows
            ),

            "tiny_16": sum(
                row["tiny_16"]
                for row in rows
            ),

            "tiny_32": sum(
                row["tiny_32"]
                for row in rows
            ),

            "tiny_64": sum(
                row["tiny_64"]
                for row in rows
            ),

            "partial_bbox": sum(
                row["partial_bbox"]
                for row in rows
            ),

            "near_border": sum(
                row["near_border"]
                for row in rows
            ),

            "duplicates": sum(
                row["duplicate_labels"]
                for row in rows
            ),

            "crowded_100": sum(
                1
                for row in rows
                if row["objects"] >= 100
            ),

            "crowded_200": sum(
                1
                for row in rows
                if row["objects"] >= 200
            ),

            "crowded_300": sum(
                1
                for row in rows
                if row["objects"] >= 300
            ),

        })

    # --------------------------------------------------------
    # Clases y tamaños
    # --------------------------------------------------------

    class_area_stats = []

    for class_id, class_name in (
        CLASS_NAMES.items()
    ):

        areas = [
            row["area_px2"]
            for row in all_annotations
            if row["class_id"]
            == class_id
        ]

        if not areas:

            continue

        class_area_stats.append({

            "class_id": class_id,

            "class_name": class_name,

            "objects": len(areas),

            "mean_area_px2": (
                sum(areas)
                / len(areas)
            ),

            "min_area_px2": min(
                areas
            ),

            "max_area_px2": max(
                areas
            ),

            "below_16": sum(
                1
                for x in areas
                if x < 16
            ),

            "below_32": sum(
                1
                for x in areas
                if x < 32
            ),

            "below_64": sum(
                1
                for x in areas
                if x < 64
            ),

            "below_100": sum(
                1
                for x in areas
                if x < 100
            ),

        })

    # --------------------------------------------------------
    # TOP imágenes para revisión
    # --------------------------------------------------------

    review_images = sorted(
        all_images,
        key=lambda row: (
            row["risk_score"],
            row["partial_bbox"],
            row["tiny_16"],
            row["objects"],
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    write_csv(
        reports_dir
        / "image_audit_v4.csv",
        all_images
    )

    write_csv(
        reports_dir
        / "object_audit_v4.csv",
        all_annotations
    )

    write_csv(
        reports_dir
        / "split_summary_v4.csv",
        split_summary
    )

    write_csv(
        reports_dir
        / "class_area_statistics_v4.csv",
        class_area_stats
    )

    write_csv(
        reports_dir
        / "top_review_images_v4.csv",
        review_images[:500]
    )

    # --------------------------------------------------------
    # Duplicados
    # --------------------------------------------------------

    duplicate_annotations = []

    for row in all_annotations:

        duplicate_annotations.append(row)

    # Agrupar por imagen + clase + bbox

    duplicate_groups = defaultdict(list)

    for row in all_annotations:

        key = (
            row["image"],
            row["class_id"],
            round(row["x"], 8),
            round(row["y"], 8),
            round(row["w"], 8),
            round(row["h"], 8),
        )

        duplicate_groups[key].append(
            row
        )

    duplicate_rows = []

    for key, rows in (
        duplicate_groups.items()
    ):

        if len(rows) > 1:

            for row in rows:

                duplicate_rows.append({

                    **row,

                    "duplicate_count":
                        len(rows),

                })

    write_csv(
        reports_dir
        / "duplicate_annotations_v4.csv",
        duplicate_rows
    )

    # --------------------------------------------------------
    # Resumen TXT
    # --------------------------------------------------------

    summary_path = (
        reports_dir
        / "AUDIT_V4_SUMMARY.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V4\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "DATASET\n"
        )

        f.write(
            str(DATASET_ROOT)
            + "\n\n"
        )

        f.write(
            "RESUMEN GENERAL\n"
        )

        f.write(
            "-" * 70
            + "\n"
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

        f.write(
            f"Objetos/imagen: "
            f"{total_objects / total_images:.2f}\n"
            if total_images
            else
            "Objetos/imagen: 0\n"
        )

        f.write("\n")

        f.write(
            "TAMAÑO DE OBJETOS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"<16 px²: "
            f"{tiny_16:,} "
            f"("
            f"{tiny_16 / total_objects * 100:.2f}%"
            f")\n"
            if total_objects
            else
            "<16 px²: 0\n"
        )

        f.write(
            f"<32 px²: "
            f"{tiny_32:,} "
            f"("
            f"{tiny_32 / total_objects * 100:.2f}%"
            f")\n"
            if total_objects
            else
            "<32 px²: 0\n"
        )

        f.write(
            f"<64 px²: "
            f"{tiny_64:,} "
            f"("
            f"{tiny_64 / total_objects * 100:.2f}%"
            f")\n"
            if total_objects
            else
            "<64 px²: 0\n"
        )

        f.write(
            f"<100 px²: "
            f"{tiny_100:,} "
            f"("
            f"{tiny_100 / total_objects * 100:.2f}%"
            f")\n"
            if total_objects
            else
            "<100 px²: 0\n"
        )

        f.write("\n")

        f.write(
            "BBOX / BORDES\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Parcialmente fuera: "
            f"{partial_bbox:,}\n"
        )

        f.write(
            f"Completamente fuera: "
            f"{outside_bbox:,}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{near_border:,}\n"
        )

        f.write("\n")

        f.write(
            "INTEGRIDAD\n"
        )

        f.write(
            "-" * 70
            + "\n"
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
            f"{invalid_bboxes:,}\n"
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
            "CROWDED\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for threshold, count in (
            crowded_counts.items()
        ):

            f.write(
                f">= {threshold} objetos: "
                f"{count:,} imágenes\n"
            )

        f.write("\n")

        f.write(
            "CLASIFICACIÓN DE RIESGO\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for risk in (
            "OK",
            "REVIEW",
            "WARNING",
            "CRITICAL",
        ):

            f.write(
                f"{risk}: "
                f"{risk_counter.get(risk, 0):,}"
                f" imágenes\n"
            )

        f.write("\n")

        f.write(
            "ARCHIVOS GENERADOS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "image_audit_v4.csv\n"
        )

        f.write(
            "object_audit_v4.csv\n"
        )

        f.write(
            "split_summary_v4.csv\n"
        )

        f.write(
            "class_area_statistics_v4.csv\n"
        )

        f.write(
            "top_review_images_v4.csv\n"
        )

        f.write(
            "duplicate_annotations_v4.csv\n"
        )

        f.write(
            "AUDIT_V4_SUMMARY.txt\n"
        )

        f.write("\n")

        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Este script SOLO diagnostica.\n"
        )

        f.write(
            "No elimina ni modifica imágenes.\n"
        )

        f.write(
            "No elimina ni modifica labels.\n"
        )

    # --------------------------------------------------------
    # CONSOLA
    # --------------------------------------------------------

    print("\n")
    print(
        "=" * 70
    )

    print(
        "RESULTADO AUDIT V4"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImágenes:              "
        f"{total_images:,}"
    )

    print(
        f"Personas:              "
        f"{total_persons:,}"
    )

    print(
        f"Vehículos:             "
        f"{total_vehicles:,}"
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

    print()

    print(
        "TAMAÑO"
    )

    print(
        f"<16 px²               : "
        f"{tiny_16:,}"
    )

    print(
        f"<32 px²               : "
        f"{tiny_32:,}"
    )

    print(
        f"<64 px²               : "
        f"{tiny_64:,}"
    )

    print(
        f"<100 px²              : "
        f"{tiny_100:,}"
    )

    print()

    print(
        "BORDES"
    )

    print(
        f"BBox parcialmente fuera: "
        f"{partial_bbox:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{outside_bbox:,}"
    )

    print(
        f"Cerca del borde: "
        f"{near_border:,}"
    )

    print()

    print(
        "INTEGRIDAD"
    )

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
        f"{invalid_bboxes:,}"
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

    print(
        "RIESGO"
    )

    print(
        f"OK:                    "
        f"{risk_counter.get('OK', 0):,}"
    )

    print(
        f"REVIEW:                "
        f"{risk_counter.get('REVIEW', 0):,}"
    )

    print(
        f"WARNING:               "
        f"{risk_counter.get('WARNING', 0):,}"
    )

    print(
        f"CRITICAL:              "
        f"{risk_counter.get('CRITICAL', 0):,}"
    )

    print()

    print(
        "CROWDED"
    )

    for threshold in CROWDED_THRESHOLDS:

        print(
            f">= {threshold:3} objetos: "
            f"{crowded_counts[threshold]:6,} imágenes"
        )

    print()

    print(
        "TOP 20 PARA REVISIÓN"
    )

    print(
        "-" * 70
    )

    for index, row in enumerate(
        review_images[:20],
        start=1
    ):

        print(
            f"{index:2}. "
            f"{row['risk']:8} "
            f"score={row['risk_score']:2} "
            f"objects={row['objects']:3} "
            f"tiny16={row['tiny_16']:3} "
            f"partial={row['partial_bbox']:3} "
            f"border={row['near_border']:3}"
        )

        print(
            f"    {row['image']}"
        )

    print()

    print(
        "Reports:"
    )

    print(
        reports_dir
    )

    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "Este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes."
    )

    print(
        "No elimina ni modifica labels."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()