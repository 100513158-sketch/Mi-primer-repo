from pathlib import Path
from collections import Counter
from PIL import Image
import csv
import hashlib
import math
import shutil

# ============================================================
# SAR YOLO26 - DATASET AUDIT
# ============================================================
#
# OBJETIVO
# -------
# Auditar la calidad del dataset sin modificarlo.
#
# Detecta:
#   1. Labels inválidos
#   2. Coordenadas fuera de rango
#   3. Bboxes con width/height inválidos
#   4. Objetos extremadamente pequeños
#   5. Objetos extremadamente grandes
#   6. Imágenes sin objetos
#   7. Imágenes extremadamente congestionadas
#   8. Dimensiones de imágenes
#   9. Posibles imágenes duplicadas
#  10. Estadísticas por split y clase
#  11. Ejemplos visuales de anomalías
#
# IMPORTANTE:
#   Este script NO modifica el dataset original.
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
    r"\baseline\evaluation\dataset_analysis\audit"
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

# ------------------------------------------------------------
# Umbrales de auditoría
# ------------------------------------------------------------

# Área de bbox en píxeles.
TINY_AREA = 16
SMALL_AREA = 32

# Objetos extremadamente grandes.
# 50% del área de la imagen.
HUGE_IMAGE_RATIO = 0.50

# Escenas congestionadas.
CROWDED_THRESHOLDS = [
    100,
    200,
    300,
    400,
    500,
]

# Guardaremos ejemplos visuales.
MAX_EXAMPLES_PER_CATEGORY = 100


# ============================================================
# DIRECTORIOS
# ============================================================

REPORTS_DIR = OUTPUT_ROOT / "reports"
EXAMPLES_DIR = OUTPUT_ROOT / "examples"

TINY_DIR = EXAMPLES_DIR / "tiny_objects"
HUGE_DIR = EXAMPLES_DIR / "huge_objects"
CROWDED_DIR = EXAMPLES_DIR / "crowded"
INVALID_DIR = EXAMPLES_DIR / "invalid"


for directory in [
    REPORTS_DIR,
    TINY_DIR,
    HUGE_DIR,
    CROWDED_DIR,
    INVALID_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def write_csv(path, rows, fieldnames):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
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


def md5_file(path, chunk_size=1024 * 1024):

    md5 = hashlib.md5()

    try:

        with path.open("rb") as f:

            while True:

                chunk = f.read(chunk_size)

                if not chunk:
                    break

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:
        return ""


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(value, maximum)
    )


def normalized_bbox_to_pixels(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height
):

    x1 = (
        x_center - width / 2
    ) * image_width

    y1 = (
        y_center - height / 2
    ) * image_height

    x2 = (
        x_center + width / 2
    ) * image_width

    y2 = (
        y_center + height / 2
    ) * image_height

    return (
        x1,
        y1,
        x2,
        y2
    )


def safe_copy(
    source,
    destination_dir,
    prefix=""
):

    try:

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            destination_dir
            / f"{prefix}{source.name}"
        )

        # Evitar sobrescribir.
        counter = 1

        while destination.exists():

            destination = (
                destination_dir
                / f"{prefix}{counter}_{source.name}"
            )

            counter += 1

        shutil.copy2(
            source,
            destination
        )

    except Exception as exc:

        print(
            f"[WARN] No se pudo copiar "
            f"{source}: {exc}"
        )


# ============================================================
# ANÁLISIS DE UN SPLIT
# ============================================================

def analyze_split(split_dir):

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():

        print(
            f"[WARN] No existe: {images_dir}"
        )

        return None

    if not labels_dir.exists():

        print(
            f"[WARN] No existe: {labels_dir}"
        )

        return None

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]:

        image_files.extend(
            images_dir.rglob(extension)
        )

    image_files = sorted(
        image_files
    )

    print(
        f"Imágenes encontradas: "
        f"{len(image_files):,}"
    )

    stats = {

        "split": split_dir.name,

        "images": 0,

        "images_with_objects": 0,

        "images_without_objects": 0,

        "images_with_invalid_labels": 0,

        "images_with_tiny_objects": 0,

        "images_with_huge_objects": 0,

        "images_crowded_100": 0,

        "images_crowded_200": 0,

        "images_crowded_300": 0,

        "images_crowded_400": 0,

        "images_crowded_500": 0,

        "persons": 0,

        "vehicles": 0,

        "objects": 0,

        "tiny_objects": 0,

        "small_objects": 0,

        "huge_objects": 0,

        "invalid_labels": 0,

        "invalid_coordinates": 0,

        "invalid_bbox": 0,

        "invalid_class": 0,

        "missing_labels": 0,

    }

    image_rows = []

    tiny_rows = []

    huge_rows = []

    crowded_rows = []

    invalid_rows = []

    dimensions_rows = []

    duplicate_candidates = []

    hash_map = {}

    example_counters = Counter()

    total_images = len(
        image_files
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        if index % 1000 == 0:

            print(
                f"Procesadas: "
                f"{index:,}/"
                f"{total_images:,}"
            )

        stats["images"] += 1

        # ----------------------------------------------------
        # Dimensiones
        # ----------------------------------------------------

        try:

            with Image.open(
                image_path
            ) as image:

                image_width, image_height = (
                    image.size
                )

        except Exception as exc:

            stats[
                "images_with_invalid_labels"
            ] += 1

            invalid_rows.append({

                "split": split_dir.name,

                "image": str(image_path),

                "label": "",

                "type": "invalid_image",

                "detail": str(exc),

            })

            continue

        dimensions_rows.append({

            "split": split_dir.name,

            "image": str(image_path),

            "width": image_width,

            "height": image_height,

            "pixels": (
                image_width
                * image_height
            ),

        })

        # ----------------------------------------------------
        # Hash
        # ----------------------------------------------------

        image_hash = md5_file(
            image_path
        )

        if image_hash:

            if image_hash in hash_map:

                duplicate_candidates.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "duplicate_of": hash_map[
                        image_hash
                    ],

                    "md5": image_hash,

                })

            else:

                hash_map[
                    image_hash
                ] = str(image_path)

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            stats[
                "missing_labels"
            ] += 1

            invalid_rows.append({

                "split": split_dir.name,

                "image": str(image_path),

                "label": str(label_path),

                "type": "missing_label",

                "detail": "Label no encontrado",

            })

            continue

        person_count = 0
        vehicle_count = 0
        object_count = 0

        tiny_count = 0
        huge_count = 0

        image_has_invalid = False
        image_has_tiny = False
        image_has_huge = False

        try:

            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

        except Exception as exc:

            stats[
                "images_with_invalid_labels"
            ] += 1

            invalid_rows.append({

                "split": split_dir.name,

                "image": str(image_path),

                "label": str(label_path),

                "type": "label_read_error",

                "detail": str(exc),

            })

            continue

        # ----------------------------------------------------
        # Analizar cada bbox
        # ----------------------------------------------------

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # ------------------------------------------------
            # Número de columnas
            # ------------------------------------------------

            if len(parts) != 5:

                stats[
                    "invalid_labels"
                ] += 1

                image_has_invalid = True

                invalid_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "line": line_number,

                    "type": "wrong_columns",

                    "detail": line,

                })

                continue

            try:

                class_id = int(
                    parts[0]
                )

                x_center = float(
                    parts[1]
                )

                y_center = float(
                    parts[2]
                )

                width = float(
                    parts[3]
                )

                height = float(
                    parts[4]
                )

            except ValueError:

                stats[
                    "invalid_labels"
                ] += 1

                image_has_invalid = True

                invalid_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "line": line_number,

                    "type": "non_numeric",

                    "detail": line,

                })

                continue

            # ------------------------------------------------
            # Clase
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                stats[
                    "invalid_class"
                ] += 1

                image_has_invalid = True

                invalid_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "line": line_number,

                    "type": "invalid_class",

                    "detail": (
                        f"class_id={class_id}"
                    ),

                })

                continue

            # ------------------------------------------------
            # Coordenadas
            # ------------------------------------------------

            coordinates_valid = (

                0.0 <= x_center <= 1.0

                and

                0.0 <= y_center <= 1.0

                and

                0.0 <= width <= 1.0

                and

                0.0 <= height <= 1.0

            )

            if not coordinates_valid:

                stats[
                    "invalid_coordinates"
                ] += 1

                image_has_invalid = True

                invalid_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "line": line_number,

                    "type": "coordinates_out_of_range",

                    "detail": line,

                })

                continue

            # ------------------------------------------------
            # Bbox válida
            # ------------------------------------------------

            if width <= 0 or height <= 0:

                stats[
                    "invalid_bbox"
                ] += 1

                image_has_invalid = True

                invalid_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "line": line_number,

                    "type": "zero_or_negative_bbox",

                    "detail": line,

                })

                continue

            # ------------------------------------------------
            # Bbox en píxeles
            # ------------------------------------------------

            pixel_width = (
                width
                * image_width
            )

            pixel_height = (
                height
                * image_height
            )

            area = (
                pixel_width
                * pixel_height
            )

            # ------------------------------------------------
            # Límites reales de bbox
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                normalized_bbox_to_pixels(
                    x_center,
                    y_center,
                    width,
                    height,
                    image_width,
                    image_height
                )
            )

            clipped_x1 = clamp(
                x1,
                0,
                image_width
            )

            clipped_y1 = clamp(
                y1,
                0,
                image_height
            )

            clipped_x2 = clamp(
                x2,
                0,
                image_width
            )

            clipped_y2 = clamp(
                y2,
                0,
                image_height
            )

            outside = (

                abs(
                    x1 - clipped_x1
                ) > 1e-6

                or

                abs(
                    y1 - clipped_y1
                ) > 1e-6

                or

                abs(
                    x2 - clipped_x2
                ) > 1e-6

                or

                abs(
                    y2 - clipped_y2
                ) > 1e-6

            )

            if outside:

                stats[
                    "invalid_coordinates"
                ] += 1

                image_has_invalid = True

                invalid_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "line": line_number,

                    "type": "bbox_outside_image",

                    "detail": line,

                })

            # ------------------------------------------------
            # Estadísticas de clase
            # ------------------------------------------------

            class_name = CLASS_NAMES[
                class_id
            ]

            object_count += 1

            if class_id == 0:

                person_count += 1

            elif class_id == 1:

                vehicle_count += 1

            # ------------------------------------------------
            # Objetos pequeños
            # ------------------------------------------------

            if area < TINY_AREA:

                tiny_count += 1

                stats[
                    "tiny_objects"
                ] += 1

                image_has_tiny = True

                tiny_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "class_id": class_id,

                    "class": class_name,

                    "area_px2": area,

                    "width_px": pixel_width,

                    "height_px": pixel_height,

                    "x_center": x_center,

                    "y_center": y_center,

                    "line": line_number,

                })

                if (
                    example_counters[
                        "tiny"
                    ]
                    <
                    MAX_EXAMPLES_PER_CATEGORY
                ):

                    safe_copy(
                        image_path,
                        TINY_DIR,
                        f"{split_dir.name}_"
                    )

                    example_counters[
                        "tiny"
                    ] += 1

            elif area < SMALL_AREA:

                stats[
                    "small_objects"
                ] += 1

            # ------------------------------------------------
            # Objetos enormes
            # ------------------------------------------------

            image_area = (
                image_width
                * image_height
            )

            if (
                image_area > 0
                and
                area / image_area
                >= HUGE_IMAGE_RATIO
            ):

                huge_count += 1

                stats[
                    "huge_objects"
                ] += 1

                image_has_huge = True

                huge_rows.append({

                    "split": split_dir.name,

                    "image": str(image_path),

                    "label": str(label_path),

                    "class_id": class_id,

                    "class": class_name,

                    "area_px2": area,

                    "image_area_px2": image_area,

                    "area_ratio": (
                        area / image_area
                    ),

                    "width_px": pixel_width,

                    "height_px": pixel_height,

                })

                if (
                    example_counters[
                        "huge"
                    ]
                    <
                    MAX_EXAMPLES_PER_CATEGORY
                ):

                    safe_copy(
                        image_path,
                        HUGE_DIR,
                        f"{split_dir.name}_"
                    )

                    example_counters[
                        "huge"
                    ] += 1

        # ----------------------------------------------------
        # Estadísticas por imagen
        # ----------------------------------------------------

        stats["persons"] += (
            person_count
        )

        stats["vehicles"] += (
            vehicle_count
        )

        stats["objects"] += (
            object_count
        )

        if object_count > 0:

            stats[
                "images_with_objects"
            ] += 1

        else:

            stats[
                "images_without_objects"
            ] += 1

        if image_has_invalid:

            stats[
                "images_with_invalid_labels"
            ] += 1

        if image_has_tiny:

            stats[
                "images_with_tiny_objects"
            ] += 1

        if image_has_huge:

            stats[
                "images_with_huge_objects"
            ] += 1

        # ----------------------------------------------------
        # Congestión
        # ----------------------------------------------------

        for threshold in CROWDED_THRESHOLDS:

            if object_count >= threshold:

                stats[
                    f"images_crowded_{threshold}"
                ] += 1

        if object_count >= 100:

            crowded_rows.append({

                "split": split_dir.name,

                "image": str(image_path),

                "objects": object_count,

                "persons": person_count,

                "vehicles": vehicle_count,

            })

            if (
                example_counters[
                    "crowded"
                ]
                <
                MAX_EXAMPLES_PER_CATEGORY
            ):

                safe_copy(
                    image_path,
                    CROWDED_DIR,
                    f"{split_dir.name}_"
                )

                example_counters[
                    "crowded"
                ] += 1

        image_rows.append({

            "split": split_dir.name,

            "image": str(image_path),

            "width": image_width,

            "height": image_height,

            "persons": person_count,

            "vehicles": vehicle_count,

            "objects": object_count,

            "tiny_objects": tiny_count,

            "huge_objects": huge_count,

            "has_invalid": int(
                image_has_invalid
            ),

        })

    return {

        "stats": stats,

        "image_rows": image_rows,

        "tiny_rows": tiny_rows,

        "huge_rows": huge_rows,

        "crowded_rows": crowded_rows,

        "invalid_rows": invalid_rows,

        "dimensions_rows": dimensions_rows,

        "duplicate_candidates": (
            duplicate_candidates
        ),

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAR YOLO26 - DATASET AUDIT")
    print("=" * 70)

    print("\nDataset:")
    print(DATASET_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    all_results = []

    # ========================================================
    # SPLITS
    # ========================================================

    for split in SPLITS:

        split_dir = (
            DATASET_ROOT / split
        )

        if not split_dir.exists():

            print(
                f"\n[INFO] Split no encontrado: "
                f"{split}"
            )

            continue

        print("\n")
        print("-" * 70)
        print(
            f"Analizando: {split}"
        )
        print("-" * 70)

        result = analyze_split(
            split_dir
        )

        if result:

            all_results.append(
                result
            )

    # ========================================================
    # COMBINAR RESULTADOS
    # ========================================================

    summary_rows = []

    all_image_rows = []
    all_tiny_rows = []
    all_huge_rows = []
    all_crowded_rows = []
    all_invalid_rows = []
    all_dimensions_rows = []
    all_duplicate_rows = []

    for result in all_results:

        stats = result["stats"]

        summary_rows.append(
            stats.copy()
        )

        all_image_rows.extend(
            result["image_rows"]
        )

        all_tiny_rows.extend(
            result["tiny_rows"]
        )

        all_huge_rows.extend(
            result["huge_rows"]
        )

        all_crowded_rows.extend(
            result["crowded_rows"]
        )

        all_invalid_rows.extend(
            result["invalid_rows"]
        )

        all_dimensions_rows.extend(
            result["dimensions_rows"]
        )

        all_duplicate_rows.extend(
            result[
                "duplicate_candidates"
            ]
        )

    # ========================================================
    # CSV SUMMARY
    # ========================================================

    if summary_rows:

        write_csv(
            REPORTS_DIR
            / "audit_summary.csv",
            summary_rows,
            summary_rows[0].keys()
        )

    if all_image_rows:

        write_csv(
            REPORTS_DIR
            / "image_statistics.csv",
            all_image_rows,
            all_image_rows[0].keys()
        )

    if all_tiny_rows:

        write_csv(
            REPORTS_DIR
            / "tiny_objects.csv",
            all_tiny_rows,
            all_tiny_rows[0].keys()
        )

    if all_huge_rows:

        write_csv(
            REPORTS_DIR
            / "huge_objects.csv",
            all_huge_rows,
            all_huge_rows[0].keys()
        )

    if all_crowded_rows:

        all_crowded_rows.sort(
            key=lambda x:
            x["objects"],
            reverse=True
        )

        write_csv(
            REPORTS_DIR
            / "crowded_images.csv",
            all_crowded_rows,
            all_crowded_rows[0].keys()
        )

    if all_invalid_rows:

        write_csv(
            REPORTS_DIR
            / "invalid_labels.csv",
            all_invalid_rows,
            all_invalid_rows[0].keys()
        )

    if all_dimensions_rows:

        write_csv(
            REPORTS_DIR
            / "image_dimensions.csv",
            all_dimensions_rows,
            all_dimensions_rows[0].keys()
        )

    if all_duplicate_rows:

        write_csv(
            REPORTS_DIR
            / "duplicate_images.csv",
            all_duplicate_rows,
            all_duplicate_rows[0].keys()
        )

    # ========================================================
    # RESUMEN GLOBAL
    # ========================================================

    total_images = sum(
        x["images"]
        for x in summary_rows
    )

    total_persons = sum(
        x["persons"]
        for x in summary_rows
    )

    total_vehicles = sum(
        x["vehicles"]
        for x in summary_rows
    )

    total_objects = sum(
        x["objects"]
        for x in summary_rows
    )

    total_tiny = sum(
        x["tiny_objects"]
        for x in summary_rows
    )

    total_small = sum(
        x["small_objects"]
        for x in summary_rows
    )

    total_huge = sum(
        x["huge_objects"]
        for x in summary_rows
    )

    total_invalid = sum(
        x["invalid_labels"]
        for x in summary_rows
    )

    total_invalid_coordinates = sum(
        x["invalid_coordinates"]
        for x in summary_rows
    )

    total_invalid_bbox = sum(
        x["invalid_bbox"]
        for x in summary_rows
    )

    total_invalid_class = sum(
        x["invalid_class"]
        for x in summary_rows
    )

    total_missing = sum(
        x["missing_labels"]
        for x in summary_rows
    )

    print("\n")
    print("=" * 70)
    print("AUDITORÍA FINALIZADA")
    print("=" * 70)

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

    print(
        f"Objetos < {TINY_AREA} px²: "
        f"{total_tiny:,}"
    )

    print(
        f"Objetos < {SMALL_AREA} px²: "
        f"{total_small:,}"
    )

    print(
        f"Objetos enormes:       "
        f"{total_huge:,}"
    )

    print(
        f"Labels inválidos:      "
        f"{total_invalid:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{total_invalid_coordinates:,}"
    )

    print(
        f"Bboxes inválidas:      "
        f"{total_invalid_bbox:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{total_invalid_class:,}"
    )

    print(
        f"Labels inexistentes:   "
        f"{total_missing:,}"
    )

    print("\n")
    print("Resultados:")
    print(REPORTS_DIR)

    print("\nEjemplos:")
    print(EXAMPLES_DIR)

    print("\n")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()