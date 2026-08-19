from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import csv
import hashlib
import math
import shutil


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO"
    r"\processed\sar\VisDrone_SAR_2CLASS"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\audit_v3"
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
# Umbrales de tamaño
# ------------------------------------------------------------

EXTREME_TINY_AREA = 16
TINY_AREA = 32
VERY_SMALL_AREA = 64

# Objetos prácticamente degenerados
DEGENERATE_WIDTH = 1.0
DEGENERATE_HEIGHT = 1.0

# ------------------------------------------------------------
# Borde
# ------------------------------------------------------------

EDGE_MARGIN_PX = 2.0

# ------------------------------------------------------------
# Crowded
# ------------------------------------------------------------

CROWDED_THRESHOLDS = [
    100,
    150,
    200,
    300,
    400,
    500,
]

# ------------------------------------------------------------
# Ejemplos visuales / CSV
# ------------------------------------------------------------

MAX_EXAMPLES_PER_CATEGORY = 100
MAX_DUPLICATE_EXAMPLES = 100


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def clamp(value, low, high):
    return max(low, min(value, high))


def bbox_hash(
    class_id,
    x_center,
    y_center,
    width,
    height,
):
    """
    Hash exacto de una anotación.

    Se utiliza para detectar anotaciones duplicadas
    dentro de la misma imagen.
    """

    raw = (
        f"{class_id}|"
        f"{x_center:.10f}|"
        f"{y_center:.10f}|"
        f"{width:.10f}|"
        f"{height:.10f}"
    )

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


def get_image_files(images_dir):
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
        files.extend(
            images_dir.rglob(ext)
        )

    return sorted(set(files))


def get_image_size(image_path):
    """
    Devuelve width, height de la imagen.
    """

    try:
        with Image.open(image_path) as img:
            return img.width, img.height

    except Exception:
        return None, None


# ============================================================
# CLASIFICACIÓN DEL OBJETO
# ============================================================

def classify_object(
    area_px,
    width_px,
    height_px,
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    coordinate_invalid,
    bbox_invalid,
):
    """
    Clasifica el objeto.

    KEEP:
        Anotación aparentemente normal.

    REVIEW:
        Anotación válida pero potencialmente problemática.

    REMOVE_CANDIDATE:
        Anotación con anomalía fuerte.
    """

    reasons = []

    status = "KEEP"

    # --------------------------------------------------------
    # Coordenadas inválidas
    # --------------------------------------------------------

    if coordinate_invalid:
        status = "REMOVE_CANDIDATE"
        reasons.append(
            "normalized_coordinate_out_of_range"
        )

    # --------------------------------------------------------
    # BBox inválida
    # --------------------------------------------------------

    if bbox_invalid:
        status = "REMOVE_CANDIDATE"
        reasons.append(
            "invalid_bbox_dimensions"
        )

    # --------------------------------------------------------
    # Tamaños extremadamente pequeños
    # --------------------------------------------------------

    if area_px < EXTREME_TINY_AREA:
        if status == "KEEP":
            status = "REVIEW"

        reasons.append(
            "extreme_tiny_object"
        )

    elif area_px < TINY_AREA:
        if status == "KEEP":
            status = "REVIEW"

        reasons.append(
            "tiny_object"
        )

    elif area_px < VERY_SMALL_AREA:
        if status == "KEEP":
            status = "REVIEW"

        reasons.append(
            "very_small_object"
        )

    # --------------------------------------------------------
    # Dimensiones degeneradas
    # --------------------------------------------------------

    if (
        width_px <= DEGENERATE_WIDTH
        or height_px <= DEGENERATE_HEIGHT
    ):
        if status == "KEEP":
            status = "REVIEW"

        reasons.append(
            "degenerate_bbox"
        )

    # --------------------------------------------------------
    # Completamente fuera
    # --------------------------------------------------------

    completely_outside = (
        x2 <= 0
        or y2 <= 0
        or x1 >= image_width
        or y1 >= image_height
    )

    if completely_outside:
        status = "REMOVE_CANDIDATE"

        reasons.append(
            "bbox_completely_outside_image"
        )

    # --------------------------------------------------------
    # Parcialmente fuera
    # --------------------------------------------------------

    partially_outside = (
        x1 < 0
        or y1 < 0
        or x2 > image_width
        or y2 > image_height
    )

    if partially_outside and not completely_outside:

        if status == "KEEP":
            status = "REVIEW"

        reasons.append(
            "bbox_partially_outside_image"
        )

    # --------------------------------------------------------
    # Cerca del borde
    # --------------------------------------------------------

    near_edge = (
        x1 <= EDGE_MARGIN_PX
        or y1 <= EDGE_MARGIN_PX
        or x2 >= image_width - EDGE_MARGIN_PX
        or y2 >= image_height - EDGE_MARGIN_PX
    )

    if near_edge:

        if status == "KEEP":
            status = "REVIEW"

        reasons.append(
            "near_image_border"
        )

    return (
        status,
        reasons,
        completely_outside,
        partially_outside,
        near_edge,
    )


# ============================================================
# ANALIZAR SPLIT
# ============================================================

def analyze_split(
    split_dir,
    object_writer,
    image_writer,
    duplicate_writer,
    counters,
    examples,
):
    """
    Analiza un split completo.

    No modifica archivos.
    """

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        print(
            f"[INFO] Images no encontrado: "
            f"{images_dir}"
        )
        return

    if not labels_dir.exists():
        print(
            f"[INFO] Labels no encontrado: "
            f"{labels_dir}"
        )
        return

    image_files = get_image_files(
        images_dir
    )

    print(
        f"\nImágenes encontradas: "
        f"{len(image_files):,}"
    )

    split_name = split_dir.name

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        image_width, image_height = (
            get_image_size(image_path)
        )

        if (
            image_width is None
            or image_height is None
        ):
            counters[
                "corrupt_images"
            ] += 1

            continue

        counters[
            "images"
        ] += 1

        total_objects = 0
        persons = 0
        vehicles = 0

        image_anomalies = Counter()

        # Para detectar duplicados
        annotation_hashes = defaultdict(list)

        if not label_path.exists():

            counters[
                "labels_missing"
            ] += 1

            image_writer.writerow({
                "split": split_name,
                "image": str(image_path),
                "label_file": str(label_path),
                "image_width": image_width,
                "image_height": image_height,
                "objects": 0,
                "persons": 0,
                "vehicles": 0,
                "status": "REVIEW",
                "reason": "label_file_missing",
            })

            if len(
                examples["label_missing"]
            ) < MAX_EXAMPLES_PER_CATEGORY:

                examples[
                    "label_missing"
                ].append(
                    str(image_path)
                )

            continue

        try:

            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

        except Exception as exc:

            counters[
                "label_read_errors"
            ] += 1

            print(
                f"[ERROR] "
                f"{label_path}: {exc}"
            )

            continue

        # ====================================================
        # OBJETOS
        # ====================================================

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            raw_line = line.strip()

            if not raw_line:
                continue

            counters[
                "label_lines"
            ] += 1

            parts = raw_line.split()

            # ------------------------------------------------
            # Número de campos
            # ------------------------------------------------

            if len(parts) < 5:

                counters[
                    "invalid_labels"
                ] += 1

                image_anomalies[
                    "invalid_label"
                ] += 1

                object_writer.writerow({
                    "split": split_name,
                    "image": str(image_path),
                    "label_file": str(label_path),
                    "line": line_number,
                    "class_id": "",
                    "class_name": "",
                    "x_center": "",
                    "y_center": "",
                    "width": "",
                    "height": "",
                    "image_width": image_width,
                    "image_height": image_height,
                    "bbox_width_px": "",
                    "bbox_height_px": "",
                    "bbox_area_px2": "",
                    "x1_px": "",
                    "y1_px": "",
                    "x2_px": "",
                    "y2_px": "",
                    "status": "REMOVE_CANDIDATE",
                    "reason": "invalid_label_format",
                    "duplicate": False,
                })

                continue

            # ------------------------------------------------
            # Parseo
            # ------------------------------------------------

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

            except Exception:

                counters[
                    "invalid_labels"
                ] += 1

                image_anomalies[
                    "invalid_label"
                ] += 1

                continue

            class_name = CLASS_NAMES.get(
                class_id,
                "unknown",
            )

            # ------------------------------------------------
            # Clase
            # ------------------------------------------------

            class_invalid = (
                class_id
                not in CLASS_NAMES
            )

            if class_invalid:

                counters[
                    "invalid_classes"
                ] += 1

                image_anomalies[
                    "invalid_class"
                ] += 1

            # ------------------------------------------------
            # Coordenadas normalizadas
            # ------------------------------------------------

            coordinate_invalid = (
                x_center < 0
                or x_center > 1
                or y_center < 0
                or y_center > 1
            )

            if coordinate_invalid:

                counters[
                    "invalid_coordinates"
                ] += 1

                image_anomalies[
                    "invalid_coordinate"
                ] += 1

                if len(
                    examples["coordinate_invalid"]
                ) < MAX_EXAMPLES_PER_CATEGORY:

                    examples[
                        "coordinate_invalid"
                    ].append(
                        str(image_path)
                    )

            # ------------------------------------------------
            # Dimensiones normalizadas
            # ------------------------------------------------

            bbox_invalid = (
                width <= 0
                or height <= 0
            )

            if bbox_invalid:

                counters[
                    "invalid_bboxes"
                ] += 1

                image_anomalies[
                    "invalid_bbox"
                ] += 1

            # ------------------------------------------------
            # Coordenadas en píxeles
            # ------------------------------------------------

            bbox_width_px = (
                width * image_width
            )

            bbox_height_px = (
                height * image_height
            )

            bbox_area_px = (
                bbox_width_px
                * bbox_height_px
            )

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

            # ------------------------------------------------
            # Clasificación
            # ------------------------------------------------

            (
                status,
                reasons,
                completely_outside,
                partially_outside,
                near_edge,
            ) = classify_object(
                bbox_area_px,
                bbox_width_px,
                bbox_height_px,
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height,
                coordinate_invalid,
                bbox_invalid,
            )

            # ------------------------------------------------
            # Clase inválida
            # ------------------------------------------------

            if class_invalid:

                status = "REMOVE_CANDIDATE"

                reasons.append(
                    "invalid_class"
                )

            # ------------------------------------------------
            # Tiny
            # ------------------------------------------------

            if (
                bbox_area_px
                < EXTREME_TINY_AREA
            ):

                counters[
                    "extreme_tiny"
                ] += 1

                image_anomalies[
                    "extreme_tiny"
                ] += 1

                if len(
                    examples["extreme_tiny"]
                ) < MAX_EXAMPLES_PER_CATEGORY:

                    examples[
                        "extreme_tiny"
                    ].append(
                        str(image_path)
                    )

            elif (
                bbox_area_px
                < TINY_AREA
            ):

                counters[
                    "tiny"
                ] += 1

                image_anomalies[
                    "tiny"
                ] += 1

            elif (
                bbox_area_px
                < VERY_SMALL_AREA
            ):

                counters[
                    "very_small"
                ] += 1

                image_anomalies[
                    "very_small"
                ] += 1

            # ------------------------------------------------
            # Fuera de imagen
            # ------------------------------------------------

            if completely_outside:

                counters[
                    "bbox_completely_outside"
                ] += 1

                image_anomalies[
                    "bbox_completely_outside"
                ] += 1

                if len(
                    examples[
                        "bbox_completely_outside"
                    ]
                ) < MAX_EXAMPLES_PER_CATEGORY:

                    examples[
                        "bbox_completely_outside"
                    ].append(
                        str(image_path)
                    )

            elif partially_outside:

                counters[
                    "bbox_partially_outside"
                ] += 1

                image_anomalies[
                    "bbox_partially_outside"
                ] += 1

                if len(
                    examples[
                        "bbox_partially_outside"
                    ]
                ) < MAX_EXAMPLES_PER_CATEGORY:

                    examples[
                        "bbox_partially_outside"
                    ].append(
                        str(image_path)
                    )

            # ------------------------------------------------
            # Cerca del borde
            # ------------------------------------------------

            if near_edge:

                counters[
                    "near_edge"
                ] += 1

                image_anomalies[
                    "near_edge"
                ] += 1

            # ------------------------------------------------
            # Conteo de clases
            # ------------------------------------------------

            total_objects += 1

            if class_id == 0:
                persons += 1

            elif class_id == 1:
                vehicles += 1

            # ------------------------------------------------
            # Duplicados
            # ------------------------------------------------

            current_hash = bbox_hash(
                class_id,
                x_center,
                y_center,
                width,
                height,
            )

            annotation_hashes[
                current_hash
            ].append(
                line_number
            )

            # ------------------------------------------------
            # CSV objeto
            # ------------------------------------------------

            object_writer.writerow({
                "split": split_name,
                "image": str(image_path),
                "label_file": str(label_path),
                "line": line_number,
                "class_id": class_id,
                "class_name": class_name,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "image_width": image_width,
                "image_height": image_height,
                "bbox_width_px": bbox_width_px,
                "bbox_height_px": bbox_height_px,
                "bbox_area_px2": bbox_area_px,
                "x1_px": x1,
                "y1_px": y1,
                "x2_px": x2,
                "y2_px": y2,
                "status": status,
                "reason": ";".join(
                    sorted(set(reasons))
                ),
                "duplicate": False,
            })

        # ====================================================
        # DUPLICADOS
        # ====================================================

        duplicate_count = 0

        for annotation_hash, line_numbers in (
            annotation_hashes.items()
        ):

            if len(line_numbers) > 1:

                duplicate_count += (
                    len(line_numbers) - 1
                )

                counters[
                    "duplicate_annotations"
                ] += (
                    len(line_numbers) - 1
                )

                duplicate_writer.writerow({
                    "split": split_name,
                    "image": str(image_path),
                    "label_file": str(label_path),
                    "duplicate_count": len(
                        line_numbers
                    ),
                    "line_numbers": ",".join(
                        map(
                            str,
                            line_numbers,
                        )
                    ),
                    "hash": annotation_hash,
                })

                if len(
                    examples["duplicates"]
                ) < MAX_DUPLICATE_EXAMPLES:

                    examples[
                        "duplicates"
                    ].append(
                        str(image_path)
                    )

        # ====================================================
        # ESTADÍSTICAS IMAGEN
        # ====================================================

        image_status = "KEEP"

        reasons = []

        if total_objects == 0:

            image_status = "REVIEW"

            reasons.append(
                "image_without_objects"
            )

        if duplicate_count > 0:

            image_status = "REVIEW"

            reasons.append(
                "duplicate_annotations"
            )

        if image_anomalies[
            "invalid_label"
        ] > 0:

            image_status = (
                "REMOVE_CANDIDATE"
            )

            reasons.append(
                "invalid_labels"
            )

        if image_anomalies[
            "invalid_coordinate"
        ] > 0:

            image_status = (
                "REMOVE_CANDIDATE"
            )

            reasons.append(
                "invalid_coordinates"
            )

        if image_anomalies[
            "invalid_bbox"
        ] > 0:

            image_status = (
                "REMOVE_CANDIDATE"
            )

            reasons.append(
                "invalid_bboxes"
            )

        if image_anomalies[
            "bbox_completely_outside"
        ] > 0:

            image_status = (
                "REMOVE_CANDIDATE"
            )

            reasons.append(
                "bbox_completely_outside"
            )

        elif image_anomalies[
            "bbox_partially_outside"
        ] > 0:

            if image_status == "KEEP":
                image_status = "REVIEW"

            reasons.append(
                "bbox_partially_outside"
            )

        if image_anomalies[
            "extreme_tiny"
        ] > 0:

            if image_status == "KEEP":
                image_status = "REVIEW"

            reasons.append(
                "extreme_tiny_objects"
            )

        image_writer.writerow({
            "split": split_name,
            "image": str(image_path),
            "label_file": str(label_path),
            "image_width": image_width,
            "image_height": image_height,
            "objects": total_objects,
            "persons": persons,
            "vehicles": vehicles,
            "duplicate_annotations": duplicate_count,
            "tiny_objects": image_anomalies[
                "tiny"
            ],
            "extreme_tiny_objects": image_anomalies[
                "extreme_tiny"
            ],
            "very_small_objects": image_anomalies[
                "very_small"
            ],
            "bbox_partially_outside": image_anomalies[
                "bbox_partially_outside"
            ],
            "bbox_completely_outside": image_anomalies[
                "bbox_completely_outside"
            ],
            "near_edge": image_anomalies[
                "near_edge"
            ],
            "status": image_status,
            "reason": ";".join(
                sorted(set(reasons))
            ),
        })

        # ====================================================
        # CROWDED
        # ====================================================

        for threshold in CROWDED_THRESHOLDS:

            if total_objects >= threshold:

                counters[
                    f"crowded_{threshold}"
                ] += 1

                key = (
                    f"crowded_{threshold}"
                )

                if len(
                    examples[key]
                ) < MAX_EXAMPLES_PER_CATEGORY:

                    examples[key].append(
                        str(image_path)
                    )

        # ====================================================
        # PROGRESO
        # ====================================================

        if index % 1000 == 0:

            print(
                f"Procesadas: "
                f"{index:,}/"
                f"{len(image_files):,}"
            )


# ============================================================
# ESCRIBIR CSV
# ============================================================

def create_csv(path, fieldnames):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return path.open(
        "w",
        newline="",
        encoding="utf-8"
    ), csv.DictWriter(
        path.open(
            "w",
            newline="",
            encoding="utf-8"
        ),
        fieldnames=fieldnames
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAR YOLO26 - DATASET AUDIT V3")
    print("=" * 70)

    print("\nDataset:")
    print(DATASET_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    # --------------------------------------------------------
    # Validar dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print(
            "\n[ERROR] No existe DATASET_ROOT:"
        )

        print(DATASET_ROOT)

        return

    # --------------------------------------------------------
    # Directorios
    # --------------------------------------------------------

    reports_dir = (
        OUTPUT_ROOT / "reports"
    )

    examples_dir = (
        OUTPUT_ROOT / "examples"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    examples_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Contadores
    # --------------------------------------------------------

    counters = Counter()

    examples = defaultdict(list)

    # --------------------------------------------------------
    # CSV objetos
    # --------------------------------------------------------

    object_csv_path = (
        reports_dir
        / "object_audit.csv"
    )

    image_csv_path = (
        reports_dir
        / "image_audit.csv"
    )

    duplicate_csv_path = (
        reports_dir
        / "duplicate_annotations.csv"
    )

    object_fields = [
        "split",
        "image",
        "label_file",
        "line",
        "class_id",
        "class_name",
        "x_center",
        "y_center",
        "width",
        "height",
        "image_width",
        "image_height",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_px2",
        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",
        "status",
        "reason",
        "duplicate",
    ]

    image_fields = [
        "split",
        "image",
        "label_file",
        "image_width",
        "image_height",
        "objects",
        "persons",
        "vehicles",
        "duplicate_annotations",
        "tiny_objects",
        "extreme_tiny_objects",
        "very_small_objects",
        "bbox_partially_outside",
        "bbox_completely_outside",
        "near_edge",
        "status",
        "reason",
    ]

    duplicate_fields = [
        "split",
        "image",
        "label_file",
        "duplicate_count",
        "line_numbers",
        "hash",
    ]

    with (
        object_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as object_file,

        image_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as image_file,

        duplicate_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as duplicate_file,
    ):

        object_writer = csv.DictWriter(
            object_file,
            fieldnames=object_fields,
        )

        image_writer = csv.DictWriter(
            image_file,
            fieldnames=image_fields,
        )

        duplicate_writer = csv.DictWriter(
            duplicate_file,
            fieldnames=duplicate_fields,
        )

        object_writer.writeheader()
        image_writer.writeheader()
        duplicate_writer.writeheader()

        # ----------------------------------------------------
        # SPLITS
        # ----------------------------------------------------

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

            print(
                f"\n{'-' * 70}"
            )

            print(
                f"## Analizando: {split}"
            )

            analyze_split(
                split_dir,
                object_writer,
                image_writer,
                duplicate_writer,
                counters,
                examples,
            )

    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESULTADO GENERAL")
    print("=" * 70)

    print(
        f"Imágenes:              "
        f"{counters['images']:,}"
    )

    print(
        f"Líneas de labels:      "
        f"{counters['label_lines']:,}"
    )

    print(
        f"Objetos extremos <16:  "
        f"{counters['extreme_tiny']:,}"
    )

    print(
        f"Objetos <32:           "
        f"{counters['tiny']:,}"
    )

    print(
        f"Objetos <64:           "
        f"{counters['very_small']:,}"
    )

    print(
        f"Labels inválidos:      "
        f"{counters['invalid_labels']:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{counters['invalid_coordinates']:,}"
    )

    print(
        f"BBoxes inválidas:      "
        f"{counters['invalid_bboxes']:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{counters['invalid_classes']:,}"
    )

    print(
        f"BBox fuera completa:   "
        f"{counters['bbox_completely_outside']:,}"
    )

    print(
        f"BBox parcialmente fuera:"
        f" {counters['bbox_partially_outside']:,}"
    )

    print(
        f"Cerca del borde:       "
        f"{counters['near_edge']:,}"
    )

    print(
        f"Labels duplicados:     "
        f"{counters['duplicate_annotations']:,}"
    )

    print(
        f"Imágenes sin labels:   "
        f"{counters['labels_missing']:,}"
    )

    print(
        f"Imágenes corruptas:    "
        f"{counters['corrupt_images']:,}"
    )

    print("\n")
    print("=" * 70)
    print("ESCENAS DENSAS")
    print("=" * 70)

    for threshold in CROWDED_THRESHOLDS:

        print(
            f">= {threshold:3} objetos: "
            f"{counters[f'crowded_{threshold}']:,}"
            f" imágenes"
        )

    # ========================================================
    # GENERAR ARCHIVO SUMMARY
    # ========================================================

    summary_path = (
        reports_dir
        / "audit_summary.csv"
    )

    summary_rows = [
        {
            "metric": key,
            "value": value,
        }
        for key, value in sorted(
            counters.items()
        )
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "value",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # ========================================================
    # LISTAS DE EJEMPLOS
    # ========================================================

    examples_csv_path = (
        reports_dir
        / "audit_examples.csv"
    )

    with examples_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "image",
            ],
        )

        writer.writeheader()

        for category, paths in sorted(
            examples.items()
        ):

            for path in paths:

                writer.writerow({
                    "category": category,
                    "image": path,
                })

    # ========================================================
    # TXT RESUMEN
    # ========================================================

    summary_txt = (
        reports_dir
        / "AUDIT_SUMMARY.txt"
    )

    with summary_txt.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V3\n"
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
            f"Imágenes: "
            f"{counters['images']:,}\n"
        )

        f.write(
            f"Objetos <16 px²: "
            f"{counters['extreme_tiny']:,}\n"
        )

        f.write(
            f"Objetos <32 px²: "
            f"{counters['tiny']:,}\n"
        )

        f.write(
            f"Objetos <64 px²: "
            f"{counters['very_small']:,}\n"
        )

        f.write(
            f"Labels inválidos: "
            f"{counters['invalid_labels']:,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{counters['invalid_coordinates']:,}\n"
        )

        f.write(
            f"BBoxes inválidas: "
            f"{counters['invalid_bboxes']:,}\n"
        )

        f.write(
            f"Clases inválidas: "
            f"{counters['invalid_classes']:,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{counters['bbox_completely_outside']:,}\n"
        )

        f.write(
            f"BBox parcialmente fuera: "
            f"{counters['bbox_partially_outside']:,}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{counters['near_edge']:,}\n"
        )

        f.write(
            f"Labels duplicados: "
            f"{counters['duplicate_annotations']:,}\n"
        )

        f.write(
            f"Labels inexistentes: "
            f"{counters['labels_missing']:,}\n"
        )

        f.write(
            f"Imágenes corruptas: "
            f"{counters['corrupt_images']:,}\n"
        )

        f.write("\nCROWDED\n")
        f.write("-" * 70 + "\n")

        for threshold in (
            CROWDED_THRESHOLDS
        ):

            f.write(
                f">= {threshold} objetos: "
                f"{counters[f'crowded_{threshold}']:,}"
                f" imágenes\n"
            )

        f.write("\n")
        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Este script SOLO diagnostica.\n"
            "No elimina ni modifica imágenes "
            "ni labels.\n"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("AUDITORÍA V3 FINALIZADA")
    print("=" * 70)

    print(
        "\nReports:"
    )

    print(
        reports_dir
    )

    print(
        "\nArchivos principales:"
    )

    print(
        f"  - {object_csv_path.name}"
    )

    print(
        f"  - {image_csv_path.name}"
    )

    print(
        f"  - {duplicate_csv_path.name}"
    )

    print(
        f"  - {summary_path.name}"
    )

    print(
        f"  - {summary_txt.name}"
    )

    print(
        "\nIMPORTANTE: "
        "este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica "
        "imágenes o labels."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()