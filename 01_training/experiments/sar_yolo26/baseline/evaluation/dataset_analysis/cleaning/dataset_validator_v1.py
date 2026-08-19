from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import math

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow no está instalado.")
    print("Instala con:")
    print("pip install pillow")
    raise


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

OUTPUT_ROOT = DATASET_ROOT / "validation"

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

VALID_CLASSES = {
    0: "person",
    1: "vehicle",
}

PROGRESS_EVERY = 1000

# Umbrales informativos
TINY_16 = 16
TINY_32 = 32
TINY_64 = 64

BORDER_MARGIN = 0.02


# ============================================================
# UTILIDADES
# ============================================================

def ensure_output():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

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

        for row in rows:
            writer.writerow(row)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def get_images(split):
    images_dir = DATASET_ROOT / split / "images"

    if not images_dir.exists():
        return []

    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def corresponding_label(image_path, split):
    images_root = DATASET_ROOT / split / "images"
    labels_root = DATASET_ROOT / split / "labels"

    relative = image_path.relative_to(images_root)

    return labels_root / relative.with_suffix(".txt")


def relative_dataset_path(path):
    try:
        return str(path.relative_to(DATASET_ROOT))
    except ValueError:
        return str(path)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# VALIDACIÓN DE LABEL
# ============================================================

def validate_label_file(
    label_path,
    image_width,
    image_height
):

    result = {
        "labels": 0,
        "invalid_lines": 0,
        "invalid_coordinates": 0,
        "invalid_bbox": 0,
        "invalid_class": 0,
        "duplicates": 0,
        "objects": 0,
        "person": 0,
        "vehicle": 0,
        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "partial_outside": 0,
        "complete_outside": 0,
        "border": 0,
        "rows": [],
    }

    if not label_path.exists():
        return result

    try:
        text = label_path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        try:
            text = label_path.read_text(
                encoding="utf-8-sig"
            )
        except Exception:
            result["invalid_lines"] += 1
            return result

    lines = text.splitlines()

    seen = set()

    for line_number, raw_line in enumerate(
        lines,
        start=1
    ):

        line = raw_line.strip()

        if not line:
            continue

        result["labels"] += 1

        parts = line.split()

        if len(parts) != 5:
            result["invalid_lines"] += 1

            result["rows"].append({
                "line": line_number,
                "type": "invalid_format",
                "content": line,
            })

            continue

        class_value = parts[0]

        try:
            cls_float = float(class_value)

            if not cls_float.is_integer():
                raise ValueError

            cls = int(cls_float)

        except Exception:

            result["invalid_class"] += 1

            result["rows"].append({
                "line": line_number,
                "type": "invalid_class",
                "content": line,
            })

            continue

        if cls not in VALID_CLASSES:

            result["invalid_class"] += 1

            result["rows"].append({
                "line": line_number,
                "type": "invalid_class",
                "content": line,
            })

            continue

        values = []

        conversion_error = False

        for value in parts[1:]:

            number = safe_float(value)

            if number is None or not math.isfinite(number):
                conversion_error = True
                break

            values.append(number)

        if conversion_error:

            result["invalid_coordinates"] += 1

            result["rows"].append({
                "line": line_number,
                "type": "invalid_coordinates",
                "content": line,
            })

            continue

        x_center, y_center, width, height = values

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 <= width <= 1.0
            and 0.0 <= height <= 1.0
        ):

            result["invalid_coordinates"] += 1

            result["rows"].append({
                "line": line_number,
                "type": "coordinates_outside_0_1",
                "content": line,
            })

        if width <= 0 or height <= 0:

            result["invalid_bbox"] += 1

            result["rows"].append({
                "line": line_number,
                "type": "invalid_bbox",
                "content": line,
            })

            continue

        # ----------------------------------------------------
        # Bounding box en píxeles
        # ----------------------------------------------------

        box_width = width * image_width
        box_height = height * image_height

        area = box_width * box_height

        # ----------------------------------------------------
        # Estadísticas de tamaño
        # ----------------------------------------------------

        if area < TINY_16:
            result["tiny16"] += 1

        if area < TINY_32:
            result["tiny32"] += 1

        if area < TINY_64:
            result["tiny64"] += 1

        # ----------------------------------------------------
        # Coordenadas de la bbox
        # ----------------------------------------------------

        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        if (
            x1 < 0
            or y1 < 0
            or x2 > 1
            or y2 > 1
        ):

            result["partial_outside"] += 1

        if (
            x2 <= 0
            or x1 >= 1
            or y2 <= 0
            or y1 >= 1
        ):

            result["complete_outside"] += 1

        # ----------------------------------------------------
        # Cerca del borde
        # ----------------------------------------------------

        if (
            x1 <= BORDER_MARGIN
            or y1 <= BORDER_MARGIN
            or x2 >= 1 - BORDER_MARGIN
            or y2 >= 1 - BORDER_MARGIN
        ):

            result["border"] += 1

        # ----------------------------------------------------
        # Clases
        # ----------------------------------------------------

        if cls == 0:
            result["person"] += 1

        elif cls == 1:
            result["vehicle"] += 1

        result["objects"] += 1

        # ----------------------------------------------------
        # Duplicados de labels
        # ----------------------------------------------------

        normalized = " ".join(
            [
                str(cls),
                f"{x_center:.8f}",
                f"{y_center:.8f}",
                f"{width:.8f}",
                f"{height:.8f}",
            ]
        )

        if normalized in seen:
            result["duplicates"] += 1

        seen.add(normalized)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("# SAR YOLO26 - DATASET VALIDATOR V1")
    print()

    print("Dataset:")
    print(DATASET_ROOT)

    print()

    print("Output:")
    print(OUTPUT_ROOT)

    print()

    if not DATASET_ROOT.exists():

        print()
        print("[ERROR] No existe DATASET_ROOT:")
        print(DATASET_ROOT)
        return

    ensure_output()

    # --------------------------------------------------------
    # Estructura
    # --------------------------------------------------------

    structure_rows = []

    structure_ok = True

    for split in SPLITS:

        split_root = DATASET_ROOT / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"

        images_ok = images_dir.exists()
        labels_ok = labels_dir.exists()

        if not images_ok or not labels_ok:
            structure_ok = False

        structure_rows.append({
            "split": split,
            "split_exists": split_root.exists(),
            "images_exists": images_ok,
            "labels_exists": labels_ok,
        })

    write_csv(
        OUTPUT_ROOT / "structure_audit.csv",
        structure_rows,
        [
            "split",
            "split_exists",
            "images_exists",
            "labels_exists",
        ],
    )

    # --------------------------------------------------------
    # Estadísticas globales
    # --------------------------------------------------------

    total_images = 0
    total_objects = 0
    total_person = 0
    total_vehicle = 0

    total_tiny16 = 0
    total_tiny32 = 0
    total_tiny64 = 0

    total_partial = 0
    total_complete = 0
    total_border = 0

    total_invalid_lines = 0
    total_invalid_coordinates = 0
    total_invalid_bbox = 0
    total_invalid_class = 0
    total_duplicate_labels = 0

    images_without_labels = []
    labels_without_images = []

    corrupt_images = []

    object_rows = []
    image_rows = []

    split_stats = {}

    all_hashes = defaultdict(list)

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    for split in SPLITS:

        print()
        print(f"## Validando: {split}")

        images = get_images(split)

        if not images:

            print("[INFO] Split sin imágenes:", split)

            split_stats[split] = {
                "images": 0,
                "objects": 0,
                "person": 0,
                "vehicle": 0,
            }

            continue

        print(
            f"Imágenes encontradas: {len(images)}"
        )

        split_objects = 0
        split_person = 0
        split_vehicle = 0

        split_invalid = 0

        for index, image_path in enumerate(
            images,
            start=1
        ):

            label_path = corresponding_label(
                image_path,
                split
            )

            relative_path = relative_dataset_path(
                image_path
            )

            # ------------------------------------------------
            # Imagen
            # ------------------------------------------------

            image_ok = True

            image_width = 0
            image_height = 0

            try:

                with Image.open(image_path) as img:

                    img.verify()

                with Image.open(image_path) as img:

                    image_width, image_height = img.size

            except Exception as exc:

                image_ok = False

                corrupt_images.append({
                    "split": split,
                    "image": relative_path,
                    "error": str(exc),
                })

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if not label_path.exists():

                images_without_labels.append({
                    "split": split,
                    "image": relative_path,
                })

                label_result = {
                    "labels": 0,
                    "invalid_lines": 0,
                    "invalid_coordinates": 0,
                    "invalid_bbox": 0,
                    "invalid_class": 0,
                    "duplicates": 0,
                    "objects": 0,
                    "person": 0,
                    "vehicle": 0,
                    "tiny16": 0,
                    "tiny32": 0,
                    "tiny64": 0,
                    "partial_outside": 0,
                    "complete_outside": 0,
                    "border": 0,
                    "rows": [],
                }

            elif image_ok:

                label_result = validate_label_file(
                    label_path,
                    image_width,
                    image_height
                )

            else:

                label_result = {
                    "labels": 0,
                    "invalid_lines": 0,
                    "invalid_coordinates": 0,
                    "invalid_bbox": 0,
                    "invalid_class": 0,
                    "duplicates": 0,
                    "objects": 0,
                    "person": 0,
                    "vehicle": 0,
                    "tiny16": 0,
                    "tiny32": 0,
                    "tiny64": 0,
                    "partial_outside": 0,
                    "complete_outside": 0,
                    "border": 0,
                    "rows": [],
                }

            # ------------------------------------------------
            # Agregados
            # ------------------------------------------------

            objects = label_result["objects"]

            split_objects += objects
            split_person += label_result["person"]
            split_vehicle += label_result["vehicle"]

            total_images += 1
            total_objects += objects

            total_person += label_result["person"]
            total_vehicle += label_result["vehicle"]

            total_tiny16 += label_result["tiny16"]
            total_tiny32 += label_result["tiny32"]
            total_tiny64 += label_result["tiny64"]

            total_partial += label_result[
                "partial_outside"
            ]

            total_complete += label_result[
                "complete_outside"
            ]

            total_border += label_result[
                "border"
            ]

            total_invalid_lines += label_result[
                "invalid_lines"
            ]

            total_invalid_coordinates += label_result[
                "invalid_coordinates"
            ]

            total_invalid_bbox += label_result[
                "invalid_bbox"
            ]

            total_invalid_class += label_result[
                "invalid_class"
            ]

            total_duplicate_labels += label_result[
                "duplicates"
            ]

            if (
                label_result["invalid_lines"]
                or label_result["invalid_coordinates"]
                or label_result["invalid_bbox"]
                or label_result["invalid_class"]
                or label_result["duplicates"]
            ):

                split_invalid += 1

            # ------------------------------------------------
            # Imagen audit
            # ------------------------------------------------

            image_rows.append({
                "split": split,
                "image": relative_path,
                "label": relative_dataset_path(label_path),
                "width": image_width,
                "height": image_height,
                "objects": objects,
                "person": label_result["person"],
                "vehicle": label_result["vehicle"],
                "tiny16": label_result["tiny16"],
                "tiny32": label_result["tiny32"],
                "tiny64": label_result["tiny64"],
                "partial_outside": label_result[
                    "partial_outside"
                ],
                "complete_outside": label_result[
                    "complete_outside"
                ],
                "border": label_result["border"],
                "invalid_lines": label_result[
                    "invalid_lines"
                ],
                "invalid_coordinates": label_result[
                    "invalid_coordinates"
                ],
                "invalid_bbox": label_result[
                    "invalid_bbox"
                ],
                "invalid_class": label_result[
                    "invalid_class"
                ],
                "duplicate_labels": label_result[
                    "duplicates"
                ],
                "image_ok": image_ok,
            })

            # ------------------------------------------------
            # Problemas de labels
            # ------------------------------------------------

            for problem in label_result["rows"]:

                object_rows.append({
                    "split": split,
                    "image": relative_path,
                    "label": relative_dataset_path(
                        label_path
                    ),
                    "line": problem["line"],
                    "problem": problem["type"],
                    "content": problem["content"],
                })

            # ------------------------------------------------
            # Hash
            # ------------------------------------------------

            if image_ok:

                try:

                    digest = sha256_file(
                        image_path
                    )

                    all_hashes[digest].append({
                        "split": split,
                        "image": relative_path,
                    })

                except Exception as exc:

                    corrupt_images.append({
                        "split": split,
                        "image": relative_path,
                        "error": f"SHA256: {exc}",
                    })

            # ------------------------------------------------
            # Progreso
            # ------------------------------------------------

            if index % PROGRESS_EVERY == 0:

                print(
                    f"Procesadas: "
                    f"{index:,}/{len(images):,}"
                )

        split_stats[split] = {
            "images": len(images),
            "objects": split_objects,
            "person": split_person,
            "vehicle": split_vehicle,
            "invalid_images": split_invalid,
        }

    # ========================================================
    # LABELS SIN IMAGEN
    # ========================================================

    for split in SPLITS:

        labels_dir = DATASET_ROOT / split / "labels"

        if not labels_dir.exists():
            continue

        for label_path in labels_dir.rglob("*.txt"):

            relative = label_path.relative_to(
                labels_dir
            )

            image_exists = False

            for ext in IMAGE_EXTENSIONS:

                candidate = (
                    DATASET_ROOT
                    / split
                    / "images"
                    / relative.with_suffix(ext)
                )

                if candidate.exists():

                    image_exists = True
                    break

            if not image_exists:

                labels_without_images.append({
                    "split": split,
                    "label": relative_dataset_path(
                        label_path
                    ),
                })

    # ========================================================
    # DUPLICADOS DE IMAGEN
    # ========================================================

    duplicate_groups = []

    for digest, entries in all_hashes.items():

        if len(entries) > 1:

            duplicate_groups.append({
                "sha256": digest,
                "count": len(entries),
                "entries": " | ".join(
                    f"{x['split']}:{x['image']}"
                    for x in entries
                ),
            })

    # ========================================================
    # CROSS SPLIT
    # ========================================================

    cross_split_groups = []

    for digest, entries in all_hashes.items():

        splits = set(
            x["split"]
            for x in entries
        )

        if len(splits) > 1:

            cross_split_groups.append({
                "sha256": digest,
                "splits": ",".join(
                    sorted(splits)
                ),
                "entries": " | ".join(
                    f"{x['split']}:{x['image']}"
                    for x in entries
                ),
            })

    # ========================================================
    # CSV
    # ========================================================

    write_csv(
        OUTPUT_ROOT / "image_audit.csv",
        image_rows,
        [
            "split",
            "image",
            "label",
            "width",
            "height",
            "objects",
            "person",
            "vehicle",
            "tiny16",
            "tiny32",
            "tiny64",
            "partial_outside",
            "complete_outside",
            "border",
            "invalid_lines",
            "invalid_coordinates",
            "invalid_bbox",
            "invalid_class",
            "duplicate_labels",
            "image_ok",
        ],
    )

    write_csv(
        OUTPUT_ROOT / "label_problems.csv",
        object_rows,
        [
            "split",
            "image",
            "label",
            "line",
            "problem",
            "content",
        ],
    )

    write_csv(
        OUTPUT_ROOT / "images_without_labels.csv",
        images_without_labels,
        [
            "split",
            "image",
        ],
    )

    write_csv(
        OUTPUT_ROOT / "labels_without_images.csv",
        labels_without_images,
        [
            "split",
            "label",
        ],
    )

    write_csv(
        OUTPUT_ROOT / "corrupt_images.csv",
        corrupt_images,
        [
            "split",
            "image",
            "error",
        ],
    )

    write_csv(
        OUTPUT_ROOT / "duplicate_images.csv",
        duplicate_groups,
        [
            "sha256",
            "count",
            "entries",
        ],
    )

    write_csv(
        OUTPUT_ROOT / "cross_split_duplicates.csv",
        cross_split_groups,
        [
            "sha256",
            "splits",
            "entries",
        ],
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    objects_per_image = (
        total_objects / total_images
        if total_images
        else 0
    )

    print()
    print("============================================================")
    print("# RESULTADO VALIDACIÓN V1")
    print("============================================================")

    print()
    print(
        f"Imágenes:              {total_images:,}"
    )

    print(
        f"Objetos:               {total_objects:,}"
    )

    print(
        f"Objetos/imagen:        "
        f"{objects_per_image:.2f}"
    )

    print()
    print("CLASES")

    print(
        f"Personas:              "
        f"{total_person:,}"
    )

    print(
        f"Vehículos:             "
        f"{total_vehicle:,}"
    )

    print()
    print("OBJETOS PEQUEÑOS")

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

    print()
    print("BORDES")

    print(
        f"BBox parcialmente fuera:"
        f" {total_partial:,}"
    )

    print(
        f"BBox completamente fuera:"
        f" {total_complete:,}"
    )

    print(
        f"Cerca del borde:       "
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
        f"{total_duplicate_labels:,}"
    )

    print(
        f"Imágenes sin labels:   "
        f"{len(images_without_labels):,}"
    )

    print(
        f"Labels sin imágenes:   "
        f"{len(labels_without_images):,}"
    )

    print(
        f"Imágenes corruptas:    "
        f"{len(corrupt_images):,}"
    )

    print()
    print("DUPLICADOS DE IMAGEN")

    print(
        f"Grupos duplicados:     "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Grupos cross-split:    "
        f"{len(cross_split_groups):,}"
    )

    leakage = (
        "SI"
        if cross_split_groups
        else "NO"
    )

    print(
        f"Posible data leakage:  "
        f"{leakage}"
    )

    # ========================================================
    # ESTADO FINAL
    # ========================================================

    critical_conditions = []

    if not structure_ok:
        critical_conditions.append(
            "estructura_incompleta"
        )

    if total_invalid_lines:
        critical_conditions.append(
            "labels_invalidos"
        )

    if total_invalid_coordinates:
        critical_conditions.append(
            "coordenadas_invalidas"
        )

    if total_invalid_bbox:
        critical_conditions.append(
            "bboxes_invalidas"
        )

    if total_invalid_class:
        critical_conditions.append(
            "clases_invalidas"
        )

    if images_without_labels:
        critical_conditions.append(
            "imagenes_sin_labels"
        )

    if labels_without_images:
        critical_conditions.append(
            "labels_sin_imagenes"
        )

    if corrupt_images:
        critical_conditions.append(
            "imagenes_corruptas"
        )

    if cross_split_groups:
        critical_conditions.append(
            "data_leakage_cross_split"
        )

    if critical_conditions:

        status = "REVIEW_REQUIRED"

    else:

        status = "VALIDATED"

    print()
    print("============================================================")
    print(f"ESTADO FINAL: {status}")
    print("============================================================")

    if critical_conditions:

        print()
        print("Motivos:")

        for reason in critical_conditions:

            print(
                f"  - {reason}"
            )

    # ========================================================
    # INFORME TXT
    # ========================================================

    summary_path = (
        OUTPUT_ROOT
        / "VALIDATION_SUMMARY.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET VALIDATOR V1\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Dataset:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Estado final: {status}\n\n"
        )

        f.write(
            "ESTADISTICAS\n"
        )

        f.write(
            f"Imágenes: {total_images}\n"
        )

        f.write(
            f"Objetos: {total_objects}\n"
        )

        f.write(
            f"Objetos/imagen: "
            f"{objects_per_image:.2f}\n"
        )

        f.write(
            f"Personas: {total_person}\n"
        )

        f.write(
            f"Vehículos: {total_vehicle}\n"
        )

        f.write("\n")

        f.write(
            "OBJETOS PEQUEÑOS\n"
        )

        f.write(
            f"<16 px²: {total_tiny16}\n"
        )

        f.write(
            f"<32 px²: {total_tiny32}\n"
        )

        f.write(
            f"<64 px²: {total_tiny64}\n"
        )

        f.write("\n")

        f.write(
            "BORDES\n"
        )

        f.write(
            f"Parcialmente fuera: "
            f"{total_partial}\n"
        )

        f.write(
            f"Completamente fuera: "
            f"{total_complete}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{total_border}\n"
        )

        f.write("\n")

        f.write(
            "INTEGRIDAD\n"
        )

        f.write(
            f"Labels inválidos: "
            f"{total_invalid_lines}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{total_invalid_coordinates}\n"
        )

        f.write(
            f"BBoxes inválidas: "
            f"{total_invalid_bbox}\n"
        )

        f.write(
            f"Clases inválidas: "
            f"{total_invalid_class}\n"
        )

        f.write(
            f"Duplicados labels: "
            f"{total_duplicate_labels}\n"
        )

        f.write(
            f"Imágenes sin labels: "
            f"{len(images_without_labels)}\n"
        )

        f.write(
            f"Labels sin imágenes: "
            f"{len(labels_without_images)}\n"
        )

        f.write(
            f"Imágenes corruptas: "
            f"{len(corrupt_images)}\n"
        )

        f.write("\n")

        f.write(
            "DUPLICADOS DE IMAGEN\n"
        )

        f.write(
            f"Grupos duplicados: "
            f"{len(duplicate_groups)}\n"
        )

        f.write(
            f"Grupos cross-split: "
            f"{len(cross_split_groups)}\n"
        )

        f.write(
            f"Posible data leakage: "
            f"{leakage}\n"
        )

        f.write("\n")

        f.write(
            "MOTIVOS DE REVIEW\n"
        )

        if critical_conditions:

            for reason in critical_conditions:

                f.write(
                    f"- {reason}\n"
                )

        else:

            f.write(
                "Ninguno.\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Este script SOLO valida.\n"
        )

        f.write(
            "No modifica imágenes ni labels.\n"
        )

    print()
    print("Informe:")
    print(summary_path)

    print()
    print(
        "IMPORTANTE: el dataset limpio NO ha sido modificado."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()