from pathlib import Path
from collections import Counter

ROOT = Path(r"processed\sar\VisDrone_SAR_2CLASS")

SPLITS = ["train", "val", "test_dev"]

VALID_CLASSES = {
    0: "person",
    1: "vehicle",
}

print("=" * 80)
print("AUDITORÍA DATASET SARC-DRONE - 2 CLASES")
print("=" * 80)

print()
print(f"Dataset: {ROOT.resolve()}")
print(f"Clases válidas: {list(VALID_CLASSES.keys())}")

if not ROOT.exists():
    raise SystemExit(
        f"\nERROR: no existe el dataset:\n{ROOT.resolve()}"
    )

total_images = 0
total_labels = 0
total_objects = 0

global_classes = Counter()

total_missing_labels = 0
total_orphan_labels = 0
total_invalid_lines = 0
total_invalid_values = 0
total_invalid_classes = 0

for split in SPLITS:

    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"

    if not images_dir.exists():
        print(f"\nERROR: no existe {images_dir}")
        continue

    if not labels_dir.exists():
        print(f"\nERROR: no existe {labels_dir}")
        continue

    images = [
        p for p in images_dir.iterdir()
        if p.is_file()
    ]

    labels = list(labels_dir.glob("*.txt"))

    counter = Counter()

    invalid_lines = []
    invalid_values = []
    invalid_classes = []

    image_names = {
        p.stem for p in images
    }

    label_names = {
        p.stem for p in labels
    }

    missing_labels = image_names - label_names
    orphan_labels = label_names - image_names

    for label_file in labels:

        for line_number, line in enumerate(
            label_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines(),
            start=1
        ):

            parts = line.strip().split()

            if len(parts) != 5:

                invalid_lines.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

                continue

            cls, x, y, w, h = parts

            try:
                cls_int = int(cls)

                values = [
                    float(x),
                    float(y),
                    float(w),
                    float(h)
                ]

            except ValueError:

                invalid_values.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

                continue

            counter[cls_int] += 1

            # ------------------------------------------------
            # CLASE
            # ------------------------------------------------

            if cls_int not in VALID_CLASSES:

                invalid_classes.append(
                    (
                        label_file.name,
                        line_number,
                        cls_int
                    )
                )

            # ------------------------------------------------
            # COORDENADAS
            # ------------------------------------------------

            if not all(
                0 <= value <= 1
                for value in values
            ):

                invalid_values.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

            # ------------------------------------------------
            # DIMENSIONES
            # ------------------------------------------------

            if values[2] <= 0 or values[3] <= 0:

                invalid_values.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

    objects = sum(counter.values())

    print()
    print("=" * 80)
    print(f"[{split}]")
    print("=" * 80)

    print()
    print(f"Imágenes : {len(images)}")
    print(f"Labels   : {len(labels)}")
    print(f"Objetos  : {objects}")

    print()
    print("Clases:")

    for cls in sorted(counter):

        name = VALID_CLASSES.get(
            cls,
            "INVALID"
        )

        print(
            f"{cls}: "
            f"{name:<10} "
            f"{counter[cls]:>8}"
        )

    print()
    print(
        f"Imágenes sin label : "
        f"{len(missing_labels)}"
    )

    print(
        f"Labels sin imagen  : "
        f"{len(orphan_labels)}"
    )

    print(
        f"Líneas inválidas   : "
        f"{len(invalid_lines)}"
    )

    print(
        f"Valores inválidos  : "
        f"{len(invalid_values)}"
    )

    print(
        f"Clases inválidas   : "
        f"{len(invalid_classes)}"
    )

    total_images += len(images)
    total_labels += len(labels)
    total_objects += objects

    total_missing_labels += len(missing_labels)
    total_orphan_labels += len(orphan_labels)
    total_invalid_lines += len(invalid_lines)
    total_invalid_values += len(invalid_values)
    total_invalid_classes += len(invalid_classes)

    global_classes.update(counter)


# ============================================================
# RESUMEN GLOBAL
# ============================================================

print()
print("=" * 80)
print("RESUMEN GLOBAL")
print("=" * 80)

print()
print(f"Imágenes : {total_images}")
print(f"Labels   : {total_labels}")
print(f"Objetos  : {total_objects}")

print()
print("Distribución global:")

for cls in sorted(global_classes):

    name = VALID_CLASSES.get(
        cls,
        "INVALID"
    )

    print(
        f"{cls}: "
        f"{name:<10} "
        f"{global_classes[cls]:>8}"
    )

print()
print("=" * 80)
print("VALIDACIONES")
print("=" * 80)

# ------------------------------------------------------------
# IMÁGENES / LABELS
# ------------------------------------------------------------

if total_images == total_labels:
    print(
        "OK: número de imágenes y labels coincide."
    )
else:
    print(
        "ERROR: imágenes y labels NO coinciden."
    )

# ------------------------------------------------------------
# HUÉRFANOS
# ------------------------------------------------------------

if total_missing_labels == 0:
    print(
        "OK: no existen imágenes sin label."
    )
else:
    print(
        f"ERROR: existen "
        f"{total_missing_labels} imágenes sin label."
    )

if total_orphan_labels == 0:
    print(
        "OK: no existen labels sin imagen."
    )
else:
    print(
        f"ERROR: existen "
        f"{total_orphan_labels} labels sin imagen."
    )

# ------------------------------------------------------------
# FORMATO
# ------------------------------------------------------------

if total_invalid_lines == 0:
    print(
        "OK: todas las líneas tienen formato YOLO válido."
    )
else:
    print(
        f"ERROR: "
        f"{total_invalid_lines} líneas inválidas."
    )

# ------------------------------------------------------------
# COORDENADAS
# ------------------------------------------------------------

if total_invalid_values == 0:
    print(
        "OK: todas las coordenadas son válidas."
    )
else:
    print(
        f"ERROR: "
        f"{total_invalid_values} valores inválidos."
    )

# ------------------------------------------------------------
# CLASES
# ------------------------------------------------------------

if total_invalid_classes == 0:
    print(
        "OK: todas las clases son 0 o 1."
    )
else:
    print(
        f"ERROR: "
        f"{total_invalid_classes} clases inválidas."
    )

# ------------------------------------------------------------
# CLASES ESPERADAS
# ------------------------------------------------------------

expected_classes = {0, 1}

if set(global_classes) == expected_classes:

    print(
        "OK: aparecen exactamente las "
        "2 clases YOLO: 0-1."
    )

else:

    print(
        "ERROR: distribución de clases inesperada."
    )

# ------------------------------------------------------------
# OTHERS
# ------------------------------------------------------------

if 10 not in global_classes:

    print(
        "OK: no existen objetos 'others' (clase 10)."
    )

else:

    print(
        "ERROR: todavía existen objetos "
        "'others' (clase 10)."
    )

print()
print("=" * 80)

if (
    total_images == total_labels
    and total_missing_labels == 0
    and total_orphan_labels == 0
    and total_invalid_lines == 0
    and total_invalid_values == 0
    and total_invalid_classes == 0
    and set(global_classes) == expected_classes
    and 10 not in global_classes
):

    print("DATASET SARC-DRONE 2 CLASS: OK")
    print("LISTO PARA EL SIGUIENTE PASO.")

else:

    print(
        "DATASET SARC-DRONE 2 CLASS: "
        "REQUIERE REVISIÓN."
    )

print("=" * 80)