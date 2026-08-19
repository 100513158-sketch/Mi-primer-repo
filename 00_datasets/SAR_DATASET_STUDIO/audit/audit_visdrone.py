from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(r"processed\converted\VisDrone")

SPLITS = [
    "train",
    "val",
    "test_dev",
]

# Dataset convertido a formato YOLO:
# VisDrone original -> clases 1-11
# YOLO convertido    -> clases 0-10
VALID_CLASSES = set(range(11))


# ============================================================
# CABECERA
# ============================================================

print("=" * 70)
print("AUDITORÍA DATASET VISDRONE")
print("=" * 70)

print()
print(f"Dataset: {ROOT.resolve()}")
print(f"Clases válidas: {sorted(VALID_CLASSES)}")
print()


# ============================================================
# COMPROBACIÓN DEL DATASET
# ============================================================

if not ROOT.exists():
    print("ERROR: no existe el directorio del dataset.")
    print(f"Ruta: {ROOT.resolve()}")
    raise SystemExit(1)


# ============================================================
# TOTALES GLOBALES
# ============================================================

total_images = 0
total_labels = 0
total_objects = 0

global_classes = Counter()

global_invalid_lines = 0
global_invalid_values = 0
global_invalid_classes = 0
global_missing_labels = 0
global_orphan_labels = 0


# ============================================================
# PROCESAR CADA SPLIT
# ============================================================

for split in SPLITS:

    images_dir = ROOT / split / "images"
    labels_dir = ROOT / split / "labels"

    print()
    print("=" * 70)
    print(f"[{split}]")
    print("=" * 70)

    # --------------------------------------------------------
    # Comprobar directorios
    # --------------------------------------------------------

    if not images_dir.exists():
        print(f"ERROR: no existe: {images_dir}")
        continue

    if not labels_dir.exists():
        print(f"ERROR: no existe: {labels_dir}")
        continue

    # --------------------------------------------------------
    # Obtener imágenes y labels
    # --------------------------------------------------------

    images = [
        p for p in images_dir.iterdir()
        if p.is_file()
    ]

    labels = [
        p for p in labels_dir.glob("*.txt")
        if p.is_file()
    ]

    # --------------------------------------------------------
    # Contadores del split
    # --------------------------------------------------------

    counter = Counter()

    invalid_lines = []
    invalid_values = []
    invalid_classes = []

    # --------------------------------------------------------
    # Comparar imágenes y labels
    # --------------------------------------------------------

    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}

    missing_labels = image_names - label_names
    orphan_labels = label_names - image_names

    # --------------------------------------------------------
    # Procesar cada archivo de labels
    # --------------------------------------------------------

    for label_file in labels:

        try:
            lines = label_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

        except Exception as exc:

            print(
                f"ERROR leyendo {label_file}: {exc}"
            )

            continue

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            # ------------------------------------------------
            # Línea vacía
            # ------------------------------------------------

            if not line:
                continue

            parts = line.split()

            # ------------------------------------------------
            # Formato YOLO:
            #
            # class x_center y_center width height
            #
            # exactamente 5 valores
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Validar clase y coordenadas numéricas
            # ------------------------------------------------

            try:

                cls_int = int(cls)

                x_value = float(x)
                y_value = float(y)
                w_value = float(w)
                h_value = float(h)

            except ValueError:

                invalid_values.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

                continue

            # ------------------------------------------------
            # Registrar clase
            # ------------------------------------------------

            counter[str(cls_int)] += 1

            # ------------------------------------------------
            # Validar clase
            #
            # YOLO actual:
            # 0 ... 10
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
            # Validar coordenadas
            #
            # YOLO normalizado:
            #
            # 0 <= x <= 1
            # 0 <= y <= 1
            # 0 <= w <= 1
            # 0 <= h <= 1
            # ------------------------------------------------

            values = [
                x_value,
                y_value,
                w_value,
                h_value
            ]

            if not all(0 <= value <= 1 for value in values):

                invalid_values.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

            # ------------------------------------------------
            # Ancho y alto deben ser > 0
            # ------------------------------------------------

            if w_value <= 0 or h_value <= 0:

                invalid_values.append(
                    (
                        label_file.name,
                        line_number,
                        line
                    )
                )

    # ========================================================
    # RESULTADOS DEL SPLIT
    # ========================================================

    split_objects = sum(counter.values())

    print()
    print(f"Imágenes : {len(images)}")
    print(f"Labels   : {len(labels)}")
    print(f"Objetos  : {split_objects}")

    print()
    print("Clases:")

    for cls in sorted(counter, key=int):

        print(
            f"  {cls:>2}: {counter[cls]}"
        )

    print()
    print(
        f"Imágenes sin label : {len(missing_labels)}"
    )

    print(
        f"Labels sin imagen  : {len(orphan_labels)}"
    )

    print(
        f"Líneas inválidas   : {len(invalid_lines)}"
    )

    print(
        f"Valores inválidos  : {len(invalid_values)}"
    )

    print(
        f"Clases inválidas   : {len(invalid_classes)}"
    )

    # ========================================================
    # ACUMULAR TOTALES
    # ========================================================

    total_images += len(images)
    total_labels += len(labels)
    total_objects += split_objects

    global_classes.update(counter)

    global_invalid_lines += len(invalid_lines)
    global_invalid_values += len(invalid_values)
    global_invalid_classes += len(invalid_classes)

    global_missing_labels += len(missing_labels)
    global_orphan_labels += len(orphan_labels)


# ============================================================
# RESUMEN GLOBAL
# ============================================================

print()
print("=" * 70)
print("RESUMEN GLOBAL")
print("=" * 70)

print()
print(f"Imágenes : {total_images}")
print(f"Labels   : {total_labels}")
print(f"Objetos  : {total_objects}")

print()
print("Distribución global:")

for cls in sorted(global_classes, key=int):

    print(
        f"  {cls:>2}: {global_classes[cls]}"
    )


# ============================================================
# VALIDACIONES GLOBALES
# ============================================================

print()
print("=" * 70)
print("VALIDACIONES")
print("=" * 70)

# ------------------------------------------------------------
# Imágenes vs labels
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
# Imágenes sin label
# ------------------------------------------------------------

if global_missing_labels == 0:

    print(
        "OK: no existen imágenes sin label."
    )

else:

    print(
        f"ERROR: existen {global_missing_labels} "
        f"imágenes sin label."
    )


# ------------------------------------------------------------
# Labels huérfanos
# ------------------------------------------------------------

if global_orphan_labels == 0:

    print(
        "OK: no existen labels sin imagen."
    )

else:

    print(
        f"ERROR: existen {global_orphan_labels} "
        f"labels sin imagen."
    )


# ------------------------------------------------------------
# Líneas inválidas
# ------------------------------------------------------------

if global_invalid_lines == 0:

    print(
        "OK: todas las líneas tienen formato YOLO válido."
    )

else:

    print(
        f"ERROR: existen {global_invalid_lines} "
        f"líneas inválidas."
    )


# ------------------------------------------------------------
# Valores inválidos
# ------------------------------------------------------------

if global_invalid_values == 0:

    print(
        "OK: todas las coordenadas son válidas."
    )

else:

    print(
        f"ERROR: existen {global_invalid_values} "
        f"valores/coordenadas inválidos."
    )


# ------------------------------------------------------------
# Clases inválidas
# ------------------------------------------------------------

if global_invalid_classes == 0:

    print(
        "OK: todas las clases están entre 0 y 10."
    )

else:

    print(
        f"ERROR: existen {global_invalid_classes} "
        f"objetos con clases inválidas."
    )


# ============================================================
# COMPROBAR EXACTAMENTE LAS 11 CLASES
# ============================================================

expected_classes = {
    str(i) for i in range(11)
}

detected_classes = set(global_classes.keys())

missing_classes = expected_classes - detected_classes
unexpected_classes = detected_classes - expected_classes


print()
print("=" * 70)
print("COMPROBACIÓN DE CLASES")
print("=" * 70)


if not missing_classes and not unexpected_classes:

    print(
        "OK: aparecen exactamente las 11 clases YOLO: 0-10."
    )

else:

    if missing_classes:

        print(
            "ADVERTENCIA: faltan las clases:"
        )

        for cls in sorted(
            missing_classes,
            key=int
        ):

            print(
                f"  {cls}"
            )

    if unexpected_classes:

        print(
            "ADVERTENCIA: aparecen clases inesperadas:"
        )

        for cls in sorted(
            unexpected_classes,
            key=int
        ):

            print(
                f"  {cls}"
            )


# ============================================================
# RESULTADO FINAL
# ============================================================

has_errors = any(
    [
        total_images != total_labels,
        global_missing_labels != 0,
        global_orphan_labels != 0,
        global_invalid_lines != 0,
        global_invalid_values != 0,
        global_invalid_classes != 0,
        bool(missing_classes),
        bool(unexpected_classes),
    ]
)


print()
print("=" * 70)

if has_errors:

    print(
        "RESULTADO FINAL: REVISAR DATASET"
    )

else:

    print(
        "RESULTADO FINAL: DATASET CORRECTO"
    )

print("=" * 70)