from pathlib import Path
import shutil
from collections import Counter

# ============================================================
# CONFIGURACIÓN
# ============================================================

SOURCE = Path(r"processed\converted\VisDrone")
DEST = Path(r"processed\sar\VisDrone_SAR_2CLASS")

SPLITS = ["train", "val", "test_dev"]

# VisDrone convertido -> SARC-Drone
#
# 0 pedestrian        -> 0 person
# 1 people            -> 0 person
#
# 2 bicycle            -> 1 vehicle
# 3 car                -> 1 vehicle
# 4 van                -> 1 vehicle
# 5 truck              -> 1 vehicle
# 6 tricycle           -> 1 vehicle
# 7 awning-tricycle    -> 1 vehicle
# 8 bus                -> 1 vehicle
# 9 motor              -> 1 vehicle
#
# 10 others            -> descartado

CLASS_MAP = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
}

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}


# ============================================================
# VALIDACIONES INICIALES
# ============================================================

print("=" * 80)
print("CREACIÓN DATASET SARC-DRONE - 2 CLASES")
print("=" * 80)

print()
print(f"ORIGEN : {SOURCE.resolve()}")
print(f"DESTINO: {DEST.resolve()}")

if not SOURCE.exists():
    raise SystemExit(
        f"\nERROR: no existe el dataset origen:\n{SOURCE.resolve()}"
    )

if DEST.exists():
    raise SystemExit(
        "\nERROR: el destino ya existe.\n"
        f"No se sobrescribirá:\n{DEST.resolve()}\n\n"
        "Si quieres regenerarlo, elimina primero el destino."
    )

print()
print("OK: dataset origen encontrado.")
print("OK: destino no existe.")
print("OK: no se modificará el dataset original.")


# ============================================================
# CONTADORES GLOBALES
# ============================================================

global_source = Counter()
global_target = Counter()
global_discarded = Counter()

total_images = 0
total_labels = 0
total_objects = 0
total_discarded = 0


# ============================================================
# PROCESAMIENTO
# ============================================================

for split in SPLITS:

    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"

    dest_images = DEST / split / "images"
    dest_labels = DEST / split / "labels"

    if not source_images.exists():
        raise SystemExit(
            f"\nERROR: no existe:\n{source_images.resolve()}"
        )

    if not source_labels.exists():
        raise SystemExit(
            f"\nERROR: no existe:\n{source_labels.resolve()}"
        )

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    source_image_files = [
        p for p in source_images.iterdir()
        if p.is_file()
    ]

    source_label_files = list(source_labels.glob("*.txt"))

    split_source = Counter()
    split_target = Counter()
    split_discarded = Counter()

    split_images = 0
    split_labels = 0
    split_objects = 0
    split_discarded_total = 0

    print()
    print("=" * 80)
    print(f"PROCESANDO: {split}")
    print("=" * 80)

    # --------------------------------------------------------
    # COPIAR IMÁGENES
    # --------------------------------------------------------

    for image_file in source_image_files:

        destination_image = dest_images / image_file.name

        shutil.copy2(
            image_file,
            destination_image
        )

        split_images += 1

    # --------------------------------------------------------
    # CONVERTIR LABELS
    # --------------------------------------------------------

    for label_file in source_label_files:

        output_lines = []

        lines = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):

            parts = line.strip().split()

            if len(parts) != 5:
                print(
                    f"ADVERTENCIA: línea inválida "
                    f"{label_file.name}:{line_number}"
                )
                continue

            try:
                source_class = int(parts[0])
            except ValueError:
                print(
                    f"ADVERTENCIA: clase inválida "
                    f"{label_file.name}:{line_number}"
                )
                continue

            split_source[source_class] += 1
            global_source[source_class] += 1

            # ------------------------------------------------
            # CLASE NO DESEADA: OTHERS
            # ------------------------------------------------

            if source_class not in CLASS_MAP:

                split_discarded[source_class] += 1
                global_discarded[source_class] += 1

                split_discarded_total += 1
                total_discarded += 1

                continue

            target_class = CLASS_MAP[source_class]

            split_target[target_class] += 1
            global_target[target_class] += 1

            split_objects += 1

            # Mantener exactamente las coordenadas originales
            output_lines.append(
                f"{target_class} "
                f"{parts[1]} "
                f"{parts[2]} "
                f"{parts[3]} "
                f"{parts[4]}"
            )

        destination_label = dest_labels / label_file.name

        destination_label.write_text(
            "\n".join(output_lines) + (
                "\n" if output_lines else ""
            ),
            encoding="utf-8"
        )

        split_labels += 1

    # --------------------------------------------------------
    # RESUMEN DEL SPLIT
    # --------------------------------------------------------

    print()
    print(f"Imágenes copiadas : {split_images}")
    print(f"Labels convertidos: {split_labels}")
    print(f"Objetos válidos   : {split_objects}")
    print(f"Objetos descartados: {split_discarded_total}")

    print()
    print("Clases destino:")

    for cls in sorted(CLASS_NAMES):

        print(
            f"  {cls}: "
            f"{CLASS_NAMES[cls]:<10} "
            f"{split_target[cls]:>8}"
        )

    if split_discarded:

        print()
        print("Clases descartadas:")

        for cls in sorted(split_discarded):

            print(
                f"  {cls}: "
                f"{split_discarded[cls]:>8}"
            )

    total_images += split_images
    total_labels += split_labels
    total_objects += split_objects


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print("=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print()
print(f"Imágenes : {total_images:,}")
print(f"Labels   : {total_labels:,}")
print(f"Objetos  : {total_objects:,}")
print(f"Descartados: {total_discarded:,}")

print()
print("DISTRIBUCIÓN FINAL")

for cls in sorted(CLASS_NAMES):

    count = global_target[cls]

    percentage = (
        count / total_objects * 100
        if total_objects
        else 0
    )

    print(
        f"{cls}: "
        f"{CLASS_NAMES[cls]:<10} "
        f"{count:>8,} "
        f"({percentage:6.2f}%)"
    )

print()
print("DISTRIBUCIÓN ORIGINAL UTILIZADA")

for cls in sorted(global_source):

    print(
        f"{cls:>2}: "
        f"{global_source[cls]:>8,}"
    )

print()
print("CLASES DESCARTADAS")

for cls in sorted(global_discarded):

    print(
        f"{cls:>2}: "
        f"{global_discarded[cls]:>8,}"
    )

print()
print("=" * 80)
print("VERIFICACIÓN")
print("=" * 80)

expected_objects = (
    global_source[0]
    + global_source[1]
    + global_source[2]
    + global_source[3]
    + global_source[4]
    + global_source[5]
    + global_source[6]
    + global_source[7]
    + global_source[8]
    + global_source[9]
)

if total_objects == expected_objects:
    print("OK: todos los objetos seleccionados fueron convertidos.")
else:
    print("ERROR: discrepancia en el número de objetos.")

if total_discarded == global_source[10]:
    print("OK: todos los objetos 'others' fueron descartados.")
else:
    print("ERROR: discrepancia en objetos descartados.")

print()
print("Dataset creado en:")
print(DEST.resolve())

print()
print("IMPORTANTE:")
print("El dataset VisDrone original NO ha sido modificado.")

print("=" * 80)
