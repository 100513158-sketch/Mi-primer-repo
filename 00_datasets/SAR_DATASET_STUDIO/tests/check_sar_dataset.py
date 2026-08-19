from pathlib import Path
import yaml


# ============================================================
# LOCALIZACIÓN DEL PROYECTO
# ============================================================

# Este archivo está en:
#
# DATASET_STUDIO/
# └── tests/
#     └── check_sar_dataset.py
#
# Por tanto:
# parents[0] = tests
# parents[1] = DATASET_STUDIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YAML_FILE = (
    PROJECT_ROOT
    / "configs"
    / "sar_visdrone_2class.yaml"
)


print("=" * 70)
print("VERIFICACIÓN DATASET SARC-DRONE")
print("=" * 70)

print()
print(f"Proyecto: {PROJECT_ROOT}")
print(f"YAML    : {YAML_FILE}")


# ============================================================
# 1. CONFIGURACIÓN YAML
# ============================================================

print()
print("[1] CONFIGURACIÓN YAML")

if not YAML_FILE.exists():

    print("ERROR: no existe el archivo YAML:")
    print(YAML_FILE)

    raise SystemExit(1)


with open(
    YAML_FILE,
    "r",
    encoding="utf-8"
) as f:

    config = yaml.safe_load(f)


print("OK: YAML encontrado")

print(f"path : {config.get('path')}")
print(f"train: {config.get('train')}")
print(f"val  : {config.get('val')}")
print(f"test : {config.get('test')}")
print(f"names: {config.get('names')}")


# ============================================================
# 2. DATASET ROOT
# ============================================================

dataset_root = Path(config["path"])


print()
print("[2] DATASET")

if not dataset_root.exists():

    print("ERROR: no existe el dataset:")
    print(dataset_root)

    raise SystemExit(1)


print("OK: dataset encontrado")
print(dataset_root)


# ============================================================
# 3. ESTRUCTURA DE LOS SPLITS
# ============================================================

print()
print("[3] ESTRUCTURA")


splits = {
    "train": config["train"],
    "val": config["val"],
    "test": config["test"],
}


total_images = 0
total_labels = 0


for name, relative_path in splits.items():

    images_dir = dataset_root / relative_path

    labels_dir = (
        images_dir.parent
        / "labels"
    )


    print()
    print(f"[{name}]")

    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")


    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    if not images_dir.exists():

        print(
            "ERROR: directorio de imágenes "
            "no existe"
        )

        continue


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    if not labels_dir.exists():

        print(
            "ERROR: directorio de labels "
            "no existe"
        )

        continue


    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file()
    ]


    labels = list(
        labels_dir.glob("*.txt")
    )


    print(
        f"Imágenes : {len(images)}"
    )

    print(
        f"Labels   : {len(labels)}"
    )


    total_images += len(images)
    total_labels += len(labels)


    # --------------------------------------------------------
    # CORRESPONDENCIA
    # --------------------------------------------------------

    image_names = {
        p.stem
        for p in images
    }


    label_names = {
        p.stem
        for p in labels
    }


    missing_labels = (
        image_names
        - label_names
    )


    orphan_labels = (
        label_names
        - image_names
    )


    print(
        f"Sin label : "
        f"{len(missing_labels)}"
    )


    print(
        f"Sin imagen: "
        f"{len(orphan_labels)}"
    )


    if (
        not missing_labels
        and not orphan_labels
    ):

        print(
            "OK: correspondencia "
            "imágenes/labels correcta"
        )

    else:

        print(
            "ERROR: existe una "
            "correspondencia incorrecta"
        )


# ============================================================
# 4. CLASES
# ============================================================

print()
print("[4] CLASES")


names = config["names"]


print(
    f"Clases configuradas: "
    f"{len(names)}"
)


for cls, name in names.items():

    print(
        f"{cls}: {name}"
    )


if len(names) == 2:

    print(
        "OK: exactamente 2 clases"
    )

else:

    print(
        "ERROR: se esperaban "
        "exactamente 2 clases"
    )


if (
    names.get(0) == "person"
    and
    names.get(1) == "vehicle"
):

    print(
        "OK: clases person/vehicle "
        "correctamente configuradas"
    )

else:

    print(
        "ERROR: nombres de clases "
        "incorrectos"
    )


# ============================================================
# 5. RESUMEN
# ============================================================

print()
print("=" * 70)
print("RESUMEN")
print("=" * 70)

print(
    f"Total imágenes: {total_images}"
)

print(
    f"Total labels  : {total_labels}"
)


if total_images == total_labels:

    print(
        "OK: imágenes y labels coinciden"
    )

else:

    print(
        "ERROR: imágenes y labels "
        "NO coinciden"
    )


print()
print("=" * 70)
print("VERIFICACIÓN FINALIZADA")
print("=" * 70)