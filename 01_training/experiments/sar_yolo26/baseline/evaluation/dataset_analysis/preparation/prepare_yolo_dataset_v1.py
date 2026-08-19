from pathlib import Path
from collections import Counter
import csv
import math
import sys


# ============================================================
# SAR YOLO26 - PREPARE YOLO DATASET V1
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed"
    r"\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

WORKSPACE_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "preparation"
    / "prepare_yolo_dataset_v1"
)

REPORT_DIR = OUTPUT_DIR / "reports"

DATA_YAML = OUTPUT_DIR / "data.yaml"
SUMMARY_FILE = OUTPUT_DIR / "DATASET_PREPARATION_SUMMARY.txt"
STATS_CSV = OUTPUT_DIR / "dataset_statistics.csv"


# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

SPLITS = [
    "train",
    "val",
    "test_dev",
]

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

TINY_16 = 16
TINY_32 = 32
TINY_64 = 64

CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500


# ============================================================
# UTILIDADES
# ============================================================

def print_separator():
    print("=" * 70)


def find_images(images_dir):
    if not images_dir.exists():
        return []

    return sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def label_path_for_image(image_path, images_dir, labels_dir):
    relative = image_path.relative_to(images_dir)
    return labels_dir / relative.with_suffix(".txt")


def parse_label_line(line):
    parts = line.strip().split()

    if len(parts) != 5:
        return None

    try:
        cls = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return None

    return cls, xc, yc, w, h


def bbox_area_pixels(xc, yc, w, h, image_width, image_height):
    bw = w * image_width
    bh = h * image_height

    return bw * bh


def try_get_image_size(path):
    """
    Intenta obtener dimensiones mediante PIL.

    PIL no es obligatorio para ejecutar toda la preparación.
    Si no está disponible, devuelve None.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size

    except Exception:
        return None


# ============================================================
# VALIDACIÓN DE LABELS
# ============================================================

def validate_label_file(
    label_path,
    image_path,
):
    result = {
        "lines": 0,
        "objects": 0,
        "person": 0,
        "vehicle": 0,
        "invalid": 0,
        "invalid_class": 0,
        "invalid_coordinate": 0,
        "invalid_bbox": 0,
        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "area_sum": 0.0,
    }

    image_size = try_get_image_size(image_path)

    if not label_path.exists():
        return result, "missing_label"

    try:
        text = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        result["invalid"] += 1
        return result, "read_error"

    lines = text.splitlines()

    for line in lines:

        if not line.strip():
            continue

        result["lines"] += 1

        parsed = parse_label_line(line)

        if parsed is None:
            result["invalid"] += 1
            continue

        cls, xc, yc, w, h = parsed

        if cls not in CLASS_NAMES:
            result["invalid_class"] += 1
            continue

        if not all(
            math.isfinite(v)
            for v in [xc, yc, w, h]
        ):
            result["invalid_coordinate"] += 1
            continue

        if (
            xc < 0
            or xc > 1
            or yc < 0
            or yc > 1
        ):
            result["invalid_coordinate"] += 1
            continue

        if w <= 0 or h <= 0:
            result["invalid_bbox"] += 1
            continue

        if image_size:
            image_width, image_height = image_size

            area = bbox_area_pixels(
                xc,
                yc,
                w,
                h,
                image_width,
                image_height,
            )

            result["area_sum"] += area

            if area < TINY_16:
                result["tiny16"] += 1

            if area < TINY_32:
                result["tiny32"] += 1

            if area < TINY_64:
                result["tiny64"] += 1

        result["objects"] += 1

        if cls == 0:
            result["person"] += 1

        elif cls == 1:
            result["vehicle"] += 1

    errors = (
        result["invalid"]
        + result["invalid_class"]
        + result["invalid_coordinate"]
        + result["invalid_bbox"]
    )

    if errors:
        status = "invalid"
    else:
        status = "ok"

    return result, status


# ============================================================
# ANÁLISIS DE UN SPLIT
# ============================================================

def analyze_split(split):

    split_dir = DATASET_ROOT / split

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    print()
    print(f"## Analizando: {split}")

    if not split_dir.exists():
        print(f"[ERROR] No existe: {split_dir}")
        return None

    if not images_dir.exists():
        print(f"[ERROR] No existe: {images_dir}")
        return None

    if not labels_dir.exists():
        print(f"[ERROR] No existe: {labels_dir}")
        return None

    images = find_images(images_dir)

    print(f"Imágenes encontradas: {len(images):,}")

    stats = {
        "split": split,
        "images": len(images),
        "objects": 0,
        "person": 0,
        "vehicle": 0,
        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "invalid": 0,
        "invalid_class": 0,
        "invalid_coordinate": 0,
        "invalid_bbox": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "crowded100": 0,
        "crowded200": 0,
        "crowded300": 0,
        "crowded500": 0,
        "image_records": [],
    }

    for index, image_path in enumerate(images, start=1):

        label_path = label_path_for_image(
            image_path,
            images_dir,
            labels_dir,
        )

        result, status = validate_label_file(
            label_path,
            image_path,
        )

        if status == "missing_label":
            stats["missing_labels"] += 1

        if label_path.exists():
            try:
                if not label_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip():
                    stats["empty_labels"] += 1
            except Exception:
                pass

        stats["objects"] += result["objects"]
        stats["person"] += result["person"]
        stats["vehicle"] += result["vehicle"]

        stats["tiny16"] += result["tiny16"]
        stats["tiny32"] += result["tiny32"]
        stats["tiny64"] += result["tiny64"]

        stats["invalid"] += result["invalid"]
        stats["invalid_class"] += result["invalid_class"]
        stats["invalid_coordinate"] += result["invalid_coordinate"]
        stats["invalid_bbox"] += result["invalid_bbox"]

        objects = result["objects"]

        if objects >= CROWDED_100:
            stats["crowded100"] += 1

        if objects >= CROWDED_200:
            stats["crowded200"] += 1

        if objects >= CROWDED_300:
            stats["crowded300"] += 1

        if objects >= CROWDED_500:
            stats["crowded500"] += 1

        stats["image_records"].append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "objects": objects,
                "person": result["person"],
                "vehicle": result["vehicle"],
                "tiny16": result["tiny16"],
                "tiny32": result["tiny32"],
                "tiny64": result["tiny64"],
                "status": status,
            }
        )

        if index % 1000 == 0:
            print(
                f"Procesadas: {index:,}/{len(images):,}"
            )

    return stats


# ============================================================
# LABELS HUÉRFANOS
# ============================================================

def find_orphan_labels(split):

    split_dir = DATASET_ROOT / split

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not labels_dir.exists():
        return []

    image_relative = set()

    for image in find_images(images_dir):
        relative = image.relative_to(images_dir)
        image_relative.add(
            relative.with_suffix(".txt")
        )

    orphan_labels = []

    for label in labels_dir.rglob("*.txt"):

        relative = label.relative_to(labels_dir)

        if relative not in image_relative:
            orphan_labels.append(label)

    return sorted(orphan_labels)


# ============================================================
# DATA.YAML
# ============================================================

def create_data_yaml():

    # Usamos una ruta absoluta para evitar ambigüedades
    # durante el primer entrenamiento.

    train_path = DATASET_ROOT / "train" / "images"
    val_path = DATASET_ROOT / "val" / "images"
    test_path = DATASET_ROOT / "test_dev" / "images"

    content = f"""# SAR YOLO26 - VisDrone SAR 2CLASS V1

path: {DATASET_ROOT.as_posix()}

train: train/images
val: val/images
test: test_dev/images

nc: 2

names:
  0: person
  1: vehicle
"""

    DATA_YAML.write_text(
        content,
        encoding="utf-8",
    )


# ============================================================
# CSV ESTADÍSTICAS
# ============================================================

def create_statistics_csv(all_stats):

    rows = []

    for stats in all_stats:

        rows.append(
            {
                "split": stats["split"],
                "images": stats["images"],
                "objects": stats["objects"],
                "objects_per_image": (
                    stats["objects"] / stats["images"]
                    if stats["images"]
                    else 0
                ),
                "person": stats["person"],
                "vehicle": stats["vehicle"],
                "tiny16": stats["tiny16"],
                "tiny32": stats["tiny32"],
                "tiny64": stats["tiny64"],
                "crowded100": stats["crowded100"],
                "crowded200": stats["crowded200"],
                "crowded300": stats["crowded300"],
                "crowded500": stats["crowded500"],
                "invalid": stats["invalid"],
                "invalid_class": stats["invalid_class"],
                "invalid_coordinate": stats["invalid_coordinate"],
                "invalid_bbox": stats["invalid_bbox"],
                "missing_labels": stats["missing_labels"],
                "empty_labels": stats["empty_labels"],
            }
        )

    with STATS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RESUMEN
# ============================================================

def create_summary(
    all_stats,
    orphan_labels,
):

    total_images = sum(
        s["images"]
        for s in all_stats
    )

    total_objects = sum(
        s["objects"]
        for s in all_stats
    )

    total_person = sum(
        s["person"]
        for s in all_stats
    )

    total_vehicle = sum(
        s["vehicle"]
        for s in all_stats
    )

    total_tiny16 = sum(
        s["tiny16"]
        for s in all_stats
    )

    total_tiny32 = sum(
        s["tiny32"]
        for s in all_stats
    )

    total_tiny64 = sum(
        s["tiny64"]
        for s in all_stats
    )

    total_invalid = sum(
        s["invalid"]
        for s in all_stats
    )

    total_invalid_class = sum(
        s["invalid_class"]
        for s in all_stats
    )

    total_invalid_coordinate = sum(
        s["invalid_coordinate"]
        for s in all_stats
    )

    total_invalid_bbox = sum(
        s["invalid_bbox"]
        for s in all_stats
    )

    total_missing = sum(
        s["missing_labels"]
        for s in all_stats
    )

    total_empty = sum(
        s["empty_labels"]
        for s in all_stats
    )

    total_crowded100 = sum(
        s["crowded100"]
        for s in all_stats
    )

    total_crowded200 = sum(
        s["crowded200"]
        for s in all_stats
    )

    total_crowded300 = sum(
        s["crowded300"]
        for s in all_stats
    )

    total_crowded500 = sum(
        s["crowded500"]
        for s in all_stats
    )

    if total_images:
        objects_per_image = (
            total_objects / total_images
        )
    else:
        objects_per_image = 0

    if total_objects:
        person_pct = (
            total_person / total_objects
        ) * 100

        vehicle_pct = (
            total_vehicle / total_objects
        ) * 100
    else:
        person_pct = 0
        vehicle_pct = 0

    valid = (
        total_invalid == 0
        and total_invalid_class == 0
        and total_invalid_coordinate == 0
        and total_invalid_bbox == 0
        and total_missing == 0
        and len(orphan_labels) == 0
    )

    status = "READY" if valid else "REVIEW_REQUIRED"

    lines = []

    lines.append(
        "SAR YOLO26 - DATASET PREPARATION V1"
    )

    lines.append("")
    lines.append("=" * 70)

    lines.append(
        f"Dataset:\n{DATASET_ROOT}"
    )

    lines.append(
        f"\nEstado final: {status}"
    )

    lines.append("")
    lines.append("RESUMEN GLOBAL")
    lines.append("-" * 70)

    lines.append(
        f"Imágenes:              {total_images:,}"
    )

    lines.append(
        f"Objetos:               {total_objects:,}"
    )

    lines.append(
        f"Objetos/imagen:        {objects_per_image:.2f}"
    )

    lines.append("")
    lines.append("CLASES")

    lines.append(
        f"Personas:              {total_person:,} "
        f"({person_pct:.2f} %)"
    )

    lines.append(
        f"Vehículos:             {total_vehicle:,} "
        f"({vehicle_pct:.2f} %)"
    )

    lines.append("")
    lines.append("OBJETOS PEQUEÑOS")

    lines.append(
        f"<16 px²:               {total_tiny16:,}"
    )

    lines.append(
        f"<32 px²:               {total_tiny32:,}"
    )

    lines.append(
        f"<64 px²:               {total_tiny64:,}"
    )

    lines.append("")
    lines.append("CROWDED")

    lines.append(
        f">=100 objetos:          {total_crowded100:,} imágenes"
    )

    lines.append(
        f">=200 objetos:          {total_crowded200:,} imágenes"
    )

    lines.append(
        f">=300 objetos:          {total_crowded300:,} imágenes"
    )

    lines.append(
        f">=500 objetos:          {total_crowded500:,} imágenes"
    )

    lines.append("")
    lines.append("INTEGRIDAD")

    lines.append(
        f"Labels inválidos:      {total_invalid:,}"
    )

    lines.append(
        f"Clases inválidas:      {total_invalid_class:,}"
    )

    lines.append(
        f"Coordenadas inválidas: {total_invalid_coordinate:,}"
    )

    lines.append(
        f"BBoxes inválidas:      {total_invalid_bbox:,}"
    )

    lines.append(
        f"Imágenes sin labels:   {total_missing:,}"
    )

    lines.append(
        f"Labels sin imágenes:   {len(orphan_labels):,}"
    )

    lines.append(
        f"Labels vacíos:         {total_empty:,}"
    )

    lines.append("")
    lines.append("SPLITS")

    for stats in all_stats:

        lines.append(
            f"{stats['split']:10s} "
            f"images={stats['images']:6,} "
            f"objects={stats['objects']:9,} "
            f"person={stats['person']:8,} "
            f"vehicle={stats['vehicle']:8,}"
        )

    lines.append("")
    lines.append("=" * 70)

    lines.append(
        "CONFIGURACIÓN YOLO"
    )

    lines.append(
        f"nc: 2"
    )

    lines.append(
        "class 0: person"
    )

    lines.append(
        "class 1: vehicle"
    )

    lines.append(
        f"train: {DATASET_ROOT / 'train' / 'images'}"
    )

    lines.append(
        f"val:   {DATASET_ROOT / 'val' / 'images'}"
    )

    lines.append(
        f"test:  {DATASET_ROOT / 'test_dev' / 'images'}"
    )

    lines.append("")
    lines.append("=" * 70)

    lines.append(
        "El dataset original y el dataset limpio V1 "
        "NO han sido modificados por este script."
    )

    SUMMARY_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("# SAR YOLO26 - PREPARE YOLO DATASET V1")
    print()

    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()

    # --------------------------------------------------------
    # DATASET ROOT
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print(
            "[ERROR] No existe DATASET_ROOT:"
        )

        print(DATASET_ROOT)

        sys.exit(1)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # COMPROBAR SPLITS
    # --------------------------------------------------------

    all_stats = []

    for split in SPLITS:

        stats = analyze_split(split)

        if stats is None:
            print(
                f"[ERROR] Split inválido: {split}"
            )
            sys.exit(1)

        all_stats.append(stats)

    # --------------------------------------------------------
    # LABELS HUÉRFANOS
    # --------------------------------------------------------

    print()
    print("Buscando labels sin imágenes...")

    orphan_labels = []

    for split in SPLITS:

        found = find_orphan_labels(split)

        orphan_labels.extend(found)

        if found:
            print(
                f"[WARNING] {split}: "
                f"{len(found):,} labels huérfanos"
            )

    # --------------------------------------------------------
    # GENERAR DATA.YAML
    # --------------------------------------------------------

    print()
    print("Generando data.yaml...")

    create_data_yaml()

    print(
        f"[OK] {DATA_YAML}"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    create_statistics_csv(
        all_stats
    )

    print(
        f"[OK] {STATS_CSV}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    create_summary(
        all_stats,
        orphan_labels,
    )

    print(
        f"[OK] {SUMMARY_FILE}"
    )

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    total_images = sum(
        s["images"]
        for s in all_stats
    )

    total_objects = sum(
        s["objects"]
        for s in all_stats
    )

    total_invalid = sum(
        s["invalid"]
        + s["invalid_class"]
        + s["invalid_coordinate"]
        + s["invalid_bbox"]
        for s in all_stats
    )

    total_missing = sum(
        s["missing_labels"]
        for s in all_stats
    )

    print()
    print_separator()

    print(
        "# RESULTADO PREPARACIÓN YOLO V1"
    )

    print_separator()

    print()
    print(
        f"Imágenes:              {total_images:,}"
    )

    print(
        f"Objetos:               {total_objects:,}"
    )

    print(
        f"Objetos/imagen:        "
        f"{total_objects / total_images:.2f}"
        if total_images
        else "Objetos/imagen:        0.00"
    )

    print()
    print("CLASES")

    total_person = sum(
        s["person"]
        for s in all_stats
    )

    total_vehicle = sum(
        s["vehicle"]
        for s in all_stats
    )

    print(
        f"Personas:              {total_person:,}"
    )

    print(
        f"Vehículos:             {total_vehicle:,}"
    )

    print()
    print("INTEGRIDAD")

    print(
        f"Errores de labels:     {total_invalid:,}"
    )

    print(
        f"Labels sin imágenes:   {len(orphan_labels):,}"
    )

    print(
        f"Imágenes sin labels:   {total_missing:,}"
    )

    print()

    if (
        total_invalid == 0
        and len(orphan_labels) == 0
        and total_missing == 0
    ):

        print(
            "[OK] DATASET PREPARADO PARA YOLO26"
        )

    else:

        print(
            "[WARNING] EL DATASET REQUIERE REVISIÓN"
        )

    print()
    print("Configuración:")
    print(DATA_YAML)

    print()
    print("Resumen:")
    print(SUMMARY_FILE)

    print()
    print("Estadísticas:")
    print(STATS_CSV)

    print()
    print(
        "IMPORTANTE: este script SOLO prepara "
        "la configuración de entrenamiento."
    )

    print(
        "No elimina, copia ni modifica imágenes o labels."
    )

    print_separator()


if __name__ == "__main__":
    main()