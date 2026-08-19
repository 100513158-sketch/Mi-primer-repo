from pathlib import Path
from collections import Counter
import csv
import math

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO"
    r"\processed\sar\VisDrone_SAR_2CLASS"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"baseline\evaluation\dataset_analysis"
)

SPLITS = ["train", "val", "test", "test_dev"]

# Clases del dataset SAR 2 CLASS
CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}

# Umbrales para objetos pequeños
SMALL_THRESHOLDS = [
    100,
    250,
    500,
    1000,
    2500,
]

# ============================================================
# FUNCIONES
# ============================================================

def percentile(values, p):
    """Calcula percentil sin depender de numpy."""
    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return values[f] * (c - k) + values[c] * (k - f)


def analyze_split(split_dir):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        print(f"[WARN] No existe: {images_dir}")
        return None

    if not labels_dir.exists():
        print(f"[WARN] No existe: {labels_dir}")
        return None

    image_files = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_files.extend(images_dir.rglob(ext))

    image_files = sorted(image_files)

    stats = {
        "split": split_dir.name,
        "images": 0,
        "images_with_person": 0,
        "images_with_vehicle": 0,
        "images_without_objects": 0,
        "persons": 0,
        "vehicles": 0,
        "total_objects": 0,
        "objects_per_image": [],
        "person_areas": [],
        "vehicle_areas": [],
        "image_rows": [],
    }

    for image_path in image_files:

        label_path = labels_dir / f"{image_path.stem}.txt"

        stats["images"] += 1

        person_count = 0
        vehicle_count = 0
        total_count = 0

        person_areas = []
        vehicle_areas = []

        if label_path.exists():

            try:
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()

                for line in lines:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])

                    # YOLO:
                    # class x_center y_center width height
                    width = float(parts[3])
                    height = float(parts[4])

                    # Área relativa normalizada.
                    area = width * height

                    total_count += 1

                    if class_id == 0:
                        person_count += 1
                        person_areas.append(area)

                    elif class_id == 1:
                        vehicle_count += 1
                        vehicle_areas.append(area)

            except Exception as exc:
                print(
                    f"[ERROR] {label_path}: {exc}"
                )

        stats["persons"] += person_count
        stats["vehicles"] += vehicle_count
        stats["total_objects"] += total_count

        stats["objects_per_image"].append(total_count)

        stats["person_areas"].extend(person_areas)
        stats["vehicle_areas"].extend(vehicle_areas)

        if person_count > 0:
            stats["images_with_person"] += 1

        if vehicle_count > 0:
            stats["images_with_vehicle"] += 1

        if total_count == 0:
            stats["images_without_objects"] += 1

        stats["image_rows"].append({
            "image": str(image_path),
            "persons": person_count,
            "vehicles": vehicle_count,
            "total_objects": total_count,
            "max_person_area": (
                max(person_areas)
                if person_areas else 0
            ),
            "min_person_area": (
                min(person_areas)
                if person_areas else 0
            ),
        })

    return stats


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


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("SAR YOLO26 - DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET_ROOT)

    print(f"\nOutput:")
    print(OUTPUT_ROOT)

    all_stats = []

    for split in SPLITS:

        split_dir = DATASET_ROOT / split

        if not split_dir.exists():
            print(
                f"\n[INFO] Split no encontrado: {split}"
            )
            continue

        print(
            f"\nAnalizando: {split}"
        )

        stats = analyze_split(split_dir)

        if stats is not None:
            all_stats.append(stats)

    # ========================================================
    # RESUMEN GENERAL
    # ========================================================

    summary_rows = []

    total_images = 0
    total_persons = 0
    total_vehicles = 0
    total_objects = 0

    all_person_areas = []
    all_vehicle_areas = []
    all_objects_per_image = []

    for stats in all_stats:

        persons = stats["persons"]
        vehicles = stats["vehicles"]

        total_images += stats["images"]
        total_persons += persons
        total_vehicles += vehicles
        total_objects += stats["total_objects"]

        all_person_areas.extend(
            stats["person_areas"]
        )

        all_vehicle_areas.extend(
            stats["vehicle_areas"]
        )

        all_objects_per_image.extend(
            stats["objects_per_image"]
        )

        summary_rows.append({
            "split": stats["split"],
            "images": stats["images"],
            "persons": persons,
            "vehicles": vehicles,
            "total_objects": stats["total_objects"],
            "mean_objects_per_image": (
                sum(stats["objects_per_image"])
                / len(stats["objects_per_image"])
                if stats["objects_per_image"]
                else 0
            ),
            "images_with_person": stats[
                "images_with_person"
            ],
            "images_with_vehicle": stats[
                "images_with_vehicle"
            ],
            "images_without_objects": stats[
                "images_without_objects"
            ],
        })

    # ========================================================
    # ESTADÍSTICAS DE ÁREAS
    # ========================================================

    area_rows = []

    for class_name, areas in [
        ("person", all_person_areas),
        ("vehicle", all_vehicle_areas),
    ]:

        row = {
            "class": class_name,
            "objects": len(areas),
            "mean_area": (
                sum(areas) / len(areas)
                if areas else 0
            ),
            "median_area": percentile(
                areas, 50
            ),
            "p25_area": percentile(
                areas, 25
            ),
            "p75_area": percentile(
                areas, 75
            ),
            "p90_area": percentile(
                areas, 90
            ),
            "p95_area": percentile(
                areas, 95
            ),
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
        }

        for threshold in SMALL_THRESHOLDS:

            count = sum(
                1
                for area in areas
                if area < threshold
            )

            percentage = (
                count / len(areas) * 100
                if areas
                else 0
            )

            row[
                f"below_{threshold}"
            ] = count

            row[
                f"below_{threshold}_pct"
            ] = percentage

        area_rows.append(row)

    # ========================================================
    # ESTADÍSTICAS DE CONGESTIÓN
    # ========================================================

    crowded_rows = []

    thresholds = [
        10,
        25,
        50,
        75,
        100,
        150,
        200,
    ]

    for threshold in thresholds:

        count = sum(
            1
            for x in all_objects_per_image
            if x >= threshold
        )

        percentage = (
            count / len(all_objects_per_image) * 100
            if all_objects_per_image
            else 0
        )

        crowded_rows.append({
            "threshold_objects": threshold,
            "images": count,
            "percentage": percentage,
        })

    # ========================================================
    # IMÁGENES MÁS DENSAS
    # ========================================================

    image_rows = []

    for stats in all_stats:
        image_rows.extend(
            stats["image_rows"]
        )

    image_rows.sort(
        key=lambda x: x["total_objects"],
        reverse=True
    )

    # ========================================================
    # CSV
    # ========================================================

    write_csv(
        OUTPUT_ROOT / "reports" / "dataset_summary.csv",
        summary_rows,
        summary_rows[0].keys()
        if summary_rows else []
    )

    write_csv(
        OUTPUT_ROOT / "reports" / "bbox_statistics.csv",
        area_rows,
        area_rows[0].keys()
        if area_rows else []
    )

    write_csv(
        OUTPUT_ROOT / "reports" / "crowded_scenes.csv",
        crowded_rows,
        crowded_rows[0].keys()
        if crowded_rows else []
    )

    write_csv(
        OUTPUT_ROOT / "reports" / "image_statistics.csv",
        image_rows,
        image_rows[0].keys()
        if image_rows else []
    )

    # ========================================================
    # TOP 100 IMÁGENES MÁS DENSAS
    # ========================================================

    write_csv(
        OUTPUT_ROOT / "reports" / "top_100_crowded_images.csv",
        image_rows[:100],
        image_rows[0].keys()
        if image_rows else []
    )

    # ========================================================
    # INFORME CONSOLA
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESULTADO GENERAL")
    print("=" * 70)

    print(
        f"Imágenes:          {total_images:,}"
    )

    print(
        f"Personas:          {total_persons:,}"
    )

    print(
        f"Vehículos:         {total_vehicles:,}"
    )

    print(
        f"Objetos totales:   {total_objects:,}"
    )

    if total_images:
        print(
            f"Objetos/imagen:    "
            f"{total_objects / total_images:.2f}"
        )

    print("\n")
    print("=" * 70)
    print("PERSONAS")
    print("=" * 70)

    for threshold in SMALL_THRESHOLDS:

        count = sum(
            1
            for area in all_person_areas
            if area < threshold
        )

        percentage = (
            count / len(all_person_areas) * 100
            if all_person_areas
            else 0
        )

        print(
            f"< {threshold:5} px²: "
            f"{count:10,} "
            f"({percentage:6.2f} %)"
        )

    print("\n")
    print("=" * 70)
    print("VEHÍCULOS")
    print("=" * 70)

    for threshold in SMALL_THRESHOLDS:

        count = sum(
            1
            for area in all_vehicle_areas
            if area < threshold
        )

        percentage = (
            count / len(all_vehicle_areas) * 100
            if all_vehicle_areas
            else 0
        )

        print(
            f"< {threshold:5} px²: "
            f"{count:10,} "
            f"({percentage:6.2f} %)"
        )

    print("\n")
    print("=" * 70)
    print("ESCENAS DENSAS")
    print("=" * 70)

    for threshold in thresholds:

        count = sum(
            1
            for x in all_objects_per_image
            if x >= threshold
        )

        percentage = (
            count / len(all_objects_per_image) * 100
            if all_objects_per_image
            else 0
        )

        print(
            f">= {threshold:3} objetos: "
            f"{count:6,} imágenes "
            f"({percentage:6.2f} %)"
        )

    print("\n")
    print("=" * 70)
    print("TOP 10 IMÁGENES MÁS DENSAS")
    print("=" * 70)

    for row in image_rows[:10]:

        print(
            f"{row['total_objects']:4} objetos | "
            f"P={row['persons']:3} "
            f"V={row['vehicles']:3} | "
            f"{row['image']}"
        )

    print("\n")
    print("=" * 70)
    print("ANÁLISIS FINALIZADO")
    print("=" * 70)

    print(
        f"\nResultados guardados en:\n"
        f"{OUTPUT_ROOT / 'reports'}"
    )


if __name__ == "__main__":
    main()