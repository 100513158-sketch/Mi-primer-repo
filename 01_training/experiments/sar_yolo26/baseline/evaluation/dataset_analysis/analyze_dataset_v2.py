from pathlib import Path
from collections import Counter
import csv
import math

# ============================================================
# SAR YOLO26 - DATASET ANALYSIS V2
# ============================================================
#
# Objetivos:
#   1. Analizar train / val / test_dev
#   2. Convertir bbox YOLO normalizado -> píxeles
#   3. Analizar tamaño real de personas y vehículos
#   4. Analizar densidad de objetos por imagen
#   5. Detectar objetos extremadamente pequeños
#   6. Generar CSV
#   7. Generar gráficos PNG
#
# NO modifica el dataset original.
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
    r"baseline\evaluation\dataset_analysis_v2"
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

# Umbrales de área real en píxeles cuadrados.
AREA_THRESHOLDS = [
    16,
    32,
    64,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
]

# Umbrales de ancho/alto del objeto.
SIZE_THRESHOLDS = [
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
]

# Umbrales para escenas densas.
CROWD_THRESHOLDS = [
    10,
    25,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    500,
]

# Número de imágenes que se muestran en consola.
TOP_N = 20


# ============================================================
# IMPORTACIÓN OPCIONAL DE MATPLOTLIB
# ============================================================

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True

except ImportError:
    MATPLOTLIB_AVAILABLE = False

    print(
        "[WARN] matplotlib no está instalado."
    )

    print(
        "Los CSV se generarán igualmente."
    )

    print(
        "Para gráficos:"
    )

    print(
        "pip install matplotlib"
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def percentile(values, p):
    """
    Calcula percentil sin depender de numpy.
    """

    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return (
        values[f] * (c - k)
        + values[c] * (k - f)
    )


def safe_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def safe_min(values):
    if not values:
        return 0.0

    return min(values)


def safe_max(values):
    if not values:
        return 0.0

    return max(values)


def write_csv(path, rows, fieldnames):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not fieldnames:
        return

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
# OBTENER IMÁGENES
# ============================================================

def find_images(images_dir):

    image_files = []

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]

    for ext in extensions:
        image_files.extend(
            images_dir.rglob(ext)
        )

    return sorted(image_files)


# ============================================================
# ANALIZAR UNA IMAGEN
# ============================================================

def analyze_image(
    image_path,
    labels_dir
):

    try:

        from PIL import Image

        with Image.open(image_path) as img:

            image_width, image_height = img.size

    except Exception as exc:

        print(
            f"[ERROR] No se pudo abrir "
            f"{image_path}: {exc}"
        )

        return None

    label_path = (
        labels_dir
        / f"{image_path.stem}.txt"
    )

    persons = 0
    vehicles = 0
    total_objects = 0

    person_areas = []
    vehicle_areas = []

    person_widths = []
    vehicle_widths = []

    person_heights = []
    vehicle_heights = []

    person_area_ratios = []
    vehicle_area_ratios = []

    object_rows = []

    if label_path.exists():

        try:

            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line_number, line in enumerate(
                lines,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:
                    continue

                try:

                    class_id = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])

                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                except ValueError:

                    print(
                        f"[WARN] Label inválido: "
                        f"{label_path} "
                        f"línea {line_number}"
                    )

                    continue

                if class_id not in CLASS_NAMES:
                    continue

                # ==================================================
                # CONVERSIÓN YOLO NORMALIZADO -> PIXELES
                # ==================================================

                bbox_width_px = (
                    width_norm
                    * image_width
                )

                bbox_height_px = (
                    height_norm
                    * image_height
                )

                bbox_area_px = (
                    bbox_width_px
                    * bbox_height_px
                )

                image_area_px = (
                    image_width
                    * image_height
                )

                area_ratio = (
                    bbox_area_px
                    / image_area_px
                    if image_area_px > 0
                    else 0
                )

                # Coordenadas aproximadas del bbox.

                x_center_px = (
                    x_center
                    * image_width
                )

                y_center_px = (
                    y_center
                    * image_height
                )

                x1 = (
                    x_center_px
                    - bbox_width_px / 2
                )

                y1 = (
                    y_center_px
                    - bbox_height_px / 2
                )

                x2 = (
                    x_center_px
                    + bbox_width_px / 2
                )

                y2 = (
                    y_center_px
                    + bbox_height_px / 2
                )

                class_name = CLASS_NAMES[
                    class_id
                ]

                total_objects += 1

                if class_id == 0:

                    persons += 1

                    person_areas.append(
                        bbox_area_px
                    )

                    person_widths.append(
                        bbox_width_px
                    )

                    person_heights.append(
                        bbox_height_px
                    )

                    person_area_ratios.append(
                        area_ratio
                    )

                elif class_id == 1:

                    vehicles += 1

                    vehicle_areas.append(
                        bbox_area_px
                    )

                    vehicle_widths.append(
                        bbox_width_px
                    )

                    vehicle_heights.append(
                        bbox_height_px
                    )

                    vehicle_area_ratios.append(
                        area_ratio
                    )

                object_rows.append({

                    "image": str(image_path),

                    "class_id": class_id,

                    "class": class_name,

                    "image_width": image_width,

                    "image_height": image_height,

                    "x_center_norm": x_center,

                    "y_center_norm": y_center,

                    "width_norm": width_norm,

                    "height_norm": height_norm,

                    "x1_px": x1,

                    "y1_px": y1,

                    "x2_px": x2,

                    "y2_px": y2,

                    "width_px": bbox_width_px,

                    "height_px": bbox_height_px,

                    "area_px2": bbox_area_px,

                    "area_ratio": area_ratio,

                })

        except Exception as exc:

            print(
                f"[ERROR] {label_path}: {exc}"
            )

    return {

        "image": str(image_path),

        "image_width": image_width,

        "image_height": image_height,

        "image_area": (
            image_width
            * image_height
        ),

        "persons": persons,

        "vehicles": vehicles,

        "total_objects": total_objects,

        "person_areas": person_areas,

        "vehicle_areas": vehicle_areas,

        "person_widths": person_widths,

        "vehicle_widths": vehicle_widths,

        "person_heights": person_heights,

        "vehicle_heights": vehicle_heights,

        "person_area_ratios": person_area_ratios,

        "vehicle_area_ratios": vehicle_area_ratios,

        "object_rows": object_rows,

    }


# ============================================================
# ANALIZAR SPLIT
# ============================================================

def analyze_split(split_dir):

    images_dir = (
        split_dir
        / "images"
    )

    labels_dir = (
        split_dir
        / "labels"
    )

    if not images_dir.exists():

        print(
            f"[WARN] No existe: "
            f"{images_dir}"
        )

        return None

    if not labels_dir.exists():

        print(
            f"[WARN] No existe: "
            f"{labels_dir}"
        )

        return None

    image_files = find_images(
        images_dir
    )

    stats = {

        "split": split_dir.name,

        "images": 0,

        "images_with_person": 0,

        "images_with_vehicle": 0,

        "images_with_both": 0,

        "images_without_objects": 0,

        "persons": 0,

        "vehicles": 0,

        "total_objects": 0,

        "objects_per_image": [],

        "person_areas": [],

        "vehicle_areas": [],

        "person_widths": [],

        "vehicle_widths": [],

        "person_heights": [],

        "vehicle_heights": [],

        "person_area_ratios": [],

        "vehicle_area_ratios": [],

        "image_rows": [],

        "object_rows": [],

    }

    print(
        f"Imágenes encontradas: "
        f"{len(image_files):,}"
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        result = analyze_image(
            image_path,
            labels_dir
        )

        if result is None:
            continue

        stats["images"] += 1

        persons = result["persons"]
        vehicles = result["vehicles"]
        total_objects = result[
            "total_objects"
        ]

        stats["persons"] += persons

        stats["vehicles"] += vehicles

        stats["total_objects"] += (
            total_objects
        )

        stats[
            "objects_per_image"
        ].append(
            total_objects
        )

        stats[
            "person_areas"
        ].extend(
            result["person_areas"]
        )

        stats[
            "vehicle_areas"
        ].extend(
            result["vehicle_areas"]
        )

        stats[
            "person_widths"
        ].extend(
            result["person_widths"]
        )

        stats[
            "vehicle_widths"
        ].extend(
            result["vehicle_widths"]
        )

        stats[
            "person_heights"
        ].extend(
            result["person_heights"]
        )

        stats[
            "vehicle_heights"
        ].extend(
            result["vehicle_heights"]
        )

        stats[
            "person_area_ratios"
        ].extend(
            result[
                "person_area_ratios"
            ]
        )

        stats[
            "vehicle_area_ratios"
        ].extend(
            result[
                "vehicle_area_ratios"
            ]
        )

        stats[
            "object_rows"
        ].extend(
            result["object_rows"]
        )

        if persons > 0:

            stats[
                "images_with_person"
            ] += 1

        if vehicles > 0:

            stats[
                "images_with_vehicle"
            ] += 1

        if persons > 0 and vehicles > 0:

            stats[
                "images_with_both"
            ] += 1

        if total_objects == 0:

            stats[
                "images_without_objects"
            ] += 1

        stats[
            "image_rows"
        ].append({

            "split": split_dir.name,

            "image": str(image_path),

            "image_width":
                result[
                    "image_width"
                ],

            "image_height":
                result[
                    "image_height"
                ],

            "persons":
                persons,

            "vehicles":
                vehicles,

            "total_objects":
                total_objects,

            "person_min_area_px2":
                safe_min(
                    result[
                        "person_areas"
                    ]
                ),

            "person_max_area_px2":
                safe_max(
                    result[
                        "person_areas"
                    ]
                ),

            "vehicle_min_area_px2":
                safe_min(
                    result[
                        "vehicle_areas"
                    ]
                ),

            "vehicle_max_area_px2":
                safe_max(
                    result[
                        "vehicle_areas"
                    ]
                ),

        })

        if index % 1000 == 0:

            print(
                f"  Procesadas: "
                f"{index:,}/"
                f"{len(image_files):,}"
            )

    return stats


# ============================================================
# ESTADÍSTICAS DE ÁREAS
# ============================================================

def build_area_statistics(
    class_name,
    areas
):

    row = {

        "class": class_name,

        "objects": len(areas),

        "mean_area_px2":
            safe_mean(areas),

        "median_area_px2":
            percentile(areas, 50),

        "p25_area_px2":
            percentile(areas, 25),

        "p75_area_px2":
            percentile(areas, 75),

        "p90_area_px2":
            percentile(areas, 90),

        "p95_area_px2":
            percentile(areas, 95),

        "p99_area_px2":
            percentile(areas, 99),

        "min_area_px2":
            safe_min(areas),

        "max_area_px2":
            safe_max(areas),

    }

    for threshold in AREA_THRESHOLDS:

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
            f"below_{threshold}_px2"
        ] = count

        row[
            f"below_{threshold}_pct"
        ] = percentage

    return row


# ============================================================
# ESTADÍSTICAS DE DIMENSIONES
# ============================================================

def build_dimension_statistics(
    class_name,
    widths,
    heights
):

    return {

        "class": class_name,

        "objects": len(widths),

        "mean_width_px":
            safe_mean(widths),

        "median_width_px":
            percentile(widths, 50),

        "p25_width_px":
            percentile(widths, 25),

        "p75_width_px":
            percentile(widths, 75),

        "p90_width_px":
            percentile(widths, 90),

        "mean_height_px":
            safe_mean(heights),

        "median_height_px":
            percentile(heights, 50),

        "p25_height_px":
            percentile(heights, 25),

        "p75_height_px":
            percentile(heights, 75),

        "p90_height_px":
            percentile(heights, 90),

        "min_width_px":
            safe_min(widths),

        "max_width_px":
            safe_max(widths),

        "min_height_px":
            safe_min(heights),

        "max_height_px":
            safe_max(heights),

    }


# ============================================================
# ESTADÍSTICAS DE OBJETOS PEQUEÑOS
# ============================================================

def build_small_object_statistics(
    class_name,
    areas
):

    rows = []

    total = len(areas)

    for threshold in AREA_THRESHOLDS:

        count = sum(
            1
            for area in areas
            if area < threshold
        )

        percentage = (

            count / total * 100

            if total

            else 0
        )

        rows.append({

            "class": class_name,

            "threshold_px2":
                threshold,

            "objects":
                count,

            "percentage":
                percentage,

        })

    return rows


# ============================================================
# ESTADÍSTICAS DE CONGESTIÓN
# ============================================================

def build_crowded_statistics(
    split,
    objects_per_image
):

    rows = []

    total_images = len(
        objects_per_image
    )

    for threshold in CROWD_THRESHOLDS:

        count = sum(
            1
            for x in objects_per_image
            if x >= threshold
        )

        percentage = (

            count
            / total_images
            * 100

            if total_images

            else 0
        )

        rows.append({

            "split": split,

            "threshold_objects":
                threshold,

            "images":
                count,

            "percentage":
                percentage,

        })

    return rows


# ============================================================
# GRÁFICOS
# ============================================================

def generate_plots(
    output_dir,
    all_stats
):

    if not MATPLOTLIB_AVAILABLE:
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. OBJETOS POR IMAGEN
    # ========================================================

    for stats in all_stats:

        values = stats[
            "objects_per_image"
        ]

        if not values:
            continue

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            values,
            bins=50
        )

        plt.xlabel(
            "Objetos por imagen"
        )

        plt.ylabel(
            "Número de imágenes"
        )

        plt.title(
            f"Densidad de objetos - "
            f"{stats['split']}"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{stats['split']}_objects_per_image.png",
            dpi=150
        )

        plt.close()

    # ========================================================
    # 2. ÁREA DE PERSONAS
    # ========================================================

    for stats in all_stats:

        areas = stats[
            "person_areas"
        ]

        if not areas:
            continue

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            areas,
            bins=100
        )

        plt.xlabel(
            "Área bbox (px²)"
        )

        plt.ylabel(
            "Número de personas"
        )

        plt.title(
            f"Tamaño de personas - "
            f"{stats['split']}"
        )

        plt.xscale(
            "log"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{stats['split']}_person_area.png",
            dpi=150
        )

        plt.close()

    # ========================================================
    # 3. ÁREA DE VEHÍCULOS
    # ========================================================

    for stats in all_stats:

        areas = stats[
            "vehicle_areas"
        ]

        if not areas:
            continue

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            areas,
            bins=100
        )

        plt.xlabel(
            "Área bbox (px²)"
        )

        plt.ylabel(
            "Número de vehículos"
        )

        plt.title(
            f"Tamaño de vehículos - "
            f"{stats['split']}"
        )

        plt.xscale(
            "log"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{stats['split']}_vehicle_area.png",
            dpi=150
        )

        plt.close()

    # ========================================================
    # 4. COMPARACIÓN PERSONA / VEHÍCULO
    # ========================================================

    for stats in all_stats:

        person_areas = stats[
            "person_areas"
        ]

        vehicle_areas = stats[
            "vehicle_areas"
        ]

        if not person_areas and not vehicle_areas:
            continue

        plt.figure(
            figsize=(10, 6)
        )

        data = []

        labels = []

        if person_areas:

            data.append(
                person_areas
            )

            labels.append(
                "Person"
            )

        if vehicle_areas:

            data.append(
                vehicle_areas
            )

            labels.append(
                "Vehicle"
            )

        plt.boxplot(
            data,
            tick_labels=labels,
            showfliers=False
        )

        plt.yscale(
            "log"
        )

        plt.ylabel(
            "Área bbox (px²)"
        )

        plt.title(
            f"Comparación de tamaños - "
            f"{stats['split']}"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{stats['split']}_class_area_comparison.png",
            dpi=150
        )

        plt.close()

    # ========================================================
    # 5. RESUMEN DE OBJETOS POR CLASE
    # ========================================================

    split_names = []
    person_counts = []
    vehicle_counts = []

    for stats in all_stats:

        split_names.append(
            stats["split"]
        )

        person_counts.append(
            stats["persons"]
        )

        vehicle_counts.append(
            stats["vehicles"]
        )

    if split_names:

        x = range(
            len(split_names)
        )

        plt.figure(
            figsize=(10, 6)
        )

        width = 0.35

        plt.bar(
            [
                i - width / 2
                for i in x
            ],
            person_counts,
            width=width,
            label="Person"
        )

        plt.bar(
            [
                i + width / 2
                for i in x
            ],
            vehicle_counts,
            width=width,
            label="Vehicle"
        )

        plt.xticks(
            list(x),
            split_names
        )

        plt.ylabel(
            "Número de objetos"
        )

        plt.title(
            "Distribución de clases por split"
        )

        plt.legend()

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / "class_distribution_by_split.png",
            dpi=150
        )

        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "SAR YOLO26 - DATASET ANALYSIS V2"
    )
    print("=" * 80)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Output:")
    print(OUTPUT_ROOT)

    if not DATASET_ROOT.exists():

        print()
        print(
            "[ERROR] Dataset no encontrado:"
        )

        print(
            DATASET_ROOT
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    reports_dir = (
        OUTPUT_ROOT
        / "reports"
    )

    plots_dir = (
        OUTPUT_ROOT
        / "plots"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_stats = []

    # ========================================================
    # ANALIZAR SPLITS
    # ========================================================

    for split in SPLITS:

        split_dir = (
            DATASET_ROOT
            / split
        )

        print()
        print("-" * 80)

        if not split_dir.exists():

            print(
                f"[INFO] Split no encontrado: "
                f"{split}"
            )

            continue

        print(
            f"Analizando: {split}"
        )

        stats = analyze_split(
            split_dir
        )

        if stats is not None:

            all_stats.append(
                stats
            )

    # ========================================================
    # VARIABLES GLOBALES
    # ========================================================

    total_images = 0
    total_persons = 0
    total_vehicles = 0
    total_objects = 0

    all_person_areas = []
    all_vehicle_areas = []

    all_person_widths = []
    all_vehicle_widths = []

    all_person_heights = []
    all_vehicle_heights = []

    all_object_rows = []
    all_image_rows = []

    # ========================================================
    # RESUMEN POR SPLIT
    # ========================================================

    summary_rows = []

    crowded_rows = []

    for stats in all_stats:

        images = stats["images"]

        persons = stats["persons"]

        vehicles = stats["vehicles"]

        objects = stats[
            "total_objects"
        ]

        total_images += images

        total_persons += persons

        total_vehicles += vehicles

        total_objects += objects

        all_person_areas.extend(
            stats["person_areas"]
        )

        all_vehicle_areas.extend(
            stats["vehicle_areas"]
        )

        all_person_widths.extend(
            stats["person_widths"]
        )

        all_vehicle_widths.extend(
            stats["vehicle_widths"]
        )

        all_person_heights.extend(
            stats["person_heights"]
        )

        all_vehicle_heights.extend(
            stats["vehicle_heights"]
        )

        all_object_rows.extend(
            stats["object_rows"]
        )

        all_image_rows.extend(
            stats["image_rows"]
        )

        mean_objects = (

            safe_mean(
                stats[
                    "objects_per_image"
                ]
            )

        )

        summary_rows.append({

            "split":
                stats["split"],

            "images":
                images,

            "persons":
                persons,

            "vehicles":
                vehicles,

            "total_objects":
                objects,

            "mean_objects_per_image":
                mean_objects,

            "median_objects_per_image":
                percentile(
                    stats[
                        "objects_per_image"
                    ],
                    50
                ),

            "p90_objects_per_image":
                percentile(
                    stats[
                        "objects_per_image"
                    ],
                    90
                ),

            "max_objects_per_image":
                safe_max(
                    stats[
                        "objects_per_image"
                    ]
                ),

            "images_with_person":
                stats[
                    "images_with_person"
                ],

            "images_with_vehicle":
                stats[
                    "images_with_vehicle"
                ],

            "images_with_both":
                stats[
                    "images_with_both"
                ],

            "images_without_objects":
                stats[
                    "images_without_objects"
                ],

        })

        crowded_rows.extend(
            build_crowded_statistics(
                stats["split"],
                stats[
                    "objects_per_image"
                ]
            )
        )

    # ========================================================
    # ESTADÍSTICAS GLOBALES DE ÁREA
    # ========================================================

    area_rows = [

        build_area_statistics(
            "person",
            all_person_areas
        ),

        build_area_statistics(
            "vehicle",
            all_vehicle_areas
        ),

    ]

    # ========================================================
    # ESTADÍSTICAS DE DIMENSIONES
    # ========================================================

    dimension_rows = [

        build_dimension_statistics(
            "person",
            all_person_widths,
            all_person_heights
        ),

        build_dimension_statistics(
            "vehicle",
            all_vehicle_widths,
            all_vehicle_heights
        ),

    ]

    # ========================================================
    # OBJETOS PEQUEÑOS
    # ========================================================

    small_object_rows = []

    small_object_rows.extend(
        build_small_object_statistics(
            "person",
            all_person_areas
        )
    )

    small_object_rows.extend(
        build_small_object_statistics(
            "vehicle",
            all_vehicle_areas
        )
    )

    # ========================================================
    # TOP IMÁGENES MÁS DENSAS
    # ========================================================

    all_image_rows.sort(
        key=lambda x: x[
            "total_objects"
        ],
        reverse=True
    )

    # ========================================================
    # TOP OBJETOS MÁS PEQUEÑOS
    # ========================================================

    smallest_objects = sorted(
        all_object_rows,
        key=lambda x: x[
            "area_px2"
        ]
    )

    # ========================================================
    # CSV
    # ========================================================

    if summary_rows:

        write_csv(

            reports_dir
            / "dataset_summary.csv",

            summary_rows,

            summary_rows[0].keys()

        )

    if area_rows:

        write_csv(

            reports_dir
            / "bbox_area_statistics.csv",

            area_rows,

            area_rows[0].keys()

        )

    if dimension_rows:

        write_csv(

            reports_dir
            / "bbox_dimension_statistics.csv",

            dimension_rows,

            dimension_rows[0].keys()

        )

    if small_object_rows:

        write_csv(

            reports_dir
            / "small_objects_statistics.csv",

            small_object_rows,

            small_object_rows[0].keys()

        )

    if crowded_rows:

        write_csv(

            reports_dir
            / "crowded_scenes.csv",

            crowded_rows,

            crowded_rows[0].keys()

        )

    if all_image_rows:

        write_csv(

            reports_dir
            / "image_statistics.csv",

            all_image_rows,

            all_image_rows[0].keys()

        )

        write_csv(

            reports_dir
            / "top_100_crowded_images.csv",

            all_image_rows[:100],

            all_image_rows[0].keys()

        )

    if all_object_rows:

        write_csv(

            reports_dir
            / "object_statistics.csv",

            all_object_rows,

            all_object_rows[0].keys()

        )

        write_csv(

            reports_dir
            / "top_100_smallest_objects.csv",

            smallest_objects[:100],

            smallest_objects[0].keys()

        )

    # ========================================================
    # GRÁFICOS
    # ========================================================

    print()
    print(
        "Generando gráficos..."
    )

    generate_plots(
        plots_dir,
        all_stats
    )

    # ========================================================
    # RESULTADO GENERAL
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RESULTADO GENERAL"
    )
    print("=" * 80)

    print()
    print(
        f"Imágenes:        "
        f"{total_images:,}"
    )

    print(
        f"Personas:        "
        f"{total_persons:,}"
    )

    print(
        f"Vehículos:       "
        f"{total_vehicles:,}"
    )

    print(
        f"Objetos totales: "
        f"{total_objects:,}"
    )

    if total_images:

        print(
            f"Objetos/imagen:  "
            f"{total_objects / total_images:.2f}"
        )

    # ========================================================
    # PERSONAS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PERSONAS - ÁREA REAL"
    )
    print("=" * 80)

    for threshold in AREA_THRESHOLDS:

        count = sum(

            1
            for area in all_person_areas
            if area < threshold

        )

        percentage = (

            count
            / len(all_person_areas)
            * 100

            if all_person_areas

            else 0

        )

        print(

            f"< {threshold:6} px²: "
            f"{count:10,} "
            f"({percentage:6.2f} %)"

        )

    # ========================================================
    # VEHÍCULOS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VEHÍCULOS - ÁREA REAL"
    )
    print("=" * 80)

    for threshold in AREA_THRESHOLDS:

        count = sum(

            1
            for area in all_vehicle_areas
            if area < threshold

        )

        percentage = (

            count
            / len(all_vehicle_areas)
            * 100

            if all_vehicle_areas

            else 0

        )

        print(

            f"< {threshold:6} px²: "
            f"{count:10,} "
            f"({percentage:6.2f} %)"

        )

    # ========================================================
    # ESCENAS DENSAS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "ESCENAS DENSAS"
    )
    print("=" * 80)

    if total_images:

        for threshold in CROWD_THRESHOLDS:

            count = sum(

                1
                for x in [
                    value
                    for stats in all_stats
                    for value in stats[
                        "objects_per_image"
                    ]
                ]

                if x >= threshold

            )

            percentage = (

                count
                / total_images
                * 100

            )

            print(

                f">= {threshold:4} objetos: "
                f"{count:7,} imágenes "
                f"({percentage:6.2f} %)"

            )

    # ========================================================
    # TOP IMÁGENES DENSAS
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"TOP {TOP_N} IMÁGENES MÁS DENSAS"
    )
    print("=" * 80)

    for row in all_image_rows[:TOP_N]:

        print(

            f"{row['total_objects']:4} objetos | "
            f"P={row['persons']:4} "
            f"V={row['vehicles']:4} | "
            f"{row['image']}"

        )

    # ========================================================
    # TOP OBJETOS MÁS PEQUEÑOS
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"TOP {TOP_N} OBJETOS MÁS PEQUEÑOS"
    )
    print("=" * 80)

    for row in smallest_objects[:TOP_N]:

        print(

            f"{row['area_px2']:10.2f} px² | "
            f"{row['class']:8} | "
            f"W={row['width_px']:.2f}px "
            f"H={row['height_px']:.2f}px | "
            f"{row['image']}"

        )

    # ========================================================
    # ESTADÍSTICAS GLOBALES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "ESTADÍSTICAS GLOBALES DE TAMAÑO"
    )
    print("=" * 80)

    print()

    for row in area_rows:

        print(
            f"{row['class'].upper()}:"
        )

        print(
            f"  Media:   "
            f"{row['mean_area_px2']:.2f} px²"
        )

        print(
            f"  Mediana: "
            f"{row['median_area_px2']:.2f} px²"
        )

        print(
            f"  P25:     "
            f"{row['p25_area_px2']:.2f} px²"
        )

        print(
            f"  P75:     "
            f"{row['p75_area_px2']:.2f} px²"
        )

        print(
            f"  P90:     "
            f"{row['p90_area_px2']:.2f} px²"
        )

        print(
            f"  P95:     "
            f"{row['p95_area_px2']:.2f} px²"
        )

        print(
            f"  P99:     "
            f"{row['p99_area_px2']:.2f} px²"
        )

        print(
            f"  Mínimo:  "
            f"{row['min_area_px2']:.2f} px²"
        )

        print(
            f"  Máximo:  "
            f"{row['max_area_px2']:.2f} px²"
        )

        print()

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("=" * 80)
    print(
        "ANÁLISIS FINALIZADO"
    )
    print("=" * 80)

    print()
    print(
        "Reports:"
    )

    print(
        reports_dir
    )

    print()
    print(
        "Plots:"
    )

    print(
        plots_dir
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()