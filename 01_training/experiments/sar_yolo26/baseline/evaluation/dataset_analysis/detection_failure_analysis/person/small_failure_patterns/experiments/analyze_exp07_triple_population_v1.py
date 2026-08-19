from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================================
# SAR YOLO26
# EXP07 - TRIPLE TARGET POPULATION ANALYSIS V1
# ============================================================================
#
# Analiza los objetos TRAIN que cumplen:
#
#   EXTREME_SMALL
#   + DENSE_SCENE
#   + CLOSE_NEIGHBORS
#
# NO GENERA CROPS.
# NO MODIFICA DATASET.
# NO MODIFICA LABELS.
# NO MODIFICA YAML.
#
# Objetivo:
#   conocer la estructura real de los 13.849 objetos TRIPLE antes de diseñar
#   el muestreo estratificado de EXP07.
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

TARGET_TYPE = "TRIPLE"

# Tamaño objetivo aproximado de la futura intervención.
TARGET_SAMPLE_MIN = 3500
TARGET_SAMPLE_MAX = 4000

# Estratos de tamaño.
SIZE_BUCKETS = [
    ("<8", 0.0, 8.0),
    ("8-10", 8.0, 10.0),
    ("10-12", 10.0, 12.0),
    ("12-14", 12.0, 14.0),
    ("14-16", 14.0, 16.0),
]

# Densidad basada en número de SMALL PERSON de la imagen.
DENSITY_BUCKETS = [
    ("25-49", 25, 50),
    ("50-99", 50, 100),
    ("100-199", 100, 200),
    (">=200", 200, None),
]


# ============================================================================
# LOCALIZACIÓN DEL PROYECTO
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_baseline() -> Path:
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if parent.name.lower() == "baseline":
            return parent

    raise RuntimeError(
        "No se pudo localizar el directorio baseline."
    )


BASELINE_DIR = find_baseline()


# ============================================================================
# ENTRADA / SALIDA
# ============================================================================

SOURCE_CSV = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp07_train_target_population_analysis_v1"
    / "reports"
    / "exp07_train_target_population_objects_v1.csv"
)

REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp07_triple_population_analysis_v1"
    / "reports"
)

TRIPLE_OBJECTS_CSV = (
    REPORTS_DIR
    / "exp07_triple_population_objects_v1.csv"
)

IMAGE_STATS_CSV = (
    REPORTS_DIR
    / "exp07_triple_population_by_image_v1.csv"
)

SIZE_STATS_CSV = (
    REPORTS_DIR
    / "exp07_triple_population_by_size_v1.csv"
)

DENSITY_STATS_CSV = (
    REPORTS_DIR
    / "exp07_triple_population_by_density_v1.csv"
)

DISTANCE_STATS_CSV = (
    REPORTS_DIR
    / "exp07_triple_population_by_distance_v1.csv"
)

SAMPLING_PLAN_CSV = (
    REPORTS_DIR
    / "exp07_triple_population_stratified_sampling_plan_v1.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP07_TRIPLE_POPULATION_ANALYSIS_V1_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0

    return a / b


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# CARGA
# ============================================================================

def load_source() -> list[dict]:

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            "No se encontró el report TRAIN:\n"
            f"{SOURCE_CSV}"
        )

    with SOURCE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "El CSV de entrada está vacío."
        )

    required_columns = {
        "image",
        "gt_index",
        "area",
        "size_sqrt",
        "person_count",
        "dense_scene",
        "extreme_small",
        "nearest_distance",
        "close_neighbors",
        "target_triple",
        "target_type",
    }

    missing = (
        required_columns
        -
        set(reader.fieldnames or [])
    )

    if missing:
        raise ValueError(
            "Faltan columnas necesarias:\n"
            +
            "\n".join(
                sorted(missing)
            )
        )

    return rows


# ============================================================================
# FILTRAR TRIPLE
# ============================================================================

def filter_triple(
    rows: list[dict],
) -> list[dict]:

    triple = []

    for row in rows:

        if row["target_type"] != TARGET_TYPE:
            continue

        if int(row["target_triple"]) != 1:
            continue

        item = {
            "image":
                row["image"],

            "gt_index":
                int(row["gt_index"]),

            "area":
                float(row["area"]),

            "size_sqrt":
                float(row["size_sqrt"]),

            "person_count":
                int(row["person_count"]),

            "nearest_distance":
                (
                    float(row["nearest_distance"])
                    if row["nearest_distance"]
                    not in ("", None)
                    else math.inf
                ),
        }

        triple.append(item)

    return triple


# ============================================================================
# BUCKETS
# ============================================================================

def size_bucket(
    value: float,
) -> str:

    for name, low, high in SIZE_BUCKETS:

        if (
            value >= low
            and
            value < high
        ):
            return name

    return ">=16"


def density_bucket(
    value: int,
) -> str:

    for name, low, high in DENSITY_BUCKETS:

        if high is None:

            if value >= low:
                return name

        elif (
            value >= low
            and
            value < high
        ):
            return name

    return "<25"


def distance_bucket(
    value: float,
) -> str:

    if not math.isfinite(value):
        return "NO_NEIGHBOR"

    if value < 4:
        return "<4"

    if value < 6:
        return "4-6"

    if value < 8:
        return "6-8"

    if value < 10:
        return "8-10"

    if value < 12:
        return "10-12"

    if value < 16:
        return "12-16"

    return ">=16"


# ============================================================================
# ESTADÍSTICAS POR IMAGEN
# ============================================================================

def build_image_stats(
    triple_rows: list[dict],
) -> list[dict]:

    groups = defaultdict(list)

    for row in triple_rows:
        groups[
            row["image"]
        ].append(row)

    output = []

    for image, rows in groups.items():

        densities = [
            row["person_count"]
            for row in rows
        ]

        distances = [
            row["nearest_distance"]
            for row in rows
            if math.isfinite(
                row["nearest_distance"]
            )
        ]

        sizes = [
            row["size_sqrt"]
            for row in rows
        ]

        output.append(
            {
                "image":
                    image,

                "triple_objects":
                    len(rows),

                "image_person_density":
                    max(densities),

                "min_person_density":
                    min(densities),

                "avg_target_size_sqrt":
                    round(
                        sum(sizes)
                        /
                        len(sizes),
                        4,
                    ),

                "min_target_size_sqrt":
                    round(
                        min(sizes),
                        4,
                    ),

                "max_target_size_sqrt":
                    round(
                        max(sizes),
                        4,
                    ),

                "avg_nearest_distance":
                    (
                        round(
                            sum(distances)
                            /
                            len(distances),
                            4,
                        )
                        if distances
                        else ""
                    ),
            }
        )

    output.sort(
        key=lambda row:
            row["triple_objects"],
        reverse=True,
    )

    return output


# ============================================================================
# ESTADÍSTICA POR TAMAÑO
# ============================================================================

def build_size_stats(
    triple_rows: list[dict],
) -> list[dict]:

    groups = defaultdict(list)

    for row in triple_rows:

        groups[
            size_bucket(
                row["size_sqrt"]
            )
        ].append(row)

    output = []

    for name, _, _ in SIZE_BUCKETS:

        rows = groups.get(
            name,
            [],
        )

        output.append(
            {
                "size_bucket":
                    name,

                "objects":
                    len(rows),

                "percentage":
                    safe_div(
                        len(rows),
                        len(triple_rows),
                    )
                    * 100.0,

                "avg_person_density":
                    (
                        sum(
                            r["person_count"]
                            for r in rows
                        )
                        /
                        len(rows)
                        if rows
                        else 0.0
                    ),

                "avg_nearest_distance":
                    (
                        sum(
                            r["nearest_distance"]
                            for r in rows
                            if math.isfinite(
                                r[
                                    "nearest_distance"
                                ]
                            )
                        )
                        /
                        sum(
                            1
                            for r in rows
                            if math.isfinite(
                                r[
                                    "nearest_distance"
                                ]
                            )
                        )
                        if any(
                            math.isfinite(
                                r[
                                    "nearest_distance"
                                ]
                            )
                            for r in rows
                        )
                        else 0.0
                    ),
            }
        )

    return output


# ============================================================================
# ESTADÍSTICA POR DENSIDAD
# ============================================================================

def build_density_stats(
    triple_rows: list[dict],
) -> list[dict]:

    groups = defaultdict(list)

    for row in triple_rows:

        groups[
            density_bucket(
                row["person_count"]
            )
        ].append(row)

    output = []

    for name, _, _ in DENSITY_BUCKETS:

        rows = groups.get(
            name,
            [],
        )

        output.append(
            {
                "density_bucket":
                    name,

                "objects":
                    len(rows),

                "percentage":
                    safe_div(
                        len(rows),
                        len(triple_rows),
                    )
                    * 100.0,

                "unique_images":
                    len(
                        {
                            r["image"]
                            for r in rows
                        }
                    ),

                "avg_size_sqrt":
                    (
                        sum(
                            r["size_sqrt"]
                            for r in rows
                        )
                        /
                        len(rows)
                        if rows
                        else 0.0
                    ),
            }
        )

    return output


# ============================================================================
# ESTADÍSTICA POR DISTANCIA
# ============================================================================

def build_distance_stats(
    triple_rows: list[dict],
) -> list[dict]:

    groups = defaultdict(list)

    for row in triple_rows:

        groups[
            distance_bucket(
                row["nearest_distance"]
            )
        ].append(row)

    order = [
        "<4",
        "4-6",
        "6-8",
        "8-10",
        "10-12",
        "12-16",
        ">=16",
        "NO_NEIGHBOR",
    ]

    output = []

    for name in order:

        rows = groups.get(
            name,
            [],
        )

        output.append(
            {
                "distance_bucket":
                    name,

                "objects":
                    len(rows),

                "percentage":
                    safe_div(
                        len(rows),
                        len(triple_rows),
                    )
                    * 100.0,

                "unique_images":
                    len(
                        {
                            r["image"]
                            for r in rows
                        }
                    ),
            }
        )

    return output


# ============================================================================
# PLAN DE MUESTREO
# ============================================================================

def build_sampling_plan(
    triple_rows: list[dict],
) -> list[dict]:

    # Estratificación sencilla:
    #   tamaño + densidad
    #
    # El objetivo no es seleccionar todavía los GT.
    # Solo estimar cuántas muestras corresponderían a cada estrato.

    strata = defaultdict(list)

    for row in triple_rows:

        key = (
            size_bucket(
                row["size_sqrt"]
            ),
            density_bucket(
                row["person_count"]
            ),
        )

        strata[key].append(row)

    total = len(triple_rows)

    if total == 0:
        return []

    target_sample = min(
        TARGET_SAMPLE_MAX,
        max(
            TARGET_SAMPLE_MIN,
            round(total * 0.25),
        ),
    )

    # Mantener mínimo 1 por estrato existente.
    existing_strata = {
        key: len(rows)
        for key, rows in strata.items()
    }

    allocated = {}

    for key, count in existing_strata.items():

        share = (
            count
            /
            total
        )

        proposed = round(
            target_sample
            *
            share
        )

        proposed = max(
            1,
            proposed
        )

        proposed = min(
            count,
            proposed
        )

        allocated[key] = proposed

    current_total = sum(
        allocated.values()
    )

    # Ajuste sencillo para acercarnos al target.
    sorted_strata = sorted(
        existing_strata.items(),
        key=lambda kv:
            kv[1],
        reverse=True,
    )

    while (
        current_total
        <
        target_sample
    ):

        changed = False

        for key, count in sorted_strata:

            if allocated[key] < count:

                allocated[key] += 1
                current_total += 1
                changed = True

                if current_total >= target_sample:
                    break

        if not changed:
            break

    output = []

    for key, count in sorted(
        existing_strata.items(),
    ):

        size_name, density_name = key

        selected = allocated[key]

        output.append(
            {
                "size_bucket":
                    size_name,

                "density_bucket":
                    density_name,

                "available_objects":
                    count,

                "available_percentage":
                    safe_div(
                        count,
                        total,
                    )
                    * 100.0,

                "proposed_samples":
                    selected,

                "sampling_rate":
                    safe_div(
                        selected,
                        count,
                    )
                    * 100.0,
            }
        )

    return output


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    triple_rows: list[dict],
    image_rows: list[dict],
    size_rows: list[dict],
    density_rows: list[dict],
    distance_rows: list[dict],
    sampling_rows: list[dict],
) -> None:

    unique_images = len(
        {
            row["image"]
            for row in triple_rows
        }
    )

    object_counts = [
        row["triple_objects"]
        for row in image_rows
    ]

    max_per_image = (
        max(object_counts)
        if object_counts
        else 0
    )

    avg_per_image = (
        sum(object_counts)
        /
        len(object_counts)
        if object_counts
        else 0.0
    )

    min_per_image = (
        min(object_counts)
        if object_counts
        else 0
    )

    total_proposed = sum(
        row[
            "proposed_samples"
        ]
        for row in sampling_rows
    )

    lines = [
        "=" * 72,
        "SAR YOLO26 - EXP07 TRIPLE POPULATION ANALYSIS V1",
        "=" * 72,
        "",
        "POBLACIÓN ANALIZADA",
        f"TRIPLE objects:       {len(triple_rows):,}",
        f"Unique images:        {unique_images:,}",
        "",
        "OBJETOS TRIPLE POR IMAGEN",
        f"Minimum:              {min_per_image}",
        f"Average:              {avg_per_image:.2f}",
        f"Maximum:              {max_per_image}",
        "",
        "MUESTREO PROPUESTO",
        (
            f"Target aproximado: "
            f"{TARGET_SAMPLE_MIN:,}-{TARGET_SAMPLE_MAX:,}"
        ),
        f"Total propuesto:     {total_proposed:,}",
        "",
        "LECTURA",
        (
            "El muestreo se basa en estratos tamaño + densidad."
        ),
        (
            "No se generan crops en este paso."
        ),
        (
            "El objetivo es evitar la sobreexposición masiva "
            "observada en EXP05."
        ),
        "",
        "DISTRIBUCIÓN POR TAMAÑO",
        "-" * 72,
    ]

    for row in size_rows:
        lines.append(
            f"{row['size_bucket']:<8} "
            f"{row['objects']:>7,} "
            f"{row['percentage']:>7.2f}%"
        )

    lines.extend(
        [
            "",
            "DISTRIBUCIÓN POR DENSIDAD",
            "-" * 72,
        ]
    )

    for row in density_rows:
        lines.append(
            f"{row['density_bucket']:<8} "
            f"{row['objects']:>7,} "
            f"{row['percentage']:>7.2f}% "
            f"images={row['unique_images']:>5,}"
        )

    lines.extend(
        [
            "",
            "DISTRIBUCIÓN POR PROXIMIDAD",
            "-" * 72,
        ]
    )

    for row in distance_rows:
        lines.append(
            f"{row['distance_bucket']:<10} "
            f"{row['objects']:>7,} "
            f"{row['percentage']:>7.2f}%"
        )

    lines.extend(
        [
            "",
            "RECOMENDACIÓN",
            "-" * 72,
            (
                "No generar 13.849 crops automáticamente."
            ),
            (
                "Usar el muestreo estratificado propuesto "
                "como límite inicial."
            ),
            (
                "Mantener un máximo de un crop por GT."
            ),
            (
                "Después de generar el conjunto experimental, "
                "verificar su tamaño antes de entrenar."
            ),
            "",
            "IMPORTANTE: dataset original NO modificado.",
        ]
    )

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - EXP07 TRIPLE POPULATION ANALYSIS V1"
    )
    print("=" * 72)
    print()

    print(
        "SOURCE:"
    )

    print(
        f"  {SOURCE_CSV}"
    )

    rows = load_source()

    triple_rows = filter_triple(
        rows
    )

    if not triple_rows:
        raise RuntimeError(
            "No se encontraron objetos TRIPLE."
        )

    print()
    print(
        f"[OK] SMALL rows: "
        f"{len(rows):,}"
    )

    print(
        f"[OK] TRIPLE rows: "
        f"{len(triple_rows):,}"
    )

    image_rows = build_image_stats(
        triple_rows
    )

    size_rows = build_size_stats(
        triple_rows
    )

    density_rows = build_density_stats(
        triple_rows
    )

    distance_rows = build_distance_stats(
        triple_rows
    )

    sampling_rows = build_sampling_plan(
        triple_rows
    )

    write_csv(
        TRIPLE_OBJECTS_CSV,
        triple_rows
    )

    write_csv(
        IMAGE_STATS_CSV,
        image_rows
    )

    write_csv(
        SIZE_STATS_CSV,
        size_rows
    )

    write_csv(
        DENSITY_STATS_CSV,
        density_rows
    )

    write_csv(
        DISTANCE_STATS_CSV,
        distance_rows
    )

    write_csv(
        SAMPLING_PLAN_CSV,
        sampling_rows
    )

    build_summary(
        triple_rows,
        image_rows,
        size_rows,
        density_rows,
        distance_rows,
        sampling_rows,
    )

    unique_images = len(
        {
            row["image"]
            for row in triple_rows
        }
    )

    total_proposed = sum(
        row["proposed_samples"]
        for row in sampling_rows
    )

    print()
    print("=" * 72)
    print(
        "# RESULTADO EXP07 TRIPLE POPULATION"
    )
    print("=" * 72)
    print()

    print(
        f"TRIPLE OBJECTS:       "
        f"{len(triple_rows):,}"
    )

    print(
        f"UNIQUE IMAGES:        "
        f"{unique_images:,}"
    )

    print(
        f"PROPOSED SAMPLES:     "
        f"{total_proposed:,}"
    )

    print()
    print(
        "REPORTS:"
    )

    print(
        f"[OK] {TRIPLE_OBJECTS_CSV}"
    )

    print(
        f"[OK] {IMAGE_STATS_CSV}"
    )

    print(
        f"[OK] {SIZE_STATS_CSV}"
    )

    print(
        f"[OK] {DENSITY_STATS_CSV}"
    )

    print(
        f"[OK] {DISTANCE_STATS_CSV}"
    )

    print(
        f"[OK] {SAMPLING_PLAN_CSV}"
    )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    print()
    print(
        "IMPORTANTE: no se generaron crops."
    )

    print(
        "IMPORTANTE: dataset original NO modificado."
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\n[CANCELADO]"
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "[ERROR EXP07 TRIPLE ANALYSIS]"
        )
        print()
        print(
            str(exc)
        )

        sys.exit(1)