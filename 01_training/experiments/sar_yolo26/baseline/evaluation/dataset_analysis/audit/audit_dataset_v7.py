from pathlib import Path
from collections import Counter
import csv
import math
import statistics


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO"
    r"\processed\sar\VisDrone_SAR_2CLASS"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"\baseline\evaluation\dataset_analysis\audit\audit_dataset_v7"
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

TINY16 = 16
TINY32 = 32
TINY64 = 64
SMALL100 = 100

# ------------------------------------------------------------
# Umbral para considerar un objeto cerca del borde.
#
# Se expresa como distancia normalizada desde el centro del
# objeto hasta cualquiera de los cuatro bordes.
# ------------------------------------------------------------

BORDER_DISTANCE = 0.05

# ------------------------------------------------------------
# Crowded
# ------------------------------------------------------------

CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500

# ------------------------------------------------------------
# Pesos para el score de riesgo.
#
# NO significa que una imagen sea incorrecta.
# Solo sirve para priorizar revisión.
# ------------------------------------------------------------

WEIGHT_TINY16 = 1.50
WEIGHT_TINY32 = 0.50
WEIGHT_PARTIAL = 2.00
WEIGHT_BORDER = 0.25
WEIGHT_CROWDED100 = 1.00
WEIGHT_CROWDED200 = 2.00
WEIGHT_CROWDED300 = 3.00
WEIGHT_CROWDED500 = 4.00

# ------------------------------------------------------------
# Clasificación final
# ------------------------------------------------------------

REVIEW_SCORE = 8.0
EXCLUDE_SCORE = 20.0
CRITICAL_SCORE = 40.0

# Nunca excluir automáticamente una imagen solamente por
# tener muchos objetos o objetos pequeños.
#
# EXCLUDE_CANDIDATE requiere además una combinación fuerte
# de anomalías.
# ------------------------------------------------------------


# ============================================================
# UTILIDADES
# ============================================================

def percentile(values, p):
    """
    Percentil sin depender de numpy.
    """

    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100.0)

    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(values[int(k)])

    return (
        values[f] * (c - k)
        + values[c] * (k - f)
    )


def safe_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def pct(part, total):
    if total == 0:
        return 0.0

    return part / total * 100.0


def find_images(images_dir):
    extensions = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    )

    image_files = []

    for ext in extensions:
        image_files.extend(
            images_dir.rglob(ext)
        )

    return sorted(set(image_files))


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

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
# ANÁLISIS DE UNA IMAGEN
# ============================================================

def analyze_image(
    image_path,
    labels_dir,
    split
):

    label_path = (
        labels_dir
        / f"{image_path.stem}.txt"
    )

    result = {
        "split": split,
        "image": str(image_path),
        "label": str(label_path),

        "objects": 0,

        "persons": 0,
        "vehicles": 0,

        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "small100": 0,

        "person_tiny16": 0,
        "person_tiny32": 0,
        "person_tiny64": 0,

        "vehicle_tiny16": 0,
        "vehicle_tiny32": 0,
        "vehicle_tiny64": 0,

        "partial_bbox": 0,
        "outside_bbox": 0,
        "border_objects": 0,

        "person_partial": 0,
        "vehicle_partial": 0,

        "person_border": 0,
        "vehicle_border": 0,

        "invalid_labels": 0,
        "invalid_coordinates": 0,
        "invalid_bbox": 0,
        "invalid_class": 0,

        "duplicate_annotations": 0,

        "person_areas": [],
        "vehicle_areas": [],

        "reasons": [],
        "score": 0.0,
        "decision": "KEEP",
    }

    if not label_path.exists():
        result["reasons"].append(
            "missing_label"
        )

        result["decision"] = "REVIEW"

        return result

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception:
        result["invalid_labels"] += 1

        result["reasons"].append(
            "label_read_error"
        )

        result["decision"] = "CRITICAL"

        return result

    seen_annotations = set()

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:

            result["invalid_labels"] += 1

            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except Exception:

            result["invalid_labels"] += 1

            continue

        # ----------------------------------------------------
        # Clase
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            result["invalid_class"] += 1

            continue

        # ----------------------------------------------------
        # Coordenadas
        # ----------------------------------------------------

        if not all(
            math.isfinite(x)
            for x in (
                x_center,
                y_center,
                width,
                height
            )
        ):

            result["invalid_coordinates"] += 1

            continue

        # ----------------------------------------------------
        # Dimensiones
        # ----------------------------------------------------

        if width <= 0 or height <= 0:

            result["invalid_bbox"] += 1

            continue

        # ----------------------------------------------------
        # BBox
        # ----------------------------------------------------

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0

        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0

        # ----------------------------------------------------
        # Área normalizada
        #
        # Se mantiene como proporción de imagen.
        # Para los umbrales de px² se utiliza una estimación
        # posterior basada en resolución.
        # ----------------------------------------------------

        area_norm = width * height

        # ----------------------------------------------------
        # Duplicado exacto
        # ----------------------------------------------------

        annotation_key = (
            class_id,
            round(x_center, 6),
            round(y_center, 6),
            round(width, 6),
            round(height, 6),
        )

        if annotation_key in seen_annotations:

            result["duplicate_annotations"] += 1

        seen_annotations.add(
            annotation_key
        )

        # ----------------------------------------------------
        # Fuera de imagen
        # ----------------------------------------------------

        partial = (
            x1 < 0
            or y1 < 0
            or x2 > 1
            or y2 > 1
        )

        completely_outside = (
            x2 <= 0
            or y2 <= 0
            or x1 >= 1
            or y1 >= 1
        )

        if completely_outside:

            result["outside_bbox"] += 1

        elif partial:

            result["partial_bbox"] += 1

            if class_id == 0:
                result["person_partial"] += 1
            else:
                result["vehicle_partial"] += 1

        # ----------------------------------------------------
        # Cerca del borde
        #
        # Se evalúa el centro del objeto.
        # ----------------------------------------------------

        distance_to_border = min(
            x_center,
            y_center,
            1.0 - x_center,
            1.0 - y_center,
        )

        near_border = (
            distance_to_border
            <= BORDER_DISTANCE
        )

        if near_border:

            result["border_objects"] += 1

            if class_id == 0:
                result["person_border"] += 1
            else:
                result["vehicle_border"] += 1

        # ----------------------------------------------------
        # Guardar área normalizada
        # ----------------------------------------------------

        result["objects"] += 1

        if class_id == 0:

            result["persons"] += 1

            result["person_areas"].append(
                area_norm
            )

        else:

            result["vehicles"] += 1

            result["vehicle_areas"].append(
                area_norm
            )

        # ----------------------------------------------------
        # Tamaños
        #
        # Los umbrales exactos en px² se calculan en una fase
        # posterior usando la resolución de la imagen.
        # ----------------------------------------------------

        # Por ahora clasificamos usando área normalizada
        # como apoyo relativo.

    # ========================================================
    # RESOLUCIÓN DE IMAGEN
    # ========================================================

    try:

        from PIL import Image

        with Image.open(image_path) as img:

            width_px, height_px = img.size

        image_area_px = (
            width_px * height_px
        )

        # ----------------------------------------------------
        # Recorremos nuevamente las anotaciones para obtener
        # tamaños reales en px².
        # ----------------------------------------------------

        tiny_annotations = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(parts[0])

                width = float(parts[3])
                height = float(parts[4])

            except Exception:
                continue

            if class_id not in CLASS_NAMES:
                continue

            area_px = (
                width
                * height
                * image_area_px
            )

            tiny_annotations.append(
                (
                    class_id,
                    area_px
                )
            )

            if area_px < TINY16:

                result["tiny16"] += 1

                if class_id == 0:
                    result["person_tiny16"] += 1
                else:
                    result["vehicle_tiny16"] += 1

            if area_px < TINY32:

                result["tiny32"] += 1

                if class_id == 0:
                    result["person_tiny32"] += 1
                else:
                    result["vehicle_tiny32"] += 1

            if area_px < TINY64:

                result["tiny64"] += 1

                if class_id == 0:
                    result["person_tiny64"] += 1
                else:
                    result["vehicle_tiny64"] += 1

            if area_px < SMALL100:

                result["small100"] += 1

    except Exception:

        pass

    # ========================================================
    # SCORE
    # ========================================================

    score = 0.0

    reasons = []

    if result["tiny16"] > 0:

        score += (
            result["tiny16"]
            * WEIGHT_TINY16
        )

        reasons.append(
            f"tiny16={result['tiny16']}"
        )

    if result["tiny32"] > 0:

        score += (
            result["tiny32"]
            * WEIGHT_TINY32
        )

        reasons.append(
            f"tiny32={result['tiny32']}"
        )

    if result["partial_bbox"] > 0:

        score += (
            result["partial_bbox"]
            * WEIGHT_PARTIAL
        )

        reasons.append(
            "partial_bbox"
        )

    if result["border_objects"] > 0:

        score += (
            result["border_objects"]
            * WEIGHT_BORDER
        )

        reasons.append(
            "border_objects"
        )

    if result["objects"] >= CROWDED_100:

        score += WEIGHT_CROWDED100

        reasons.append(
            "crowded100"
        )

    if result["objects"] >= CROWDED_200:

        score += WEIGHT_CROWDED200

        reasons.append(
            "crowded200"
        )

    if result["objects"] >= CROWDED_300:

        score += WEIGHT_CROWDED300

        reasons.append(
            "crowded300"
        )

    if result["objects"] >= CROWDED_500:

        score += WEIGHT_CROWDED500

        reasons.append(
            "crowded500"
        )

    # --------------------------------------------------------
    # Integridad crítica
    # --------------------------------------------------------

    integrity_errors = (
        result["invalid_labels"]
        + result["invalid_coordinates"]
        + result["invalid_bbox"]
        + result["invalid_class"]
    )

    if integrity_errors > 0:

        reasons.append(
            "integrity_error"
        )

        decision = "CRITICAL"

    # --------------------------------------------------------
    # Exclusión candidata
    #
    # Solo si existe una combinación fuerte de problemas.
    # --------------------------------------------------------

    elif (
        score >= EXCLUDE_SCORE
        and result["objects"] > 0
        and (
            result["tiny16"] >= 20
            or result["partial_bbox"] >= 5
        )
    ):

        decision = "EXCLUDE_CANDIDATE"

    elif score >= REVIEW_SCORE:

        decision = "REVIEW"

    else:

        decision = "KEEP"

    result["score"] = round(
        score,
        2
    )

    result["reasons"] = reasons

    result["decision"] = decision

    return result


# ============================================================
# ANÁLISIS DE SPLIT
# ============================================================

def analyze_split(split):

    split_dir = DATASET_ROOT / split

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():

        print(
            f"[INFO] Split no encontrado: {split}"
        )

        return []

    if not labels_dir.exists():

        print(
            f"[WARN] Labels no encontrados: "
            f"{labels_dir}"
        )

        return []

    image_files = find_images(
        images_dir
    )

    print(
        f"\n## Analizando: {split}"
    )

    print(
        f"Imágenes encontradas: "
        f"{len(image_files):,}"
    )

    results = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        result = analyze_image(
            image_path,
            labels_dir,
            split
        )

        results.append(result)

        if index % 1000 == 0:

            print(
                f"Procesadas: "
                f"{index:,}/"
                f"{len(image_files):,}"
            )

    return results


# ============================================================
# ESTADÍSTICAS POR CLASE
# ============================================================

def generate_class_statistics(
    image_results
):

    class_data = {
        "person": [],
        "vehicle": [],
    }

    for result in image_results:

        class_data["person"].extend(
            result["person_areas"]
        )

        class_data["vehicle"].extend(
            result["vehicle_areas"]
        )

    rows = []

    for class_name, values in class_data.items():

        rows.append({

            "class": class_name,

            "objects": len(values),

            "mean_normalized_area":
                safe_mean(values),

            "median_normalized_area":
                percentile(values, 50),

            "p25":
                percentile(values, 25),

            "p75":
                percentile(values, 75),

            "p90":
                percentile(values, 90),

            "p95":
                percentile(values, 95),

            "p99":
                percentile(values, 99),

            "min":
                min(values)
                if values else 0,

            "max":
                max(values)
                if values else 0,
        })

    return rows


# ============================================================
# ESTADÍSTICAS DE TAMAÑO POR CLASE
# ============================================================

def generate_class_anomaly_statistics(
    image_results
):

    counters = {
        "person": {
            "objects": 0,
            "tiny16": 0,
            "tiny32": 0,
            "tiny64": 0,
            "partial": 0,
            "border": 0,
        },
        "vehicle": {
            "objects": 0,
            "tiny16": 0,
            "tiny32": 0,
            "tiny64": 0,
            "partial": 0,
            "border": 0,
        },
    }

    for r in image_results:

        counters["person"]["objects"] += (
            r["persons"]
        )

        counters["vehicle"]["objects"] += (
            r["vehicles"]
        )

        counters["person"]["tiny16"] += (
            r["person_tiny16"]
        )

        counters["vehicle"]["tiny16"] += (
            r["vehicle_tiny16"]
        )

        counters["person"]["tiny32"] += (
            r["person_tiny32"]
        )

        counters["vehicle"]["tiny32"] += (
            r["vehicle_tiny32"]
        )

        counters["person"]["tiny64"] += (
            r["person_tiny64"]
        )

        counters["vehicle"]["tiny64"] += (
            r["vehicle_tiny64"]
        )

        counters["person"]["partial"] += (
            r["person_partial"]
        )

        counters["vehicle"]["partial"] += (
            r["vehicle_partial"]
        )

        counters["person"]["border"] += (
            r["person_border"]
        )

        counters["vehicle"]["border"] += (
            r["vehicle_border"]
        )

    rows = []

    for class_name in (
        "person",
        "vehicle"
    ):

        c = counters[class_name]

        rows.append({

            "class": class_name,

            "objects": c["objects"],

            "tiny16":
                c["tiny16"],

            "tiny16_pct":
                pct(
                    c["tiny16"],
                    c["objects"]
                ),

            "tiny32":
                c["tiny32"],

            "tiny32_pct":
                pct(
                    c["tiny32"],
                    c["objects"]
                ),

            "tiny64":
                c["tiny64"],

            "tiny64_pct":
                pct(
                    c["tiny64"],
                    c["objects"]
                ),

            "partial_bbox":
                c["partial"],

            "partial_bbox_pct":
                pct(
                    c["partial"],
                    c["objects"]
                ),

            "border_objects":
                c["border"],

            "border_objects_pct":
                pct(
                    c["border"],
                    c["objects"]
                ),
        })

    return rows


# ============================================================
# RESUMEN GENERAL
# ============================================================

def generate_summary(
    image_results
):

    total_images = len(
        image_results
    )

    total_objects = sum(
        r["objects"]
        for r in image_results
    )

    persons = sum(
        r["persons"]
        for r in image_results
    )

    vehicles = sum(
        r["vehicles"]
        for r in image_results
    )

    decisions = Counter(
        r["decision"]
        for r in image_results
    )

    return {

        "images":
            total_images,

        "persons":
            persons,

        "vehicles":
            vehicles,

        "objects":
            total_objects,

        "objects_per_image":
            (
                total_objects
                / total_images
                if total_images
                else 0
            ),

        "keep":
            decisions["KEEP"],

        "review":
            decisions["REVIEW"],

        "exclude_candidate":
            decisions["EXCLUDE_CANDIDATE"],

        "critical":
            decisions["CRITICAL"],

        "tiny16":
            sum(
                r["tiny16"]
                for r in image_results
            ),

        "tiny32":
            sum(
                r["tiny32"]
                for r in image_results
            ),

        "tiny64":
            sum(
                r["tiny64"]
                for r in image_results
            ),

        "partial_bbox":
            sum(
                r["partial_bbox"]
                for r in image_results
            ),

        "outside_bbox":
            sum(
                r["outside_bbox"]
                for r in image_results
            ),

        "border_objects":
            sum(
                r["border_objects"]
                for r in image_results
            ),

        "invalid_labels":
            sum(
                r["invalid_labels"]
                for r in image_results
            ),

        "invalid_coordinates":
            sum(
                r["invalid_coordinates"]
                for r in image_results
            ),

        "invalid_bbox":
            sum(
                r["invalid_bbox"]
                for r in image_results
            ),

        "invalid_class":
            sum(
                r["invalid_class"]
                for r in image_results
            ),

        "duplicates":
            sum(
                r["duplicate_annotations"]
                for r in image_results
            ),
    }


# ============================================================
# INFORME TXT
# ============================================================

def generate_text_report(
    summary,
    class_rows,
    class_anomaly_rows,
    image_results
):

    report_path = (
        OUTPUT_ROOT
        / "reports"
        / "AUDIT_V7_RECOMMENDATION.txt"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total = summary["images"]

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V7\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "RECOMENDACIÓN GENERAL\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        if summary["critical"] == 0:

            f.write(
                "No se detectaron anomalías "
                "críticas de integridad.\n"
            )

        else:

            f.write(
                "Existen anomalías críticas "
                "que deben revisarse.\n"
            )

        f.write(
            "\n"
        )

        f.write(
            f"KEEP: "
            f"{summary['keep']:,} "
            f"({pct(summary['keep'], total):.2f} %)\n"
        )

        f.write(
            f"REVIEW: "
            f"{summary['review']:,} "
            f"({pct(summary['review'], total):.2f} %)\n"
        )

        f.write(
            f"EXCLUDE_CANDIDATE: "
            f"{summary['exclude_candidate']:,} "
            f"({pct(summary['exclude_candidate'], total):.2f} %)\n"
        )

        f.write(
            f"CRITICAL: "
            f"{summary['critical']:,} "
            f"({pct(summary['critical'], total):.2f} %)\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Las categorías REVIEW y "
            "EXCLUDE_CANDIDATE NO implican "
            "que una imagen sea incorrecta.\n"
        )

        f.write(
            "Son prioridades para revisión "
            "antes de cualquier limpieza.\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Los objetos pequeños pueden ser "
            "especialmente importantes para "
            "detección SAR/aérea.\n"
        )

        f.write(
            "No deben eliminarse automáticamente.\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Las BBoxes parcialmente fuera "
            "pueden representar objetos "
            "realmente cortados por el borde "
            "de la imagen y no necesariamente "
            "errores de anotación.\n"
        )

        f.write(
            "\n\n"
        )

        f.write(
            "RESUMEN DATASET\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Imágenes: "
            f"{summary['images']:,}\n"
        )

        f.write(
            f"Personas: "
            f"{summary['persons']:,}\n"
        )

        f.write(
            f"Vehículos: "
            f"{summary['vehicles']:,}\n"
        )

        f.write(
            f"Objetos: "
            f"{summary['objects']:,}\n"
        )

        f.write(
            f"Objetos/imagen: "
            f"{summary['objects_per_image']:.2f}\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "ANOMALÍAS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"<16 px²: "
            f"{summary['tiny16']:,}\n"
        )

        f.write(
            f"<32 px²: "
            f"{summary['tiny32']:,}\n"
        )

        f.write(
            f"<64 px²: "
            f"{summary['tiny64']:,}\n"
        )

        f.write(
            f"BBox parcialmente fuera: "
            f"{summary['partial_bbox']:,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{summary['outside_bbox']:,}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{summary['border_objects']:,}\n"
        )

        f.write(
            f"Labels inválidos: "
            f"{summary['invalid_labels']:,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{summary['invalid_coordinates']:,}\n"
        )

        f.write(
            f"BBoxes inválidas: "
            f"{summary['invalid_bbox']:,}\n"
        )

        f.write(
            f"Clases inválidas: "
            f"{summary['invalid_class']:,}\n"
        )

        f.write(
            f"Duplicados: "
            f"{summary['duplicates']:,}\n"
        )

        f.write(
            "\n\n"
        )

        f.write(
            "ANÁLISIS POR CLASE\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for row in class_anomaly_rows:

            f.write(
                f"\n{row['class'].upper()}\n"
            )

            f.write(
                f"Objetos: "
                f"{row['objects']:,}\n"
            )

            f.write(
                f"Tiny <16: "
                f"{row['tiny16']:,} "
                f"({row['tiny16_pct']:.2f} %)\n"
            )

            f.write(
                f"Tiny <32: "
                f"{row['tiny32']:,} "
                f"({row['tiny32_pct']:.2f} %)\n"
            )

            f.write(
                f"Tiny <64: "
                f"{row['tiny64']:,} "
                f"({row['tiny64_pct']:.2f} %)\n"
            )

            f.write(
                f"Partial BBox: "
                f"{row['partial_bbox']:,} "
                f"({row['partial_bbox_pct']:.2f} %)\n"
            )

            f.write(
                f"Border: "
                f"{row['border_objects']:,} "
                f"({row['border_objects_pct']:.2f} %)\n"
            )

        f.write(
            "\n\n"
        )

        f.write(
            "CONCLUSIÓN\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        if summary["critical"] == 0:

            f.write(
                "El dataset no presenta problemas "
                "estructurales críticos.\n\n"
            )

            f.write(
                "La estrategia recomendada es "
                "NO eliminar automáticamente "
                "objetos pequeños, escenas densas "
                "ni BBoxes parcialmente fuera.\n\n"
            )

            f.write(
                "Las imágenes REVIEW deben ser "
                "inspeccionadas visualmente antes "
                "de construir una versión limpia "
                "del dataset.\n"
            )

        else:

            f.write(
                "Debe realizarse una revisión de "
                "las imágenes CRITICAL antes de "
                "utilizar el dataset para entrenamiento.\n"
            )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n# SAR YOLO26 - DATASET AUDIT V7\n"
    )

    print(
        f"\nDataset:\n{DATASET_ROOT}"
    )

    print(
        f"\nOutput:\n{OUTPUT_ROOT}"
    )

    if not DATASET_ROOT.exists():

        print(
            "\n[ERROR] No existe DATASET_ROOT:"
        )

        print(
            DATASET_ROOT
        )

        return

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # ANALIZAR SPLITS
    # ========================================================

    all_results = []

    for split in SPLITS:

        results = analyze_split(
            split
        )

        all_results.extend(
            results
        )

    if not all_results:

        print(
            "\n[ERROR] No se encontraron "
            "imágenes."
        )

        return

    # ========================================================
    # RESUMEN
    # ========================================================

    summary = generate_summary(
        all_results
    )

    class_rows = (
        generate_class_statistics(
            all_results
        )
    )

    class_anomaly_rows = (
        generate_class_anomaly_statistics(
            all_results
        )
    )

    # ========================================================
    # ORDENAR POR SCORE
    # ========================================================

    review_results = sorted(
        all_results,
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # IMAGE AUDIT
    # ========================================================

    image_rows = []

    for r in all_results:

        image_rows.append({

            "split":
                r["split"],

            "image":
                r["image"],

            "decision":
                r["decision"],

            "score":
                r["score"],

            "objects":
                r["objects"],

            "persons":
                r["persons"],

            "vehicles":
                r["vehicles"],

            "tiny16":
                r["tiny16"],

            "tiny32":
                r["tiny32"],

            "tiny64":
                r["tiny64"],

            "small100":
                r["small100"],

            "person_tiny16":
                r["person_tiny16"],

            "person_tiny32":
                r["person_tiny32"],

            "person_tiny64":
                r["person_tiny64"],

            "vehicle_tiny16":
                r["vehicle_tiny16"],

            "vehicle_tiny32":
                r["vehicle_tiny32"],

            "vehicle_tiny64":
                r["vehicle_tiny64"],

            "partial_bbox":
                r["partial_bbox"],

            "outside_bbox":
                r["outside_bbox"],

            "border_objects":
                r["border_objects"],

            "person_partial":
                r["person_partial"],

            "vehicle_partial":
                r["vehicle_partial"],

            "person_border":
                r["person_border"],

            "vehicle_border":
                r["vehicle_border"],

            "invalid_labels":
                r["invalid_labels"],

            "invalid_coordinates":
                r["invalid_coordinates"],

            "invalid_bbox":
                r["invalid_bbox"],

            "invalid_class":
                r["invalid_class"],

            "duplicates":
                r["duplicate_annotations"],

            "reasons":
                ";".join(
                    r["reasons"]
                ),
        })

    # ========================================================
    # REPORTES
    # ========================================================

    reports_dir = (
        OUTPUT_ROOT / "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    write_csv(
        reports_dir
        / "audit_v7_images.csv",
        image_rows
    )

    write_csv(
        reports_dir
        / "audit_v7_class_statistics.csv",
        class_rows
    )

    write_csv(
        reports_dir
        / "audit_v7_class_anomalies.csv",
        class_anomaly_rows
    )

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    review_rows = [
        row
        for row in image_rows
        if row["decision"]
        in (
            "REVIEW",
            "EXCLUDE_CANDIDATE",
            "CRITICAL",
        )
    ]

    review_rows.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    write_csv(
        reports_dir
        / "audit_v7_review_queue.csv",
        review_rows
    )

    # --------------------------------------------------------
    # TOP 100
    # --------------------------------------------------------

    write_csv(
        reports_dir
        / "audit_v7_top100.csv",
        image_rows[:100]
    )

    # --------------------------------------------------------
    # SUMMARY CSV
    # --------------------------------------------------------

    summary_rows = []

    for key, value in summary.items():

        summary_rows.append({

            "metric":
                key,

            "value":
                value,
        })

    write_csv(
        reports_dir
        / "audit_v7_summary.csv",
        summary_rows
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    report_path = generate_text_report(
        summary,
        class_rows,
        class_anomaly_rows,
        all_results
    )

    # ========================================================
    # CONSOLA
    # ========================================================

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "RESULTADO AUDIT V7"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImágenes:              "
        f"{summary['images']:,}"
    )

    print(
        f"Personas:              "
        f"{summary['persons']:,}"
    )

    print(
        f"Vehículos:             "
        f"{summary['vehicles']:,}"
    )

    print(
        f"Objetos:               "
        f"{summary['objects']:,}"
    )

    print(
        f"Objetos/imagen:        "
        f"{summary['objects_per_image']:.2f}"
    )

    print(
        "\n"
    )

    print(
        "DECISIONES"
    )

    print(
        f"KEEP:                  "
        f"{summary['keep']:,} "
        f"({pct(summary['keep'], summary['images']):6.2f} %)"
    )

    print(
        f"REVIEW:                "
        f"{summary['review']:,} "
        f"({pct(summary['review'], summary['images']):6.2f} %)"
    )

    print(
        f"EXCLUDE_CANDIDATE:     "
        f"{summary['exclude_candidate']:,} "
        f"({pct(summary['exclude_candidate'], summary['images']):6.2f} %)"
    )

    print(
        f"CRITICAL:              "
        f"{summary['critical']:,} "
        f"({pct(summary['critical'], summary['images']):6.2f} %)"
    )

    print(
        "\n"
    )

    print(
        "OBJETOS PEQUEÑOS"
    )

    print(
        f"<16 px²:               "
        f"{summary['tiny16']:,}"
    )

    print(
        f"<32 px²:               "
        f"{summary['tiny32']:,}"
    )

    print(
        f"<64 px²:               "
        f"{summary['tiny64']:,}"
    )

    print(
        "\n"
    )

    print(
        "BORDES"
    )

    print(
        f"BBox parcialmente fuera: "
        f"{summary['partial_bbox']:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{summary['outside_bbox']:,}"
    )

    print(
        f"Cerca del borde:       "
        f"{summary['border_objects']:,}"
    )

    print(
        "\n"
    )

    print(
        "INTEGRIDAD"
    )

    print(
        f"Labels inválidos:      "
        f"{summary['invalid_labels']:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{summary['invalid_coordinates']:,}"
    )

    print(
        f"BBoxes inválidas:      "
        f"{summary['invalid_bbox']:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{summary['invalid_class']:,}"
    )

    print(
        f"Duplicados:            "
        f"{summary['duplicates']:,}"
    )

    # ========================================================
    # ESTADÍSTICAS POR CLASE
    # ========================================================

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "ANÁLISIS POR CLASE"
    )

    print(
        "=" * 70
    )

    for row in class_anomaly_rows:

        print(
            f"\n{row['class'].upper()}"
        )

        print(
            f"Objetos:       "
            f"{row['objects']:,}"
        )

        print(
            f"Tiny <16:     "
            f"{row['tiny16']:,} "
            f"({row['tiny16_pct']:.2f} %)"
        )

        print(
            f"Tiny <32:     "
            f"{row['tiny32']:,} "
            f"({row['tiny32_pct']:.2f} %)"
        )

        print(
            f"Tiny <64:     "
            f"{row['tiny64']:,} "
            f"({row['tiny64_pct']:.2f} %)"
        )

        print(
            f"Partial BBox: "
            f"{row['partial_bbox']:,}"
        )

        print(
            f"Border:       "
            f"{row['border_objects']:,}"
        )

    # ========================================================
    # TOP REVIEW
    # ========================================================

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "TOP 20 PARA REVISIÓN"
    )

    print(
        "=" * 70
    )

    for index, row in enumerate(
        review_rows[:20],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{row['decision']:<18} "
            f"score={row['score']:5.2f} "
            f"objects={row['objects']:4} "
            f"tiny16={row['tiny16']:3} "
            f"partial={row['partial_bbox']:2} "
            f"border={row['border_objects']:3}"
        )

        print(
            f"   reasons: "
            f"{row['reasons']}"
        )

        print(
            f"   {row['image']}"
        )

    print(
        "\n"
    )

    print(
        "Reports:"
    )

    print(
        reports_dir
    )

    print(
        "\nInforme:"
    )

    print(
        report_path
    )

    print(
        "\n"
    )

    print(
        "IMPORTANTE: este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes o labels."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()