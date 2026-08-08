from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_ROOT = (
    PROJECT_ROOT
    / "processed"
    / "cleaned"
    / "VisDrone"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "statistics"
    / "VisDrone"
)


CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
    10: "others",
}


SPLITS = [
    "train",
    "val",
    "test_dev",
]


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def find_images(images_dir: Path) -> list[Path]:
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def annotation_for_image(
    image_path: Path,
    images_dir: Path,
    annotations_dir: Path,
) -> Path:
    relative = image_path.relative_to(images_dir)

    return annotations_dir / relative.with_suffix(".txt")


def size_category(area_ratio: float) -> str:
    """
    Categoría basada en porcentaje del área de la imagen.

    < 0.001  -> tiny
    < 0.01   -> small
    < 0.05   -> medium
    >= 0.05  -> large
    """

    if area_ratio < 0.001:
        return "tiny"

    if area_ratio < 0.01:
        return "small"

    if area_ratio < 0.05:
        return "medium"

    return "large"


# ============================================================
# PROCESAMIENTO DE ANOTACIONES
# ============================================================

def process_annotation(
    line: str,
    image_width: int,
    image_height: int,
    statistics: dict,
) -> None:

    line = line.strip()

    if not line:
        return

    parts = [
        p.strip()
        for p in line.rstrip(",").split(",")
    ]

    if len(parts) < 8:
        statistics["format_errors"] += 1
        return

    x = safe_float(parts[0])
    y = safe_float(parts[1])
    width = safe_float(parts[2])
    height = safe_float(parts[3])
    score = safe_int(parts[4])
    class_id = safe_int(parts[5])
    truncation = safe_int(parts[6])
    occlusion = safe_int(parts[7])

    statistics["annotations"] += 1

    # --------------------------------------------------------
    # Clase
    # --------------------------------------------------------

    if class_id not in CLASS_NAMES:
        statistics["invalid_classes"] += 1
        class_name = f"class_{class_id}"
    else:
        class_name = CLASS_NAMES[class_id]

    statistics["classes"][class_name] += 1

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    statistics["scores"][str(score)] += 1

    # --------------------------------------------------------
    # Truncamiento
    # --------------------------------------------------------

    statistics["truncation"][str(truncation)] += 1

    # --------------------------------------------------------
    # Oclusión
    # --------------------------------------------------------

    statistics["occlusion"][str(occlusion)] += 1

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    if width <= 0 or height <= 0:
        statistics["invalid_boxes"] += 1
        return

    image_area = image_width * image_height

    if image_area <= 0:
        statistics["invalid_boxes"] += 1
        return

    bbox_area = width * height

    area_ratio = bbox_area / image_area

    statistics["bbox_area_ratio_sum"] += area_ratio

    statistics["bbox_width_sum"] += width
    statistics["bbox_height_sum"] += height

    statistics["bbox_width_min"] = min(
        statistics["bbox_width_min"],
        width,
    )

    statistics["bbox_width_max"] = max(
        statistics["bbox_width_max"],
        width,
    )

    statistics["bbox_height_min"] = min(
        statistics["bbox_height_min"],
        height,
    )

    statistics["bbox_height_max"] = max(
        statistics["bbox_height_max"],
        height,
    )

    # --------------------------------------------------------
    # Categoría por tamaño
    # --------------------------------------------------------

    category = size_category(area_ratio)

    statistics["bbox_size_categories"][category] += 1

    # --------------------------------------------------------
    # Bounding boxes extremadamente pequeñas
    # --------------------------------------------------------

    if width <= 4 or height <= 4:
        statistics["very_small_boxes"] += 1

    if width <= 8 or height <= 8:
        statistics["small_dimension_boxes"] += 1

    # --------------------------------------------------------
    # Posición de la caja
    # --------------------------------------------------------

    if x < 0 or y < 0:
        statistics["out_of_image_boxes"] += 1

    if x + width > image_width:
        statistics["out_of_image_boxes"] += 1

    if y + height > image_height:
        statistics["out_of_image_boxes"] += 1


# ============================================================
# ESTRUCTURA DE ESTADÍSTICAS
# ============================================================

def create_statistics() -> dict:

    return {
        "images": 0,
        "annotations": 0,

        "format_errors": 0,
        "invalid_classes": 0,
        "invalid_boxes": 0,
        "out_of_image_boxes": 0,

        "images_without_annotations": 0,

        "classes": Counter(),

        "scores": Counter(),

        "truncation": Counter(),

        "occlusion": Counter(),

        "bbox_size_categories": Counter(),

        "very_small_boxes": 0,
        "small_dimension_boxes": 0,

        "bbox_area_ratio_sum": 0.0,

        "bbox_width_sum": 0.0,
        "bbox_height_sum": 0.0,

        "bbox_width_min": math.inf,
        "bbox_width_max": 0.0,

        "bbox_height_min": math.inf,
        "bbox_height_max": 0.0,

        "objects_per_image": [],

        "max_objects_in_image": 0,

        "images_by_object_count": Counter(),
    }


# ============================================================
# ANALIZAR SPLIT
# ============================================================

def analyze_split(split: str) -> dict:

    split_root = CLEANED_ROOT / split

    if not split_root.exists():
        raise FileNotFoundError(
            f"No existe el split: {split_root}"
        )

    images_dir = split_root / "images"
    annotations_dir = split_root / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"No existe: {images_dir}"
        )

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"No existe: {annotations_dir}"
        )

    print()
    print("=" * 70)
    print(f"ANALIZANDO SPLIT: {split}")
    print(f"Images:      {images_dir}")
    print(f"Annotations: {annotations_dir}")
    print("=" * 70)

    images = find_images(images_dir)

    print(f"Imágenes encontradas: {len(images)}")

    statistics = create_statistics()

    statistics["images"] = len(images)

    for index, image_path in enumerate(images, start=1):

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size

        except Exception as exc:
            print(
                f"[WARN] No se pudo abrir imagen: "
                f"{image_path}"
            )

            print(f"       {exc}")

            continue

        annotation_path = annotation_for_image(
            image_path,
            images_dir,
            annotations_dir,
        )

        image_objects = 0

        if annotation_path.exists():

            try:
                with annotation_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    for line in file:

                        before = statistics["annotations"]

                        process_annotation(
                            line,
                            image_width,
                            image_height,
                            statistics,
                        )

                        after = statistics["annotations"]

                        if after > before:
                            image_objects += 1

            except Exception as exc:

                print(
                    f"[WARN] Error leyendo annotation: "
                    f"{annotation_path}"
                )

                print(f"       {exc}")

        else:
            statistics["images_without_annotations"] += 1

        statistics["objects_per_image"].append(
            image_objects
        )

        statistics["images_by_object_count"][
            str(image_objects)
        ] += 1

        statistics["max_objects_in_image"] = max(
            statistics["max_objects_in_image"],
            image_objects,
        )

        if index % 500 == 0 or index == len(images):

            print(
                f"Procesadas: "
                f"{index}/{len(images)}"
            )

    # ========================================================
    # PROMEDIOS
    # ========================================================

    annotations = statistics["annotations"]

    if annotations > 0:

        statistics["average_bbox_area_ratio"] = (
            statistics["bbox_area_ratio_sum"]
            / annotations
        )

        statistics["average_bbox_width"] = (
            statistics["bbox_width_sum"]
            / annotations
        )

        statistics["average_bbox_height"] = (
            statistics["bbox_height_sum"]
            / annotations
        )

    else:

        statistics["average_bbox_area_ratio"] = 0.0
        statistics["average_bbox_width"] = 0.0
        statistics["average_bbox_height"] = 0.0

    if statistics["bbox_width_min"] == math.inf:
        statistics["bbox_width_min"] = 0.0

    if statistics["bbox_height_min"] == math.inf:
        statistics["bbox_height_min"] = 0.0

    if statistics["objects_per_image"]:

        statistics["average_objects_per_image"] = (
            sum(statistics["objects_per_image"])
            / len(statistics["objects_per_image"])
        )

    else:

        statistics["average_objects_per_image"] = 0.0

    # ========================================================
    # CONVERSIÓN DE COUNTERS
    # ========================================================

    statistics["classes"] = dict(
        statistics["classes"]
    )

    statistics["scores"] = dict(
        statistics["scores"]
    )

    statistics["truncation"] = dict(
        statistics["truncation"]
    )

    statistics["occlusion"] = dict(
        statistics["occlusion"]
    )

    statistics["bbox_size_categories"] = dict(
        statistics["bbox_size_categories"]
    )

    statistics["images_by_object_count"] = dict(
        statistics["images_by_object_count"]
    )

    # No necesitamos guardar toda la lista.
    statistics.pop(
        "objects_per_image",
        None,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    print()
    print("RESULTADO")
    print("-" * 70)

    print(
        f"Imágenes:                  "
        f"{statistics['images']}"
    )

    print(
        f"Annotations:               "
        f"{statistics['annotations']}"
    )

    print(
        f"Promedio objetos/imagen:   "
        f"{statistics['average_objects_per_image']:.2f}"
    )

    print(
        f"Máximo objetos/imagen:     "
        f"{statistics['max_objects_in_image']}"
    )

    print(
        f"Boxes muy pequeñas:        "
        f"{statistics['very_small_boxes']}"
    )

    print(
        f"Boxes dimensión <= 8 px:   "
        f"{statistics['small_dimension_boxes']}"
    )

    print(
        f"Boxes inválidas:           "
        f"{statistics['invalid_boxes']}"
    )

    print(
        f"Fuera de imagen:           "
        f"{statistics['out_of_image_boxes']}"
    )

    print()
    print("CLASES")
    print("-" * 70)

    for class_name, count in sorted(
        statistics["classes"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        percentage = (
            count / annotations * 100
            if annotations
            else 0
        )

        print(
            f"{class_name:20s} "
            f"{count:8d} "
            f"{percentage:7.2f}%"
        )

    print()
    print("TAMAÑO DE BOUNDING BOXES")
    print("-" * 70)

    for category, count in statistics[
        "bbox_size_categories"
    ].items():

        percentage = (
            count / annotations * 100
            if annotations
            else 0
        )

        print(
            f"{category:10s} "
            f"{count:8d} "
            f"{percentage:7.2f}%"
        )

    return statistics


# ============================================================
# GUARDAR JSON
# ============================================================

def save_report(
    split: str,
    statistics: dict,
) -> Path:

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        REPORT_ROOT
        / f"{split}_statistics.json"
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            statistics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output


# ============================================================
# GLOBAL
# ============================================================

def create_global_statistics(
    reports: dict[str, dict],
) -> dict:

    global_stats = {
        "dataset": "VisDrone2019-DET",
        "splits": {},
        "global": {
            "images": 0,
            "annotations": 0,
            "classes": Counter(),
            "bbox_size_categories": Counter(),
            "truncation": Counter(),
            "occlusion": Counter(),
            "very_small_boxes": 0,
            "small_dimension_boxes": 0,
            "invalid_boxes": 0,
            "out_of_image_boxes": 0,
        },
    }

    for split, stats in reports.items():

        global_stats["splits"][split] = {
            "images": stats["images"],
            "annotations": stats["annotations"],
        }

        global_stats["global"]["images"] += (
            stats["images"]
        )

        global_stats["global"]["annotations"] += (
            stats["annotations"]
        )

        global_stats["global"]["very_small_boxes"] += (
            stats["very_small_boxes"]
        )

        global_stats["global"]["small_dimension_boxes"] += (
            stats["small_dimension_boxes"]
        )

        global_stats["global"]["invalid_boxes"] += (
            stats["invalid_boxes"]
        )

        global_stats["global"]["out_of_image_boxes"] += (
            stats["out_of_image_boxes"]
        )

        for key, value in stats["classes"].items():
            global_stats["global"]["classes"][key] += value

        for key, value in stats[
            "bbox_size_categories"
        ].items():

            global_stats[
                "global"
            ]["bbox_size_categories"][key] += value

        for key, value in stats["truncation"].items():
            global_stats["global"]["truncation"][key] += value

        for key, value in stats["occlusion"].items():
            global_stats["global"]["occlusion"][key] += value

    global_stats["global"]["classes"] = dict(
        global_stats["global"]["classes"]
    )

    global_stats["global"]["bbox_size_categories"] = dict(
        global_stats["global"]["bbox_size_categories"]
    )

    global_stats["global"]["truncation"] = dict(
        global_stats["global"]["truncation"]
    )

    global_stats["global"]["occlusion"] = dict(
        global_stats["global"]["occlusion"]
    )

    return global_stats


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VISDRONE DATASET STATISTICS")
    print("=" * 70)

    print(f"Dataset limpio:")
    print(CLEANED_ROOT)

    print()
    print(f"Informes:")
    print(REPORT_ROOT)

    reports = {}

    for split in SPLITS:

        try:

            statistics = analyze_split(split)

            output = save_report(
                split,
                statistics,
            )

            reports[split] = statistics

            print()
            print(
                f"[OK] Informe generado:"
            )

            print(output)

        except FileNotFoundError as exc:

            print()
            print(
                f"[ERROR] {exc}"
            )

    # ========================================================
    # INFORME GLOBAL
    # ========================================================

    if reports:

        global_statistics = create_global_statistics(
            reports
        )

        output = (
            REPORT_ROOT
            / "global_statistics.json"
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                global_statistics,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("=" * 70)
        print("INFORME GLOBAL")
        print("=" * 70)

        print(
            f"Imágenes: "
            f"{global_statistics['global']['images']}"
        )

        print(
            f"Annotations: "
            f"{global_statistics['global']['annotations']}"
        )

        print()
        print(
            "Informe:"
        )

        print(output)


if __name__ == "__main__":
    main()