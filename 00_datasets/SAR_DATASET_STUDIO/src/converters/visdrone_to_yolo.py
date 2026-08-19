from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


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

CONVERTED_ROOT = (
    PROJECT_ROOT
    / "processed"
    / "converted"
    / "VisDrone"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "VisDrone"
)

SPLITS = [
    "train",
    "val",
    "test_dev",
]

# Clases oficiales de VisDrone2019-DET
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
    11: "class_11",
}


# ============================================================
# UTILIDADES
# ============================================================

def normalize_bbox(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convierte una bounding box de VisDrone:

        x, y, width, height

    a formato YOLO:

        center_x, center_y, width, height

    Todos los valores quedan normalizados entre 0 y 1.
    """

    center_x = x + width / 2.0
    center_y = y + height / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
    )


def clamp(value: float) -> float:
    """
    Limita un valor al rango [0, 1].
    """
    return max(0.0, min(1.0, value))


def parse_annotation_line(
    line: str,
) -> Tuple[int, float, float, float, float] | None:
    """
    Analiza una línea de annotation VisDrone.

    Formato:

    x,y,width,height,score,class,truncation,occlusion

    Devuelve:

    class_id, x, y, width, height
    """

    line = line.strip()

    if not line:
        return None

    parts = [
        p.strip()
        for p in line.rstrip(",").split(",")
    ]

    if len(parts) < 6:
        return None

    try:
        x = float(parts[0])
        y = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
        class_id = int(parts[5])
    except ValueError:
        return None

    return (
        class_id,
        x,
        y,
        width,
        height,
    )


def find_image(
    image_root: Path,
    stem: str,
) -> Path | None:
    """
    Busca una imagen correspondiente al annotation.
    """

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    for extension in extensions:

        candidate = image_root / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    return None


# ============================================================
# CONVERSIÓN DE UN SPLIT
# ============================================================

def convert_split(split: str) -> Dict:

    print()
    print("=" * 70)
    print(f"CONVIRTIENDO SPLIT: {split}")
    print("=" * 70)

    source_root = CLEANED_ROOT / split

    source_images = source_root / "images"
    source_annotations = source_root / "annotations"

    destination_root = CONVERTED_ROOT / split

    destination_images = destination_root / "images"
    destination_labels = destination_root / "labels"

    destination_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source_images.exists():
        raise FileNotFoundError(
            f"No existe: {source_images}"
        )

    if not source_annotations.exists():
        raise FileNotFoundError(
            f"No existe: {source_annotations}"
        )

    annotation_files = sorted(
        source_annotations.rglob("*.txt")
    )

    statistics = {
        "split": split,
        "images": 0,
        "annotations": 0,
        "converted_annotations": 0,
        "invalid_annotations": 0,
        "invalid_classes": 0,
        "invalid_boxes": 0,
        "images_without_annotations": 0,
        "annotations_without_images": 0,
        "copied_images": 0,
        "labels_created": 0,
        "classes": {},
    }

    processed = 0

    for annotation_file in annotation_files:

        processed += 1

        if processed % 500 == 0 or processed == len(annotation_files):
            print(
                f"Procesadas: "
                f"{processed}/{len(annotation_files)}"
            )

        stem = annotation_file.stem

        image_file = find_image(
            source_images,
            stem,
        )

        if image_file is None:

            statistics[
                "annotations_without_images"
            ] += 1

            continue

        # ----------------------------------------------------
        # Leer imagen para obtener dimensiones
        # ----------------------------------------------------

        try:

            from PIL import Image

            with Image.open(image_file) as image:
                image_width, image_height = image.size

        except Exception as exc:

            print(
                f"[WARNING] No se pudo leer imagen: "
                f"{image_file}"
            )

            print(f"          {exc}")

            continue

        # ----------------------------------------------------
        # Leer annotation
        # ----------------------------------------------------

        output_lines: List[str] = []

        lines = annotation_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        for line in lines:

            statistics["annotations"] += 1

            parsed = parse_annotation_line(line)

            if parsed is None:

                statistics[
                    "invalid_annotations"
                ] += 1

                continue

            (
                class_id,
                x,
                y,
                width,
                height,
            ) = parsed

            # ------------------------------------------------
            # Clase
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                statistics[
                    "invalid_classes"
                ] += 1

                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            if width <= 0 or height <= 0:

                statistics[
                    "invalid_boxes"
                ] += 1

                continue

            if image_width <= 0 or image_height <= 0:

                statistics[
                    "invalid_boxes"
                ] += 1

                continue

            # ------------------------------------------------
            # Conversión
            # ------------------------------------------------

            (
                center_x,
                center_y,
                norm_width,
                norm_height,
            ) = normalize_bbox(
                x,
                y,
                width,
                height,
                image_width,
                image_height,
            )

            # ------------------------------------------------
            # Limitar a rango YOLO
            # ------------------------------------------------

            center_x = clamp(center_x)
            center_y = clamp(center_y)

            norm_width = clamp(norm_width)
            norm_height = clamp(norm_height)

            # ------------------------------------------------
            # Guardar
            # ------------------------------------------------

            output_lines.append(
                f"{class_id} "
                f"{center_x:.8f} "
                f"{center_y:.8f} "
                f"{norm_width:.8f} "
                f"{norm_height:.8f}"
            )

            statistics[
                "converted_annotations"
            ] += 1

            class_name = CLASS_NAMES[class_id]

            statistics["classes"].setdefault(
                str(class_id),
                {
                    "name": class_name,
                    "count": 0,
                },
            )

            statistics["classes"][
                str(class_id)
            ]["count"] += 1

        # ----------------------------------------------------
        # Copiar imagen
        # ----------------------------------------------------

        destination_image = (
            destination_images
            / image_file.name
        )

        if not destination_image.exists():

            shutil.copy2(
                image_file,
                destination_image,
            )

            statistics[
                "copied_images"
            ] += 1

        # ----------------------------------------------------
        # Crear label
        # ----------------------------------------------------

        destination_label = (
            destination_labels
            / f"{stem}.txt"
        )

        destination_label.write_text(
            "\n".join(output_lines)
            + ("\n" if output_lines else ""),
            encoding="utf-8",
        )

        statistics[
            "labels_created"
        ] += 1

    # --------------------------------------------------------
    # Contar imágenes
    # --------------------------------------------------------

    converted_images = list(
        destination_images.glob("*")
    )

    statistics["images"] = len(
        [
            f
            for f in converted_images
            if f.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
            }
        ]
    )

    # --------------------------------------------------------
    # Informe
    # --------------------------------------------------------

    report_directory = (
        REPORT_ROOT / split
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        report_directory
        / "conversion_report.json"
    )

    report_file.write_text(
        json.dumps(
            statistics,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print()
    print("## RESULTADO")
    print()
    print(
        f"Imágenes copiadas:       "
        f"{statistics['copied_images']}"
    )

    print(
        f"Annotations originales:  "
        f"{statistics['annotations']}"
    )

    print(
        f"Annotations convertidas: "
        f"{statistics['converted_annotations']}"
    )

    print(
        f"Annotations inválidas:   "
        f"{statistics['invalid_annotations']}"
    )

    print(
        f"Clases inválidas:        "
        f"{statistics['invalid_classes']}"
    )

    print(
        f"Boxes inválidas:         "
        f"{statistics['invalid_boxes']}"
    )

    print(
        f"Annotations sin imagen:  "
        f"{statistics['annotations_without_images']}"
    )

    print(
        f"Labels generados:        "
        f"{statistics['labels_created']}"
    )

    print()
    print("## CLASES")

    for class_id in sorted(
        statistics["classes"],
        key=lambda x: int(x),
    ):

        information = statistics[
            "classes"
        ][class_id]

        print(
            f"{class_id:>2} "
            f"{information['name']:<20} "
            f"{information['count']}"
        )

    print()
    print(
        f"[OK] Informe generado:"
    )

    print(report_file)

    return statistics


# ============================================================
# VALIDACIÓN FINAL
# ============================================================

def validate_converted_dataset():

    print()
    print("=" * 70)
    print("VALIDACIÓN DEL DATASET CONVERTIDO")
    print("=" * 70)

    global_statistics = {
        "images": 0,
        "labels": 0,
        "missing_labels": 0,
        "orphan_labels": 0,
        "invalid_lines": 0,
        "invalid_classes": 0,
        "invalid_values": 0,
    }

    for split in SPLITS:

        split_root = CONVERTED_ROOT / split

        images_root = (
            split_root / "images"
        )

        labels_root = (
            split_root / "labels"
        )

        if not images_root.exists():
            continue

        image_files = []

        for extension in (
            "*.jpg",
            "*.jpeg",
            "*.png",
        ):

            image_files.extend(
                images_root.glob(extension)
            )

        label_files = list(
            labels_root.glob("*.txt")
        )

        image_stems = {
            f.stem
            for f in image_files
        }

        label_stems = {
            f.stem
            for f in label_files
        }

        missing_labels = (
            image_stems - label_stems
        )

        orphan_labels = (
            label_stems - image_stems
        )

        invalid_lines = 0
        invalid_classes = 0
        invalid_values = 0

        for label_file in label_files:

            lines = label_file.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()

            for line in lines:

                parts = line.split()

                if len(parts) != 5:

                    invalid_lines += 1
                    continue

                try:

                    class_id = int(parts[0])

                    values = [
                        float(value)
                        for value in parts[1:]
                    ]

                except ValueError:

                    invalid_lines += 1
                    continue

                if class_id not in CLASS_NAMES:

                    invalid_classes += 1

                for value in values:

                    if not (
                        0.0
                        <= value
                        <= 1.0
                    ):

                        invalid_values += 1

        print()
        print(f"--- {split.upper()} ---")

        print(
            f"Imágenes:              "
            f"{len(image_files)}"
        )

        print(
            f"Labels:                "
            f"{len(label_files)}"
        )

        print(
            f"Imágenes sin label:     "
            f"{len(missing_labels)}"
        )

        print(
            f"Labels sin imagen:      "
            f"{len(orphan_labels)}"
        )

        print(
            f"Líneas inválidas:      "
            f"{invalid_lines}"
        )

        print(
            f"Clases inválidas:      "
            f"{invalid_classes}"
        )

        print(
            f"Valores fuera [0,1]:   "
            f"{invalid_values}"
        )

        global_statistics["images"] += len(
            image_files
        )

        global_statistics["labels"] += len(
            label_files
        )

        global_statistics[
            "missing_labels"
        ] += len(missing_labels)

        global_statistics[
            "orphan_labels"
        ] += len(orphan_labels)

        global_statistics[
            "invalid_lines"
        ] += invalid_lines

        global_statistics[
            "invalid_classes"
        ] += invalid_classes

        global_statistics[
            "invalid_values"
        ] += invalid_values

    print()
    print("=" * 70)
    print("RESULTADO GLOBAL")
    print("=" * 70)

    print(
        f"Imágenes:             "
        f"{global_statistics['images']}"
    )

    print(
        f"Labels:               "
        f"{global_statistics['labels']}"
    )

    print(
        f"Imágenes sin label:    "
        f"{global_statistics['missing_labels']}"
    )

    print(
        f"Labels sin imagen:     "
        f"{global_statistics['orphan_labels']}"
    )

    print(
        f"Líneas inválidas:     "
        f"{global_statistics['invalid_lines']}"
    )

    print(
        f"Clases inválidas:     "
        f"{global_statistics['invalid_classes']}"
    )

    print(
        f"Valores fuera [0,1]:  "
        f"{global_statistics['invalid_values']}"
    )

    validation_file = (
        REPORT_ROOT
        / "global_conversion_validation.json"
    )

    validation_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_file.write_text(
        json.dumps(
            global_statistics,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[OK] Informe global:"
    )

    print(validation_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VISDRONE → YOLO")
    print("=" * 70)

    print()
    print(
        f"ORIGEN: "
        f"{CLEANED_ROOT}"
    )

    print(
        f"DESTINO: "
        f"{CONVERTED_ROOT}"
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "El directorio CLEANED no será modificado."
    )

    print()

    CONVERTED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:

        convert_split(split)

    validate_converted_dataset()

    print()
    print("=" * 70)
    print("CONVERSIÓN FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()