from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path(
    r"C:\SARC-Drone\00_DATASETS\SAR_DATASET_STUDIO\raw\VisDrone\original"
)

TARGET_CLASS = 11

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
    11: "UNKNOWN_11",
}


def main():

    total_annotations = 0
    total_files = 0

    by_split = Counter()
    by_score = Counter()
    by_truncation = Counter()
    by_occlusion = Counter()

    width_values = []
    height_values = []

    examples = []

    files_with_class11 = set()

    print()
    print("=" * 75)
    print("INSPECCIÓN DE VISDRONE - CLASS 11")
    print("=" * 75)
    print()
    print(f"Dataset root: {ROOT}")
    print()

    annotation_files = list(ROOT.rglob("*.txt"))

    for annotation_file in annotation_files:

        try:
            lines = annotation_file.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

        except Exception as exc:

            print(
                f"[ERROR] No se pudo leer: "
                f"{annotation_file}"
            )

            print(exc)
            continue

        file_has_class11 = False

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = [
                p.strip()
                for p in line.rstrip(",").split(",")
            ]

            if len(parts) < 8:
                continue

            try:

                x = float(parts[0])
                y = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])

                score = int(parts[4])
                class_id = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])

            except ValueError:

                continue

            if class_id != TARGET_CLASS:
                continue

            total_annotations += 1
            file_has_class11 = True

            width_values.append(width)
            height_values.append(height)

            by_score[score] += 1
            by_truncation[truncation] += 1
            by_occlusion[occlusion] += 1

            relative = annotation_file.relative_to(ROOT)

            parts_path = relative.parts

            if len(parts_path) > 0:
                split = parts_path[0]
            else:
                split = "UNKNOWN"

            by_split[split] += 1

            if len(examples) < 30:

                examples.append(
                    {
                        "file": str(annotation_file),
                        "line": line_number,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "score": score,
                        "class": class_id,
                        "truncation": truncation,
                        "occlusion": occlusion,
                        "raw": line,
                    }
                )

        if file_has_class11:

            total_files += 1
            files_with_class11.add(annotation_file)

    print("-" * 75)
    print("RESUMEN")
    print("-" * 75)

    print(
        f"Annotations class 11: {total_annotations}"
    )

    print(
        f"Ficheros que contienen class 11: "
        f"{total_files}"
    )

    print()

    print("POR SPLIT")
    print("-" * 75)

    for split, count in sorted(by_split.items()):

        print(
            f"{split:<20} {count:>8}"
        )

    print()

    print("SCORE")
    print("-" * 75)

    for value, count in sorted(by_score.items()):

        print(
            f"{value:<20} {count:>8}"
        )

    print()

    print("TRUNCATION")
    print("-" * 75)

    for value, count in sorted(by_truncation.items()):

        print(
            f"{value:<20} {count:>8}"
        )

    print()

    print("OCCLUSION")
    print("-" * 75)

    for value, count in sorted(by_occlusion.items()):

        print(
            f"{value:<20} {count:>8}"
        )

    print()

    if width_values:

        print("TAMAÑO DE BOUNDING BOX")
        print("-" * 75)

        print(
            f"Width mínimo:  {min(width_values):.2f}"
        )

        print(
            f"Width máximo:  {max(width_values):.2f}"
        )

        print(
            f"Width medio:   "
            f"{sum(width_values) / len(width_values):.2f}"
        )

        print(
            f"Height mínimo: {min(height_values):.2f}"
        )

        print(
            f"Height máximo: {max(height_values):.2f}"
        )

        print(
            f"Height medio:  "
            f"{sum(height_values) / len(height_values):.2f}"
        )

    print()
    print("=" * 75)
    print("EJEMPLOS DE CLASS 11")
    print("=" * 75)

    for i, example in enumerate(
        examples,
        start=1
    ):

        print()
        print(f"[{i}]")
        print(f"Archivo:       {example['file']}")
        print(f"Línea:         {example['line']}")
        print(
            f"Bounding box:  "
            f"x={example['x']} "
            f"y={example['y']} "
            f"w={example['width']} "
            f"h={example['height']}"
        )
        print(f"Score:         {example['score']}")
        print(
            f"Truncation:    "
            f"{example['truncation']}"
        )
        print(
            f"Occlusion:     "
            f"{example['occlusion']}"
        )
        print(f"Contenido:     {example['raw']}")

    print()
    print("=" * 75)
    print("FIN DE AUDITORÍA")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()