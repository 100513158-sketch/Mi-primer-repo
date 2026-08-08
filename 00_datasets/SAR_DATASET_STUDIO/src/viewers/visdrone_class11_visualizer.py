from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


ROOT = Path(
    r"C:\SARC-Drone\00_DATASETS\SAR_DATASET_STUDIO\raw\VisDrone\original"
)

OUTPUT = Path(
    r"C:\SARC-Drone\00_DATASETS\SAR_DATASET_STUDIO\reports\validation\visual\class11"
)

TARGET_CLASS = 11
NUM_SAMPLES = 20

RANDOM_SEED = 42


def find_image(annotation_file: Path) -> Path | None:
    """
    Busca la imagen correspondiente a un fichero de anotaciones.
    """

    image_name = annotation_file.stem

    possible_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    for extension in possible_extensions:

        candidates = list(
            annotation_file.parent.parent.glob(
                f"images/{image_name}{extension}"
            )
        )

        if candidates:
            return candidates[0]

    # Búsqueda alternativa por todo el split
    split_root = annotation_file.parent.parent

    for extension in possible_extensions:

        candidates = list(
            split_root.rglob(
                f"{image_name}{extension}"
            )
        )

        if candidates:
            return candidates[0]

    return None


def parse_class11(annotation_file: Path):

    results = []

    try:

        lines = annotation_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    except Exception:

        return results

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

        results.append(
            {
                "line": line_number,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "score": score,
                "class_id": class_id,
                "truncation": truncation,
                "occlusion": occlusion,
                "raw": line,
            }
        )

    return results


def load_font(size=18):

    try:

        return ImageFont.truetype(
            "arial.ttf",
            size
        )

    except Exception:

        try:

            return ImageFont.truetype(
                "DejaVuSans.ttf",
                size
            )

        except Exception:

            return ImageFont.load_default()


def draw_box(
    draw,
    annotation,
    font
):

    x = annotation["x"]
    y = annotation["y"]

    w = annotation["width"]
    h = annotation["height"]

    x2 = x + w
    y2 = y + h

    # Caja principal
    draw.rectangle(
        [
            x,
            y,
            x2,
            y2
        ],
        outline="red",
        width=4
    )

    label = (
        f"CLASS 11 | "
        f"{w:.0f}x{h:.0f} | "
        f"occ={annotation['occlusion']} | "
        f"tr={annotation['truncation']}"
    )

    # Tamaño del texto
    bbox = draw.textbbox(
        (0, 0),
        label,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Intentamos poner la etiqueta encima
    label_x = x
    label_y = max(
        0,
        y - text_height - 6
    )

    # Fondo de etiqueta
    draw.rectangle(
        [
            label_x,
            label_y,
            label_x + text_width + 8,
            label_y + text_height + 6
        ],
        fill="red"
    )

    draw.text(
        (
            label_x + 4,
            label_y + 3
        ),
        label,
        fill="white",
        font=font
    )


def main():

    random.seed(RANDOM_SEED)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 75)
    print("VISUALIZADOR VISDRONE - CLASS 11")
    print("=" * 75)
    print()

    print(
        f"Dataset: {ROOT}"
    )

    print(
        f"Salida:  {OUTPUT}"
    )

    print()

    # ---------------------------------------------------------
    # Buscar anotaciones que contienen class 11
    # ---------------------------------------------------------

    candidates = []

    annotation_files = list(
        ROOT.rglob("*.txt")
    )

    print(
        f"Anotaciones encontradas: "
        f"{len(annotation_files)}"
    )

    print()
    print(
        "Buscando imágenes con class 11..."
    )

    for annotation_file in annotation_files:

        annotations = parse_class11(
            annotation_file
        )

        if not annotations:
            continue

        image_file = find_image(
            annotation_file
        )

        if image_file is None:

            print(
                "[WARN] No se encontró imagen:"
            )

            print(
                annotation_file
            )

            continue

        candidates.append(
            {
                "annotation": annotation_file,
                "image": image_file,
                "objects": annotations,
            }
        )

    print()
    print(
        f"Imágenes candidatas: "
        f"{len(candidates)}"
    )

    if not candidates:

        print()
        print(
            "[ERROR] No se encontraron "
            "imágenes con class 11."
        )

        return

    # ---------------------------------------------------------
    # Selección aleatoria
    # ---------------------------------------------------------

    if len(candidates) > NUM_SAMPLES:

        selected = random.sample(
            candidates,
            NUM_SAMPLES
        )

    else:

        selected = candidates

    print(
        f"Muestras seleccionadas: "
        f"{len(selected)}"
    )

    print()

    # ---------------------------------------------------------
    # Generación
    # ---------------------------------------------------------

    font = load_font(18)

    generated = 0

    for index, item in enumerate(
        selected,
        start=1
    ):

        annotation_file = item[
            "annotation"
        ]

        image_file = item[
            "image"
        ]

        objects = item[
            "objects"
        ]

        print(
            f"[{index}/{len(selected)}]"
        )

        print(
            f"Imagen: {image_file.name}"
        )

        print(
            f"Class 11: {len(objects)}"
        )

        try:

            image = Image.open(
                image_file
            ).convert("RGB")

        except Exception as exc:

            print(
                f"[ERROR] No se pudo abrir "
                f"{image_file}"
            )

            print(exc)

            continue

        draw = ImageDraw.Draw(
            image
        )

        # Dibujar TODAS las cajas class 11
        for annotation in objects:

            draw_box(
                draw,
                annotation,
                font
            )

        # Nombre de salida
        output_name = (
            f"{index:02d}_"
            f"{image_file.stem}_"
            f"class11.jpg"
        )

        output_file = (
            OUTPUT / output_name
        )

        image.save(
            output_file,
            quality=95
        )

        generated += 1

        print(
            f"[OK] {output_file}"
        )

        print()

    print("=" * 75)
    print(
        f"IMÁGENES GENERADAS: "
        f"{generated}/{len(selected)}"
    )
    print("=" * 75)

    print()
    print(
        "Revisa las imágenes en:"
    )

    print(
        OUTPUT
    )

    print()


if __name__ == "__main__":
    main()