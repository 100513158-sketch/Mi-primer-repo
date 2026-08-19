from pathlib import Path
from collections import defaultdict
import random
import math

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO"
    r"\processed\sar\VisDrone_SAR_2CLASS"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"\baseline\evaluation\dataset_analysis\audit"
)

EXAMPLES_ROOT = OUTPUT_ROOT / "visual_examples"

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}

RANDOM_SEED = 42

# Número de ejemplos que queremos visualizar
N_TINY_16 = 50
N_TINY_32 = 50
N_INVALID_COORDS = 50
N_CROWDED_200 = 20
N_CROWDED_300 = 20

# Umbrales
TINY_16 = 16
TINY_32 = 32
CROWDED_200 = 200
CROWDED_300 = 300

# ============================================================
# UTILIDADES
# ============================================================


def percentile(values, p):
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


def get_image_files(images_dir):
    files = []

    for ext in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ):
        files.extend(images_dir.rglob(ext))

    return sorted(files)


def read_labels(label_path):
    """
    Lee labels YOLO.

    Devuelve:
        [
            {
                class_id,
                xc,
                yc,
                w,
                h,
                area
            }
        ]
    """

    labels = []

    if not label_path.exists():
        return labels

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

                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                continue

            labels.append(
                {
                    "class_id": class_id,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                    "area": w * h,
                    "line": line_number,
                }
            )

    except Exception as exc:

        print(
            f"[ERROR] leyendo {label_path}: {exc}"
        )

    return labels


def normalized_to_pixels(
    label,
    image_width,
    image_height
):

    xc = label["xc"] * image_width
    yc = label["yc"] * image_height

    w = label["w"] * image_width
    h = label["h"] * image_height

    x1 = xc - w / 2
    y1 = yc - h / 2

    x2 = xc + w / 2
    y2 = yc + h / 2

    return (
        x1,
        y1,
        x2,
        y2,
    )


def bbox_is_outside(
    label,
    image_width,
    image_height
):

    x1, y1, x2, y2 = normalized_to_pixels(
        label,
        image_width,
        image_height,
    )

    return (
        x1 < 0
        or y1 < 0
        or x2 > image_width
        or y2 > image_height
    )


# ============================================================
# DIBUJAR IMAGEN
# ============================================================


def draw_bbox(
    draw,
    bbox,
    class_name,
    color,
    text
):

    x1, y1, x2, y2 = bbox

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=3,
    )

    # Fondo para el texto
    try:

        font = ImageFont.truetype(
            "arial.ttf",
            16,
        )

    except Exception:

        font = None

    if font:

        bbox_text = draw.textbbox(
            (x1, y1),
            text,
            font=font,
        )

        draw.rectangle(
            bbox_text,
            fill=color,
        )

        draw.text(
            (x1, y1),
            text,
            fill="white",
            font=font,
        )


def annotate_image(
    image_path,
    labels,
    output_path,
    title
):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as exc:

        print(
            f"[ERROR] imagen {image_path}: {exc}"
        )

        return False

    draw = ImageDraw.Draw(image)

    width, height = image.size

    colors = {
        0: "red",
        1: "blue",
    }

    for index, label in enumerate(labels):

        bbox = normalized_to_pixels(
            label,
            width,
            height,
        )

        class_id = label["class_id"]

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}",
        )

        area_px = (
            label["w"]
            * width
            * label["h"]
            * height
        )

        outside = bbox_is_outside(
            label,
            width,
            height,
        )

        if outside:
            color = "yellow"
        else:
            color = colors.get(
                class_id,
                "white",
            )

        text = (
            f"{class_name} "
            f"{area_px:.1f}px2"
        )

        if outside:
            text += " OUTSIDE"

        draw_bbox(
            draw,
            bbox,
            class_name,
            color,
            text,
        )

    # Título
    try:

        font = ImageFont.truetype(
            "arial.ttf",
            22,
        )

    except Exception:

        font = None

    header_height = 45

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height + header_height,
        ),
        "black",
    )

    canvas.paste(
        image,
        (0, header_height),
    )

    header_draw = ImageDraw.Draw(
        canvas
    )

    header_draw.text(
        (10, 10),
        title,
        fill="white",
        font=font,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        quality=95,
    )

    return True


# ============================================================
# ANALIZAR DATASET
# ============================================================


def analyze_dataset():

    random.seed(
        RANDOM_SEED
    )

    categories = {
        "tiny_16": [],
        "tiny_32": [],
        "invalid_coords": [],
        "crowded_200": [],
        "crowded_300": [],
    }

    splits = [
        "train",
        "val",
        "test",
        "test_dev",
    ]

    total_images = 0

    for split in splits:

        split_dir = (
            DATASET_ROOT / split
        )

        images_dir = (
            split_dir / "images"
        )

        labels_dir = (
            split_dir / "labels"
        )

        if not images_dir.exists():

            print(
                f"[INFO] Split no encontrado: "
                f"{split}"
            )

            continue

        print(
            f"\nAnalizando: {split}"
        )

        image_files = get_image_files(
            images_dir
        )

        print(
            f"Imágenes encontradas: "
            f"{len(image_files):,}"
        )

        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            total_images += 1

            if index % 1000 == 0:

                print(
                    f"Procesadas: "
                    f"{index:,}/"
                    f"{len(image_files):,}"
                )

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            labels = read_labels(
                label_path
            )

            # ------------------------------------------------
            # Dimensiones de imagen
            # ------------------------------------------------

            try:

                with Image.open(
                    image_path
                ) as img:

                    image_width, image_height = (
                        img.size
                    )

            except Exception:

                continue

            # ------------------------------------------------
            # Objetos
            # ------------------------------------------------

            object_count = len(
                labels
            )

            # ------------------------------------------------
            # Buscar tiny / invalid
            # ------------------------------------------------

            for label in labels:

                area_px = (
                    label["w"]
                    * image_width
                    * label["h"]
                    * image_height
                )

                item = {
                    "image": image_path,
                    "label": label,
                    "area_px": area_px,
                    "split": split,
                    "image_width": image_width,
                    "image_height": image_height,
                }

                if area_px < TINY_16:

                    categories[
                        "tiny_16"
                    ].append(item)

                elif area_px < TINY_32:

                    categories[
                        "tiny_32"
                    ].append(item)

                if bbox_is_outside(
                    label,
                    image_width,
                    image_height,
                ):

                    categories[
                        "invalid_coords"
                    ].append(item)

            # ------------------------------------------------
            # Escenas densas
            # ------------------------------------------------

            if object_count >= CROWDED_200:

                categories[
                    "crowded_200"
                ].append(
                    {
                        "image": image_path,
                        "labels": labels,
                        "objects": object_count,
                        "split": split,
                        "image_width": image_width,
                        "image_height": image_height,
                    }
                )

            if object_count >= CROWDED_300:

                categories[
                    "crowded_300"
                ].append(
                    {
                        "image": image_path,
                        "labels": labels,
                        "objects": object_count,
                        "split": split,
                        "image_width": image_width,
                        "image_height": image_height,
                    }
                )

    print(
        "\nTotal imágenes analizadas:",
        f"{total_images:,}",
    )

    return categories


# ============================================================
# SELECCIÓN ALEATORIA
# ============================================================


def random_sample(
    items,
    n
):

    if len(items) <= n:

        return list(items)

    return random.sample(
        items,
        n,
    )


# ============================================================
# GENERAR EJEMPLOS
# ============================================================


def generate_examples(
    categories
):

    print(
        "\nGenerando ejemplos visuales..."
    )

    # --------------------------------------------------------
    # TINY < 16
    # --------------------------------------------------------

    samples = random_sample(
        categories["tiny_16"],
        N_TINY_16,
    )

    output_dir = (
        EXAMPLES_ROOT
        / "tiny_under_16"
    )

    for i, item in enumerate(
        samples,
        start=1
    ):

        image_path = item["image"]

        label = item["label"]

        output_path = (
            output_dir
            / f"{i:03d}_"
            f"{image_path.stem}.jpg"
        )

        annotate_image(
            image_path,
            [label],
            output_path,
            (
                "TINY < 16 px2 | "
                f"area={item['area_px']:.2f} | "
                f"{item['split']}"
            ),
        )

    print(
        f"Tiny <16: "
        f"{len(samples)} imágenes"
    )

    # --------------------------------------------------------
    # TINY < 32
    # --------------------------------------------------------

    samples = random_sample(
        categories["tiny_32"],
        N_TINY_32,
    )

    output_dir = (
        EXAMPLES_ROOT
        / "tiny_16_32"
    )

    for i, item in enumerate(
        samples,
        start=1
    ):

        image_path = item["image"]

        label = item["label"]

        output_path = (
            output_dir
            / f"{i:03d}_"
            f"{image_path.stem}.jpg"
        )

        annotate_image(
            image_path,
            [label],
            output_path,
            (
                "TINY 16-32 px2 | "
                f"area={item['area_px']:.2f} | "
                f"{item['split']}"
            ),
        )

    print(
        f"Tiny 16-32: "
        f"{len(samples)} imágenes"
    )

    # --------------------------------------------------------
    # COORDENADAS FUERA DE IMAGEN
    # --------------------------------------------------------

    samples = random_sample(
        categories["invalid_coords"],
        N_INVALID_COORDS,
    )

    output_dir = (
        EXAMPLES_ROOT
        / "invalid_coordinates"
    )

    for i, item in enumerate(
        samples,
        start=1
    ):

        image_path = item["image"]

        label = item["label"]

        output_path = (
            output_dir
            / f"{i:03d}_"
            f"{image_path.stem}.jpg"
        )

        annotate_image(
            image_path,
            [label],
            output_path,
            (
                "COORDENADA FUERA | "
                f"area={item['area_px']:.2f} | "
                f"{item['split']}"
            ),
        )

    print(
        f"Coordenadas fuera: "
        f"{len(samples)} imágenes"
    )

    # --------------------------------------------------------
    # CROWDED >= 200
    # --------------------------------------------------------

    samples = random_sample(
        categories["crowded_200"],
        N_CROWDED_200,
    )

    output_dir = (
        EXAMPLES_ROOT
        / "crowded_200"
    )

    for i, item in enumerate(
        samples,
        start=1
    ):

        output_path = (
            output_dir
            / f"{i:03d}_"
            f"{item['image'].stem}.jpg"
        )

        annotate_image(
            item["image"],
            item["labels"],
            output_path,
            (
                f"CROWDED >= 200 | "
                f"objects={item['objects']} | "
                f"{item['split']}"
            ),
        )

    print(
        f"Crowded >=200: "
        f"{len(samples)} imágenes"
    )

    # --------------------------------------------------------
    # CROWDED >= 300
    # --------------------------------------------------------

    samples = random_sample(
        categories["crowded_300"],
        N_CROWDED_300,
    )

    output_dir = (
        EXAMPLES_ROOT
        / "crowded_300"
    )

    for i, item in enumerate(
        samples,
        start=1
    ):

        output_path = (
            output_dir
            / f"{i:03d}_"
            f"{item['image'].stem}.jpg"
        )

        annotate_image(
            item["image"],
            item["labels"],
            output_path,
            (
                f"CROWDED >= 300 | "
                f"objects={item['objects']} | "
                f"{item['split']}"
            ),
        )

    print(
        f"Crowded >=300: "
        f"{len(samples)} imágenes"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "=" * 70
    )

    print(
        "SAR YOLO26 - VISUAL DATASET AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        "\nDataset:"
    )

    print(
        DATASET_ROOT
    )

    print(
        "\nOutput:"
    )

    print(
        EXAMPLES_ROOT
    )

    categories = (
        analyze_dataset()
    )

    print(
        "\nResumen de anomalías:"
    )

    print(
        f"Objetos <16 px²: "
        f"{len(categories['tiny_16']):,}"
    )

    print(
        f"Objetos 16-32 px²: "
        f"{len(categories['tiny_32']):,}"
    )

    print(
        f"Coordenadas fuera: "
        f"{len(categories['invalid_coords']):,}"
    )

    print(
        f"Imágenes >=200 objetos: "
        f"{len(categories['crowded_200']):,}"
    )

    print(
        f"Imágenes >=300 objetos: "
        f"{len(categories['crowded_300']):,}"
    )

    generate_examples(
        categories
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AUDITORÍA VISUAL FINALIZADA"
    )

    print(
        "=" * 70
    )

    print(
        "\nEjemplos generados en:"
    )

    print(
        EXAMPLES_ROOT
    )


if __name__ == "__main__":
    main()