from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "processed"
    / "converted"
    / "VisDrone"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "validation"
    / "VisDrone"
    / "class11"
)

CLASS_ID = 11

# Número máximo de imágenes a generar por split
MAX_IMAGES_PER_SPLIT = 30

# Mostrar también otras cajas para poder entender el contexto
DRAW_OTHER_CLASSES = True


# ============================================================
# COLORES
# ============================================================

CLASS11_COLOR = (255, 0, 0)
OTHER_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 0)


# ============================================================
# UTILIDADES
# ============================================================

def find_image(images_dir: Path, stem: str):
    """
    Busca la imagen correspondiente al label.
    """

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    for ext in extensions:
        candidate = images_dir / f"{stem}{ext}"

        if candidate.exists():
            return candidate

    return None


def yolo_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):
    """
    Convierte YOLO normalizado a coordenadas de imagen.
    """

    x_center *= image_width
    y_center *= image_height

    width *= image_width
    height *= image_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)

    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2


def clamp_box(x1, y1, x2, y2, width, height):

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    return x1, y1, x2, y2


# ============================================================
# PROCESAMIENTO DE UN LABEL
# ============================================================

def process_label(label_path: Path, images_dir: Path, output_dir: Path):

    stem = label_path.stem

    image_path = find_image(images_dir, stem)

    if image_path is None:
        print(f"[WARN] Imagen no encontrada: {stem}")
        return False

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[ERROR] No se pudo abrir {image_path}: {exc}")
        return False

    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    class11_count = 0
    other_count = 0

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception as exc:
        print(f"[ERROR] No se pudo leer {label_path}: {exc}")
        return False

    for line_number, line in enumerate(lines, start=1):

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center,
            y_center,
            width,
            height,
            image_width,
            image_height,
        )

        x1, y1, x2, y2 = clamp_box(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
        )

        # ----------------------------------------------------
        # CLASS 11
        # ----------------------------------------------------

        if class_id == CLASS_ID:

            class11_count += 1

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=CLASS11_COLOR,
                width=3,
            )

            text = f"CLASS 11 #{class11_count}"

            draw.text(
                (x1, max(0, y1 - 12)),
                text,
                fill=TEXT_COLOR,
                font=font,
            )

        # ----------------------------------------------------
        # OTRAS CLASES
        # ----------------------------------------------------

        elif DRAW_OTHER_CLASSES:

            other_count += 1

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=OTHER_COLOR,
                width=1,
            )

            draw.text(
                (x1, y1),
                str(class_id),
                fill=TEXT_COLOR,
                font=font,
            )

    # Solo guardar si realmente existe class 11
    if class11_count == 0:
        return False

    # Información en la imagen
    header = (
        f"CLASS 11: {class11_count} | "
        f"OTHER OBJECTS: {other_count} | "
        f"{stem}"
    )

    draw.rectangle(
        [0, 0, image_width, 20],
        fill=(0, 0, 0),
    )

    draw.text(
        (5, 4),
        header,
        fill=TEXT_COLOR,
        font=font,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / f"{stem}_class11.jpg"

    try:
        image.save(
            output_path,
            quality=95
        )
    except Exception as exc:
        print(f"[ERROR] No se pudo guardar {output_path}: {exc}")
        return False

    print(
        f"[OK] {stem} -> "
        f"{class11_count} CLASS 11"
    )

    return True


# ============================================================
# PROCESAR SPLIT
# ============================================================

def process_split(split: str):

    split_root = DATASET_ROOT / split

    images_dir = split_root / "images"
    labels_dir = split_root / "labels"

    output_dir = OUTPUT_ROOT / split

    print()
    print("=" * 70)
    print(f"SPLIT: {split}")
    print("=" * 70)

    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Output: {output_dir}")

    if not images_dir.exists():
        print("[ERROR] No existe el directorio de imágenes.")
        return

    if not labels_dir.exists():
        print("[ERROR] No existe el directorio de labels.")
        return

    label_files = sorted(
        labels_dir.rglob("*.txt")
    )

    print(
        f"Labels encontrados: {len(label_files)}"
    )

    generated = 0
    class11_labels = 0

    for label_path in label_files:

        # Comprobar primero si contiene class 11
        try:
            content = label_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception:
            continue

        has_class11 = False

        for line in content.splitlines():

            parts = line.strip().split()

            if len(parts) == 5:

                try:
                    if int(parts[0]) == CLASS_ID:
                        has_class11 = True
                        break
                except ValueError:
                    pass

        if not has_class11:
            continue

        class11_labels += 1

        if generated >= MAX_IMAGES_PER_SPLIT:
            continue

        if process_label(
            label_path,
            images_dir,
            output_dir,
        ):
            generated += 1

    print()
    print("RESULTADO")
    print("-" * 70)

    print(
        f"Labels con CLASS 11: {class11_labels}"
    )

    print(
        f"Imágenes generadas: {generated}"
    )

    print(
        f"Directorio: {output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VISUALIZADOR CLASS 11 - VISDRONE")
    print("=" * 70)

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Salida: {OUTPUT_ROOT}"
    )

    process_split("train")
    process_split("val")
    process_split("test_dev")

    print()
    print("=" * 70)
    print("PROCESO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    main()