from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import csv

# ============================================================
# SAR YOLO26 - DATASET AUDIT V2
# SOLO DIAGNOSTICO: NO MODIFICA EL DATASET ORIGINAL
# ============================================================

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO"
    r"\processed\sar\VisDrone_SAR_2CLASS"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26"
    r"\baseline\evaluation\dataset_analysis\audit_v2"
)

# Splits reales del dataset
SPLITS = [
    "train",
    "val",
    "test_dev",
]

# Clases del dataset SAR 2 CLASS
CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}


# ============================================================
# UMBRALES DE AUDITORÍA
# ============================================================

# Objetos extremadamente pequeños
TINY = 16

# Objetos muy pequeños
VERY_SMALL = 32

# Objetos pequeños
SMALL = 64

# Límite para considerar un objeto pequeño
MEDIUM = 250

# Distancia en píxeles al borde para marcar proximidad
BORDER_MARGIN_PX = 2

# Umbrales para escenas muy densas
CROWDED = [
    100,
    150,
    200,
    300,
    400,
    500,
]

# Máximo de ejemplos visuales por categoría
MAX_EXAMPLES = 50

# Máximo de imágenes crowded
MAX_CROWDED_200 = 20
MAX_CROWDED_300 = 20


# ============================================================
# UTILIDADES
# ============================================================

def image_files(directory):
    """
    Devuelve todas las imágenes encontradas
    recursivamente dentro de un directorio.
    """

    out = []

    for ext in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ):
        out.extend(directory.rglob(ext))

    return sorted(set(out))


def write_csv(path, rows):
    """
    Escribe una lista de diccionarios en CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = (
        list(rows[0].keys())
        if rows
        else []
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# LECTURA DE LABELS
# ============================================================

def read_labels(path):
    """
    Lee un fichero YOLO.

    Formato esperado:

    class x_center y_center width height

    Devuelve:

        rows
        errors
    """

    if not path.exists():

        return [], [
            "LABEL_FILE_MISSING"
        ]

    rows = []
    errors = []

    try:

        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

    except Exception as exc:

        return [], [
            f"LABEL_READ_ERROR:{exc}"
        ]

    for line_number, line in enumerate(
        lines,
        1,
    ):

        parts = line.split()

        if not parts:
            continue

        # ====================================================
        # FORMATO YOLO
        # ====================================================

        if len(parts) != 5:

            errors.append(
                f"INVALID_FIELD_COUNT_LINE_{line_number}"
            )

            continue

        try:

            values = [
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            ]

        except ValueError:

            errors.append(
                f"INVALID_NUMERIC_LINE_{line_number}"
            )

            continue

        rows.append(
            {
                "line": line_number,
                "class_id": values[0],
                "x": values[1],
                "y": values[2],
                "w": values[3],
                "h": values[4],
            }
        )

    return rows, errors


# ============================================================
# ANALIZAR SPLIT
# ============================================================

def analyze_split(split_dir):

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # ========================================================
    # VALIDAR DIRECTORIOS
    # ========================================================

    if not images_dir.exists():

        print(
            f"[WARN] No existe: {images_dir}"
        )

        return None

    if not labels_dir.exists():

        print(
            f"[WARN] No existe: {labels_dir}"
        )

        return None

    # ========================================================
    # BUSCAR IMÁGENES
    # ========================================================

    files = image_files(images_dir)

    print(
        f"Imágenes encontradas: "
        f"{len(files):,}"
    )

    objects = []
    images = []

    counters = defaultdict(int)

    # ========================================================
    # PROCESAR IMÁGENES
    # ========================================================

    for i, image_path in enumerate(
        files,
        1,
    ):

        if i % 1000 == 0:

            print(
                f"Procesadas: "
                f"{i:,}/{len(files):,}"
            )

        # ====================================================
        # OBTENER DIMENSIONES
        # ====================================================

        try:

            with Image.open(image_path) as im:

                image_width, image_height = im.size

        except Exception:

            counters[
                "image_read_error"
            ] += 1

            continue

        # ====================================================
        # LABEL
        # ====================================================

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        labels, errors = read_labels(
            label_path
        )

        counters[
            "invalid_labels"
        ] += len(errors)

        if "LABEL_FILE_MISSING" in errors:

            counters[
                "missing_labels"
            ] += 1

        # ====================================================
        # CONTADORES POR IMAGEN
        # ====================================================

        person_count = 0
        vehicle_count = 0

        # ====================================================
        # PROCESAR CADA LABEL
        # ====================================================

        for row in labels:

            class_id = row["class_id"]

            x = row["x"]
            y = row["y"]
            w = row["w"]
            h = row["h"]

            # =================================================
            # CLASE
            # =================================================

            if class_id not in CLASS_NAMES:

                counters[
                    "invalid_class"
                ] += 1

                continue

            # =================================================
            # BBOX POSITIVO
            # =================================================

            if w <= 0 or h <= 0:

                counters[
                    "invalid_bbox"
                ] += 1

                continue

            # =================================================
            # COORDENADAS NORMALIZADAS
            # =================================================

            coordinates_valid = (
                0 <= x <= 1
                and
                0 <= y <= 1
                and
                0 < w <= 1
                and
                0 < h <= 1
            )

            if not coordinates_valid:

                counters[
                    "invalid_coordinates"
                ] += 1

            # =================================================
            # CONVERTIR A PIXELES
            # =================================================

            center_x = (
                x * image_width
            )

            center_y = (
                y * image_height
            )

            bbox_width = (
                w * image_width
            )

            bbox_height = (
                h * image_height
            )

            x1 = (
                center_x
                - bbox_width / 2
            )

            y1 = (
                center_y
                - bbox_height / 2
            )

            x2 = (
                center_x
                + bbox_width / 2
            )

            y2 = (
                center_y
                + bbox_height / 2
            )

            # =================================================
            # ÁREA
            # =================================================

            area = (
                bbox_width
                * bbox_height
            )

            # =================================================
            # CATEGORÍA DE TAMAÑO
            # =================================================

            if area < TINY:

                size_category = (
                    "extreme_tiny"
                )

            elif area < VERY_SMALL:

                size_category = (
                    "tiny_16_32"
                )

            elif area < SMALL:

                size_category = (
                    "very_small_32_64"
                )

            elif area < MEDIUM:

                size_category = (
                    "small_64_250"
                )

            else:

                size_category = (
                    "normal"
                )

            counters[
                size_category
            ] += 1

            # =================================================
            # OBJETO FUERA DE IMAGEN
            # =================================================

            center_outside = (
                center_x < 0
                or center_x > image_width
                or center_y < 0
                or center_y > image_height
            )

            bbox_outside = (
                x1 < 0
                or y1 < 0
                or x2 > image_width
                or y2 > image_height
            )

            partial_outside = (
                bbox_outside
                and
                not center_outside
            )

            # =================================================
            # CERCA DEL BORDE
            # =================================================

            near_border = (
                x1 <= BORDER_MARGIN_PX
                or
                y1 <= BORDER_MARGIN_PX
                or
                x2 >= image_width - BORDER_MARGIN_PX
                or
                y2 >= image_height - BORDER_MARGIN_PX
            )

            # =================================================
            # CONTADORES
            # =================================================

            counters[
                "center_outside"
            ] += int(center_outside)

            counters[
                "partial_outside"
            ] += int(partial_outside)

            counters[
                "outside_any"
            ] += int(bbox_outside)

            counters[
                "near_border"
            ] += int(near_border)

            counters[
                "persons"
            ] += int(class_id == 0)

            counters[
                "vehicles"
            ] += int(class_id == 1)

            counters[
                "objects"
            ] += 1

            # =================================================
            # CONTADORES POR IMAGEN
            # =================================================

            if class_id == 0:

                person_count += 1

            else:

                vehicle_count += 1

            # =================================================
            # REGISTRO DEL OBJETO
            # =================================================

            objects.append(
                {
                    "split": split_dir.name,

                    "image": str(
                        image_path
                    ),

                    "label": str(
                        label_path
                    ),

                    "line": row["line"],

                    "class_id": class_id,

                    "class_name":
                        CLASS_NAMES[
                            class_id
                        ],

                    "image_width":
                        image_width,

                    "image_height":
                        image_height,

                    "x_norm": x,
                    "y_norm": y,
                    "w_norm": w,
                    "h_norm": h,

                    "x1_px": x1,
                    "y1_px": y1,
                    "x2_px": x2,
                    "y2_px": y2,

                    "bbox_width_px":
                        bbox_width,

                    "bbox_height_px":
                        bbox_height,

                    "area_px2":
                        area,

                    "size_category":
                        size_category,

                    "outside_any":
                        bbox_outside,

                    "center_outside":
                        center_outside,

                    "partial_outside":
                        partial_outside,

                    "near_border":
                        near_border,
                }
            )

        # ====================================================
        # REGISTRO DE LA IMAGEN
        # ====================================================

        total_objects = (
            person_count
            + vehicle_count
        )

        images.append(
            {
                "split":
                    split_dir.name,

                "image":
                    str(image_path),

                "label":
                    str(label_path),

                "objects":
                    total_objects,

                "persons":
                    person_count,

                "vehicles":
                    vehicle_count,

                "image_width":
                    image_width,

                "image_height":
                    image_height,
            }
        )

        if total_objects == 0:

            counters[
                "images_without_objects"
            ] += 1

    # ========================================================
    # ESTADÍSTICAS DEL SPLIT
    # ========================================================

    counters["images"] = len(files)
    counters["split"] = split_dir.name

    return (
        counters,
        objects,
        images,
    )


# ============================================================
# FUENTE
# ============================================================

def font(size=18):

    try:

        return ImageFont.truetype(
            "arial.ttf",
            size,
        )

    except Exception:

        return ImageFont.load_default()


# ============================================================
# EJEMPLOS VISUALES
# ============================================================

def visual_examples(
    objects,
    out_root,
):

    grouped = defaultdict(list)

    # Agrupar objetos por imagen
    for obj in objects:

        key = str(
            Path(
                obj["image"]
            ).resolve()
        ).lower()

        grouped[key].append(obj)

    # ========================================================
    # REGLAS
    # ========================================================

    rules = {

        "extreme_tiny":
            lambda o:
                o["area_px2"] < TINY,

        "tiny_16_32":
            lambda o:
                TINY
                <= o["area_px2"]
                < VERY_SMALL,

        "very_small_32_64":
            lambda o:
                VERY_SMALL
                <= o["area_px2"]
                < SMALL,

        "center_outside":
            lambda o:
                o["center_outside"],

        "partial_outside":
            lambda o:
                o["partial_outside"],
    }

    # ========================================================
    # GENERAR IMÁGENES
    # ========================================================

    for name, predicate in rules.items():

        selected = []

        for rows in grouped.values():

            if any(
                predicate(o)
                for o in rows
            ):

                selected.append(rows)

            if len(selected) >= MAX_EXAMPLES:

                break

        output_dir = (
            out_root
            / "examples"
            / name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"{name}: "
            f"{len(selected)} "
            f"imágenes únicas"
        )

        for number, rows in enumerate(
            selected,
            1,
        ):

            path = Path(
                rows[0]["image"]
            )

            try:

                image = Image.open(
                    path
                ).convert("RGB")

            except Exception:

                continue

            draw = ImageDraw.Draw(
                image
            )

            # =================================================
            # DIBUJAR BBOX
            # =================================================

            for obj in rows:

                x1 = max(
                    0,
                    obj["x1_px"]
                )

                y1 = max(
                    0,
                    obj["y1_px"]
                )

                x2 = min(
                    image.width,
                    obj["x2_px"]
                )

                y2 = min(
                    image.height,
                    obj["y2_px"]
                )

                special = (
                    obj["center_outside"]
                    or
                    obj["partial_outside"]
                )

                if special:

                    color = "red"

                elif (
                    obj["area_px2"]
                    < VERY_SMALL
                ):

                    color = "yellow"

                else:

                    color = "lime"

                draw.rectangle(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    outline=color,
                    width=3,
                )

                draw.text(
                    (
                        x1,
                        max(
                            0,
                            y1 - 20,
                        )
                    ),
                    (
                        f'{obj["class_name"]} '
                        f'{obj["area_px2"]:.0f}px²'
                    ),
                    fill=color,
                    font=font(15),
                )

            # =================================================
            # CABECERA
            # =================================================

            canvas = Image.new(
                "RGB",
                (
                    image.width,
                    image.height + 45,
                ),
                "black",
            )

            canvas.paste(
                image,
                (0, 45),
            )

            ImageDraw.Draw(
                canvas
            ).text(
                (10, 10),
                (
                    f"{name.upper()} | "
                    f"{path.name} | "
                    f"objetos={len(rows)}"
                ),
                fill="white",
                font=font(20),
            )

            canvas.save(
                output_dir
                / f"{number:03d}_{path.stem}.jpg",
                quality=92,
            )


# ============================================================
# IMÁGENES CROWDED
# ============================================================

def crowded_examples(
    images,
    out_root,
):

    for threshold, limit in (
        (200, MAX_CROWDED_200),
        (300, MAX_CROWDED_300),
    ):

        rows = sorted(
            [
                row
                for row in images
                if row["objects"]
                >= threshold
            ],
            key=lambda x:
                x["objects"],
            reverse=True,
        )[:limit]

        output_dir = (
            out_root
            / "examples"
            / f"crowded_ge_{threshold}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"crowded >= {threshold}: "
            f"{len(rows)} imágenes"
        )

        for number, row in enumerate(
            rows,
            1,
        ):

            try:

                image = Image.open(
                    row["image"]
                ).convert("RGB")

            except Exception:

                continue

            canvas = Image.new(
                "RGB",
                (
                    image.width,
                    image.height + 45,
                ),
                "black",
            )

            canvas.paste(
                image,
                (0, 45),
            )

            ImageDraw.Draw(
                canvas
            ).text(
                (10, 10),
                (
                    f"CROWDED >= "
                    f"{threshold} | "
                    f"objects="
                    f"{row['objects']} | "
                    f"P={row['persons']} "
                    f"V={row['vehicles']}"
                ),
                fill="white",
                font=font(20),
            )

            canvas.save(
                output_dir
                / (
                    f"{number:03d}_"
                    f"{Path(row['image']).stem}.jpg"
                ),
                quality=92,
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "# SAR YOLO26 - DATASET AUDIT V2"
    )

    print("\nDataset:")
    print(DATASET_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    # ========================================================
    # VALIDAR DATASET
    # ========================================================

    if not DATASET_ROOT.exists():

        print(
            "\n[ERROR] No existe DATASET_ROOT:"
        )

        print(DATASET_ROOT)

        return

    # ========================================================
    # ACUMULADORES
    # ========================================================

    all_objects = []
    all_images = []
    split_rows = []

    # ========================================================
    # ANALIZAR SPLITS
    # ========================================================

    for split in SPLITS:

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Analizando: {split}"
        )

        print(
            "-" * 70
        )

        split_dir = (
            DATASET_ROOT
            / split
        )

        if not split_dir.exists():

            print(
                f"[INFO] Split no encontrado: "
                f"{split}"
            )

            continue

        result = analyze_split(
            split_dir
        )

        if result:

            counters, objects, images = (
                result
            )

            all_objects.extend(
                objects
            )

            all_images.extend(
                images
            )

            split_rows.append(
                dict(counters)
            )

    # ========================================================
    # DIRECTORIO REPORTS
    # ========================================================

    reports = (
        OUTPUT_ROOT
        / "reports"
    )

    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REPORTES COMPLETOS
    # ========================================================

    write_csv(
        reports
        / "all_objects.csv",
        all_objects,
    )

    write_csv(
        reports
        / "all_images.csv",
        all_images,
    )

    # ========================================================
    # OBJETOS TINY
    # ========================================================

    write_csv(
        reports
        / "tiny_objects.csv",
        [
            obj
            for obj in all_objects
            if obj["area_px2"]
            < VERY_SMALL
        ],
    )

    # ========================================================
    # OBJETOS DE BORDE
    # ========================================================

    write_csv(
        reports
        / "border_objects.csv",
        [
            obj
            for obj in all_objects
            if (
                obj["outside_any"]
                or
                obj["near_border"]
            )
        ],
    )

    # ========================================================
    # CENTRO FUERA
    # ========================================================

    write_csv(
        reports
        / "center_outside_objects.csv",
        [
            obj
            for obj in all_objects
            if obj["center_outside"]
        ],
    )

    # ========================================================
    # BBOX PARCIALMENTE FUERA
    # ========================================================

    write_csv(
        reports
        / "partial_outside_objects.csv",
        [
            obj
            for obj in all_objects
            if obj["partial_outside"]
        ],
    )

    # ========================================================
    # OBJETOS SOSPECHOSOS
    # ========================================================

    suspicious = [
        obj
        for obj in all_objects
        if (
            obj["x_norm"] < -0.05
            or
            obj["x_norm"] > 1.05
            or
            obj["y_norm"] < -0.05
            or
            obj["y_norm"] > 1.05
            or
            obj["w_norm"] <= 0
            or
            obj["h_norm"] <= 0
        )
    ]

    write_csv(
        reports
        / "suspicious_objects.csv",
        suspicious,
    )

    # ========================================================
    # IMÁGENES CROWDED
    # ========================================================

    crowded_images = sorted(
        [
            row
            for row in all_images
            if row["objects"] >= 100
        ],
        key=lambda x:
            x["objects"],
        reverse=True,
    )

    write_csv(
        reports
        / "crowded_images_unique.csv",
        crowded_images,
    )

    # ========================================================
    # RESUMEN POR SPLIT
    # ========================================================

    write_csv(
        reports
        / "audit_summary_by_split.csv",
        split_rows,
    )

    # ========================================================
    # EJEMPLOS VISUALES
    # ========================================================

    visual_examples(
        all_objects,
        OUTPUT_ROOT,
    )

    crowded_examples(
        all_images,
        OUTPUT_ROOT,
    )

    # ========================================================
    # ESTADÍSTICAS GENERALES
    # ========================================================

    total_images = len(
        all_images
    )

    total_objects = len(
        all_objects
    )

    total_persons = sum(
        obj["class_id"] == 0
        for obj in all_objects
    )

    total_vehicles = sum(
        obj["class_id"] == 1
        for obj in all_objects
    )

    # ========================================================
    # RESULTADO GENERAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESULTADO GENERAL"
    )

    print(
        "=" * 70
    )

    print(
        f"Imágenes:              "
        f"{total_images:,}"
    )

    print(
        f"Personas:              "
        f"{total_persons:,}"
    )

    print(
        f"Vehículos:             "
        f"{total_vehicles:,}"
    )

    print(
        f"Objetos:               "
        f"{total_objects:,}"
    )

    if total_images:

        print(
            f"Objetos/imagen:        "
            f"{total_objects / total_images:.2f}"
        )

    else:

        print(
            "Objetos/imagen:        0"
        )

    # ========================================================
    # TAMAÑOS
    # ========================================================

    print(
        "\nTAMAÑO"
    )

    size_rules = (

        (
            "<16 px²",
            lambda o:
                o["area_px2"] < TINY,
        ),

        (
            "16-32 px²",
            lambda o:
                TINY
                <= o["area_px2"]
                < VERY_SMALL,
        ),

        (
            "32-64 px²",
            lambda o:
                VERY_SMALL
                <= o["area_px2"]
                < SMALL,
        ),
    )

    for label, predicate in size_rules:

        print(
            f"{label:20}: "
            f"{sum(predicate(o) for o in all_objects):,}"
        )

    # ========================================================
    # BORDE / COORDENADAS
    # ========================================================

    print(
        "\nBORDE / COORDENADAS"
    )

    border_rules = (

        (
            "Centro fuera",
            lambda o:
                o["center_outside"],
        ),

        (
            "BBox parcialmente fuera",
            lambda o:
                o["partial_outside"],
        ),

        (
            "BBox fuera",
            lambda o:
                o["outside_any"],
        ),

        (
            "Cerca del borde",
            lambda o:
                o["near_border"],
        ),
    )

    for label, predicate in border_rules:

        print(
            f"{label:24}: "
            f"{sum(predicate(o) for o in all_objects):,}"
        )

    # ========================================================
    # CROWDED
    # ========================================================

    print(
        "\nCROWDED"
    )

    for threshold in CROWDED:

        count = sum(
            row["objects"]
            >= threshold
            for row in all_images
        )

        print(
            f">= {threshold:3} objetos: "
            f"{count:,} imágenes"
        )

    # ========================================================
    # RUTAS
    # ========================================================

    print(
        "\nReports:"
    )

    print(
        reports
    )

    print(
        "\nExamples:"
    )

    print(
        OUTPUT_ROOT
        / "examples"
    )

    print(
        "\nIMPORTANTE: "
        "este script SOLO diagnostica. "
        "No elimina ni modifica imágenes "
        "o labels."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()