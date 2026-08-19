"""
SARC-DRONE - VISUAL ANALYSIS
============================

Análisis visual del modelo YOLO26n baseline.

El script utiliza las predicciones previamente generadas en:

    evaluation/predictions/predictions.csv

y genera ejemplos visuales organizados por categorías:

    evaluation/visual/
        low_confidence/
        small_person/
        crowded_scenes/
        many_detections/
        high_confidence/
        comparison/
        reports/

Objetivo:
    Inspeccionar visualmente el comportamiento del detector antes
    de realizar nuevas modificaciones sobre el dataset o el modelo.

No realiza entrenamiento.
No modifica el dataset.
No modifica predictions.csv.
"""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_CSV = (
    BASE_DIR
    / "evaluation"
    / "predictions"
    / "predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "evaluation" / "visual"

# Número máximo de imágenes que queremos guardar por categoría
TOP_N = 20

# Umbral de baja confianza
LOW_CONFIDENCE_THRESHOLD = 0.40

# Área máxima considerada "persona pequeña"
SMALL_PERSON_AREA = 500

# Tamaño mínimo de imagen para evitar problemas
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100


# ============================================================
# DIRECTORIOS
# ============================================================

DIR_LOW_CONFIDENCE = OUTPUT_DIR / "low_confidence"
DIR_SMALL_PERSON = OUTPUT_DIR / "small_person"
DIR_CROWDED = OUTPUT_DIR / "crowded_scenes"
DIR_MANY_DETECTIONS = OUTPUT_DIR / "many_detections"
DIR_HIGH_CONFIDENCE = OUTPUT_DIR / "high_confidence"
DIR_COMPARISON = OUTPUT_DIR / "comparison"

DIR_REPORTS = OUTPUT_DIR / "reports"


# ============================================================
# COLORES
# ============================================================

# BGR para OpenCV

COLOR_PERSON = (0, 255, 255)
COLOR_VEHICLE = (255, 180, 0)
COLOR_LOW_CONF = (0, 0, 255)

COLOR_TEXT = (255, 255, 255)
COLOR_BACKGROUND = (30, 30, 30)


# ============================================================
# UTILIDADES
# ============================================================

def create_directories():
    """Crea la estructura de salida."""

    directories = [
        DIR_LOW_CONFIDENCE,
        DIR_SMALL_PERSON,
        DIR_CROWDED,
        DIR_MANY_DETECTIONS,
        DIR_HIGH_CONFIDENCE,
        DIR_COMPARISON,
        DIR_REPORTS,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def print_header(title):
    """Imprime una sección del programa."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CARGA DE PREDICCIONES
# ============================================================

def load_predictions():
    """Carga predictions.csv y valida las columnas necesarias."""

    print_header("SARC-DRONE - VISUAL ANALYSIS")

    print(f"Cargando predicciones:")
    print(PREDICTIONS_CSV)

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"No existe el fichero de predicciones:\n{PREDICTIONS_CSV}"
        )

    df = pd.read_csv(PREDICTIONS_CSV)

    print(f"Predicciones cargadas: {len(df):,}")

    required_columns = [
        "image",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias en predictions.csv: {missing}"
        )

    # Conversión defensiva de tipos
    numeric_columns = [
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Eliminar filas corruptas
    df = df.dropna(
        subset=numeric_columns + ["image", "class_name"]
    ).copy()

    # Área de bounding box
    df["bbox_width"] = (
        df["x2"] - df["x1"]
    ).clip(lower=0)

    df["bbox_height"] = (
        df["y2"] - df["y1"]
    ).clip(lower=0)

    df["bbox_area"] = (
        df["bbox_width"] *
        df["bbox_height"]
    )

    return df


# ============================================================
# ESTADÍSTICAS POR IMAGEN
# ============================================================

def calculate_image_statistics(df):
    """
    Calcula estadísticas agregadas por imagen.
    """

    print_header("CALCULANDO ESTADÍSTICAS POR IMAGEN")

    grouped = []

    for image_path, group in df.groupby("image"):

        persons = int(
            (group["class_name"] == "person").sum()
        )

        vehicles = int(
            (group["class_name"] == "vehicle").sum()
        )

        total = len(group)

        grouped.append(
            {
                "image": image_path,
                "persons": persons,
                "vehicles": vehicles,
                "total_detections": total,
                "mean_confidence": group[
                    "confidence"
                ].mean(),
                "min_confidence": group[
                    "confidence"
                ].min(),
                "max_confidence": group[
                    "confidence"
                ].max(),
            }
        )

    stats = pd.DataFrame(grouped)

    print(f"Imágenes analizadas: {len(stats):,}")

    return stats


# ============================================================
# SELECCIÓN DE IMÁGENES
# ============================================================

def select_low_confidence(df):
    """
    Selecciona imágenes con personas detectadas
    con baja confianza.
    """

    persons = df[
        (df["class_name"] == "person")
        & (
            df["confidence"]
            < LOW_CONFIDENCE_THRESHOLD
        )
    ]

    if persons.empty:
        return []

    result = (
        persons.groupby("image")
        .agg(
            low_confidence_count=("confidence", "count"),
            min_confidence=("confidence", "min"),
            mean_confidence=("confidence", "mean"),
        )
        .sort_values(
            ["low_confidence_count", "min_confidence"],
            ascending=[False, True],
        )
        .head(TOP_N)
    )

    return list(result.index)


def select_small_persons(df):
    """
    Selecciona imágenes que contienen personas
    con bounding boxes pequeños.
    """

    persons = df[
        (df["class_name"] == "person")
        & (
            df["bbox_area"]
            <= SMALL_PERSON_AREA
        )
    ]

    if persons.empty:
        return []

    result = (
        persons.groupby("image")
        .agg(
            small_person_count=("bbox_area", "count"),
            min_bbox_area=("bbox_area", "min"),
            mean_bbox_area=("bbox_area", "mean"),
        )
        .sort_values(
            ["small_person_count", "min_bbox_area"],
            ascending=[False, True],
        )
        .head(TOP_N)
    )

    return list(result.index)


def select_crowded_scenes(image_stats):
    """
    Selecciona las escenas con mayor cantidad de personas.
    """

    result = (
        image_stats
        .sort_values(
            ["persons", "total_detections"],
            ascending=[False, False],
        )
        .head(TOP_N)
    )

    return list(result["image"])


def select_many_detections(image_stats):
    """
    Selecciona las imágenes con mayor número
    total de detecciones.
    """

    result = (
        image_stats
        .sort_values(
            "total_detections",
            ascending=False,
        )
        .head(TOP_N)
    )

    return list(result["image"])


def select_high_confidence(df):
    """
    Selecciona imágenes con detecciones de alta confianza.

    Se utiliza principalmente como referencia visual
    de situaciones donde el modelo funciona correctamente.
    """

    image_stats = (
        df.groupby("image")
        .agg(
            mean_confidence=("confidence", "mean"),
            detections=("confidence", "count"),
        )
    )

    # Evitamos seleccionar imágenes con una única detección
    image_stats = image_stats[
        image_stats["detections"] >= 5
    ]

    result = (
        image_stats
        .sort_values(
            ["mean_confidence", "detections"],
            ascending=[False, False],
        )
        .head(TOP_N)
    )

    return list(result.index)


# ============================================================
# DIBUJAR PREDICCIONES
# ============================================================

def draw_predictions(image_path, predictions):
    """
    Dibuja las bounding boxes de las predicciones
    sobre la imagen original.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"[WARN] No se pudo cargar imagen: {image_path}"
        )
        return None

    height, width = image.shape[:2]

    if (
        width < MIN_IMAGE_WIDTH
        or height < MIN_IMAGE_HEIGHT
    ):
        print(
            f"[WARN] Imagen demasiado pequeña: "
            f"{image_path}"
        )
        return None

    for _, row in predictions.iterrows():

        x1 = int(max(0, min(width - 1, row["x1"])))
        y1 = int(max(0, min(height - 1, row["y1"])))
        x2 = int(max(0, min(width - 1, row["x2"])))
        y2 = int(max(0, min(height - 1, row["y2"])))

        confidence = float(row["confidence"])
        class_name = str(row["class_name"])

        if class_name == "person":
            color = COLOR_PERSON
        elif class_name == "vehicle":
            color = COLOR_VEHICLE
        else:
            color = COLOR_TEXT

        # Destacar detecciones de baja confianza
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            color = COLOR_LOW_CONF

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        # Tamaño del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )
        )

        # Fondo del texto
        text_y = max(
            y1,
            text_height + baseline + 2
        )

        cv2.rectangle(
            image,
            (
                x1,
                text_y - text_height - baseline - 2,
            ),
            (
                x1 + text_width + 4,
                text_y,
            ),
            color,
            -1,
        )

        cv2.putText(
            image,
            label,
            (x1 + 2, text_y - 2),
            font,
            font_scale,
            COLOR_TEXT,
            thickness,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# INFORMACIÓN SUPERIOR
# ============================================================

def add_information_panel(
    image,
    image_name,
    predictions,
):
    """
    Añade información estadística a la parte superior
    de la imagen.
    """

    persons = int(
        (
            predictions["class_name"]
            == "person"
        ).sum()
    )

    vehicles = int(
        (
            predictions["class_name"]
            == "vehicle"
        ).sum()
    )

    total = len(predictions)

    mean_conf = predictions[
        "confidence"
    ].mean()

    panel_height = 70

    canvas = cv2.copyMakeBorder(
        image,
        panel_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=COLOR_BACKGROUND,
    )

    text1 = (
        f"Persons: {persons} | "
        f"Vehicles: {vehicles} | "
        f"Detections: {total}"
    )

    text2 = (
        f"Mean confidence: "
        f"{mean_conf:.3f}"
    )

    cv2.putText(
        canvas,
        image_name,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text1,
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text2,
        (10, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# GENERACIÓN DE VISUALIZACIONES
# ============================================================

def generate_category_images(
    df,
    image_list,
    output_directory,
    category_name,
):
    """
    Genera las imágenes correspondientes a una categoría.
    """

    print()
    print(
        f"Generando categoría: "
        f"{category_name}"
    )

    generated = 0

    for index, image_path_string in enumerate(
        image_list,
        start=1,
    ):

        image_path = Path(image_path_string)

        if not image_path.exists():
            print(
                f"[WARN] Imagen no encontrada: "
                f"{image_path}"
            )
            continue

        predictions = df[
            df["image"] == image_path_string
        ]

        if predictions.empty:
            continue

        rendered = draw_predictions(
            image_path,
            predictions,
        )

        if rendered is None:
            continue

        rendered = add_information_panel(
            rendered,
            image_path.name,
            predictions,
        )

        output_name = (
            f"{index:02d}_"
            f"{image_path.stem}_"
            f"{category_name}.jpg"
        )

        output_path = (
            output_directory
            / output_name
        )

        success = cv2.imwrite(
            str(output_path),
            rendered,
        )

        if success:
            generated += 1

    print(
        f"Imágenes generadas: {generated}"
    )

    return generated


# ============================================================
# CSV DE RESUMEN
# ============================================================

def generate_summary_csv(
    df,
    image_stats,
    category_images,
):
    """
    Genera un CSV resumen de las imágenes seleccionadas.
    """

    rows = []

    for category, images in category_images.items():

        for image in images:

            stats = image_stats[
                image_stats["image"] == image
            ]

            if stats.empty:
                continue

            row = stats.iloc[0].to_dict()

            row["category"] = category

            rows.append(row)

    if not rows:
        return

    summary = pd.DataFrame(rows)

    summary_path = (
        DIR_REPORTS
        / "visual_analysis_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"CSV generado:\n{summary_path}"
    )


# ============================================================
# INFORME TXT
# ============================================================

def generate_text_report(
    df,
    image_stats,
    category_images,
):
    """
    Genera informe textual de la inspección visual.
    """

    report_path = (
        DIR_REPORTS
        / "visual_analysis_summary.txt"
    )

    persons = df[
        df["class_name"] == "person"
    ]

    vehicles = df[
        df["class_name"] == "vehicle"
    ]

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "SARC-DRONE - VISUAL ANALYSIS\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        report.write(
            f"Predicciones analizadas: "
            f"{len(df):,}\n"
        )

        report.write(
            f"Imágenes analizadas: "
            f"{len(image_stats):,}\n\n"
        )

        report.write(
            "ESTADÍSTICAS GENERALES\n"
        )

        report.write(
            "-" * 80 + "\n"
        )

        report.write(
            f"Person predictions: "
            f"{len(persons):,}\n"
        )

        report.write(
            f"Vehicle predictions: "
            f"{len(vehicles):,}\n"
        )

        report.write(
            f"Mean confidence persons: "
            f"{persons['confidence'].mean():.4f}\n"
        )

        report.write(
            f"Mean confidence vehicles: "
            f"{vehicles['confidence'].mean():.4f}\n"
        )

        report.write("\n")

        report.write(
            "CONFIGURACIÓN\n"
        )

        report.write(
            "-" * 80 + "\n"
        )

        report.write(
            f"Low confidence threshold: "
            f"{LOW_CONFIDENCE_THRESHOLD}\n"
        )

        report.write(
            f"Small person bbox area: "
            f"{SMALL_PERSON_AREA}\n"
        )

        report.write(
            f"Top N: {TOP_N}\n\n"
        )

        report.write(
            "IMÁGENES SELECCIONADAS\n"
        )

        report.write(
            "=" * 80 + "\n\n"
        )

        for category, images in category_images.items():

            report.write(
                f"\n[{category}]\n"
            )

            report.write(
                "-" * 80 + "\n"
            )

            for image in images:
                report.write(
                    f"{image}\n"
                )

    print(
        f"Informe generado:\n{report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    # --------------------------------------------------------
    # 1. Cargar predicciones
    # --------------------------------------------------------

    df = load_predictions()

    # --------------------------------------------------------
    # 2. Estadísticas por imagen
    # --------------------------------------------------------

    image_stats = calculate_image_statistics(
        df
    )

    # --------------------------------------------------------
    # 3. Selección de casos interesantes
    # --------------------------------------------------------

    category_images = {}

    category_images["low_confidence"] = (
        select_low_confidence(df)
    )

    category_images["small_person"] = (
        select_small_persons(df)
    )

    category_images["crowded_scenes"] = (
        select_crowded_scenes(image_stats)
    )

    category_images["many_detections"] = (
        select_many_detections(image_stats)
    )

    category_images["high_confidence"] = (
        select_high_confidence(df)
    )

    # --------------------------------------------------------
    # 4. Mostrar resumen
    # --------------------------------------------------------

    print_header(
        "IMÁGENES SELECCIONADAS"
    )

    for category, images in category_images.items():

        print(
            f"{category:20s}: "
            f"{len(images)}"
        )

    # --------------------------------------------------------
    # 5. Generar imágenes
    # --------------------------------------------------------

    directories = {
        "low_confidence": DIR_LOW_CONFIDENCE,
        "small_person": DIR_SMALL_PERSON,
        "crowded_scenes": DIR_CROWDED,
        "many_detections": DIR_MANY_DETECTIONS,
        "high_confidence": DIR_HIGH_CONFIDENCE,
    }

    total_generated = 0

    for category, images in category_images.items():

        total_generated += (
            generate_category_images(
                df,
                images,
                directories[category],
                category,
            )
        )

    # --------------------------------------------------------
    # 6. Generar CSV
    # --------------------------------------------------------

    generate_summary_csv(
        df,
        image_stats,
        category_images,
    )

    # --------------------------------------------------------
    # 7. Generar informe
    # --------------------------------------------------------

    generate_text_report(
        df,
        image_stats,
        category_images,
    )

    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print_header(
        "PROCESO COMPLETADO"
    )

    print(
        f"Total imágenes generadas: "
        f"{total_generated}"
    )

    print(
        f"\nResultados:"
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()