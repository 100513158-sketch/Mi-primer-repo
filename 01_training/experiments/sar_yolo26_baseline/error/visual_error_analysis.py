from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import json

# ============================================================
# SARC-DRONE
# VISUAL ERROR ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = (
    BASE_DIR
    / "evaluation"
    / "predictions"
    / "predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "analysis"
    / "visual"
)

# Número de imágenes por categoría
TOP_N = 20

# Umbrales
HIGH_CONF = 0.70
LOW_CONF_MIN = 0.20
LOW_CONF_MAX = 0.40

# Área aproximada de bbox
SMALL_AREA = 500


# ============================================================
# UTILIDADES
# ============================================================

def create_dirs():
    dirs = [
        OUTPUT_DIR / "person" / "high_confidence",
        OUTPUT_DIR / "person" / "low_confidence",
        OUTPUT_DIR / "person" / "very_small",
        OUTPUT_DIR / "person" / "crowded",
        OUTPUT_DIR / "vehicle" / "high_confidence",
        OUTPUT_DIR / "vehicle" / "low_confidence",
        OUTPUT_DIR / "crowded_scenes",
        OUTPUT_DIR / "reports",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_predictions():

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"No existe predictions.csv:\n{PREDICTIONS_CSV}"
        )

    print()
    print("=" * 80)
    print("SARC-DRONE - VISUAL ERROR ANALYSIS")
    print("=" * 80)

    print()
    print("Cargando predicciones...")
    print(PREDICTIONS_CSV)

    df = pd.read_csv(PREDICTIONS_CSV)

    print(f"Predicciones cargadas: {len(df):,}")

    required = [
        "image",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas en predictions.csv: {missing}"
        )

    # Área de bounding box
    df["bbox_width"] = df["x2"] - df["x1"]
    df["bbox_height"] = df["y2"] - df["y1"]
    df["bbox_area"] = (
        df["bbox_width"] * df["bbox_height"]
    )

    return df


# ============================================================
# DIBUJAR DETECCIONES
# ============================================================

def draw_predictions(image_path, detections):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: no se pudo abrir {image_path}")
        return None

    for _, row in detections.iterrows():

        x1 = int(max(0, row["x1"]))
        y1 = int(max(0, row["y1"]))
        x2 = int(max(0, row["x2"]))
        y2 = int(max(0, row["y2"]))

        confidence = float(row["confidence"])
        class_name = str(row["class_name"])

        # Verde para person
        # Azul para vehicle
        if class_name == "person":
            color = (0, 255, 0)
        else:
            color = (255, 120, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# GUARDAR ESCENAS
# ============================================================

def save_detection_group(
    df,
    output_subdir,
    name_prefix,
    max_images=TOP_N,
):

    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seleccionar imágenes únicas
    images = []

    for image_path in df["image"].tolist():

        if image_path not in images:
            images.append(image_path)

        if len(images) >= max_images:
            break

    saved = 0

    for idx, image_path in enumerate(images):

        image_path = Path(image_path)

        if not image_path.exists():
            print(
                f"WARNING: imagen no encontrada:\n"
                f"{image_path}"
            )
            continue

        detections = df[df["image"] == str(image_path)]

        rendered = draw_predictions(
            image_path,
            detections,
        )

        if rendered is None:
            continue

        output_file = (
            output_dir
            / f"{name_prefix}_{idx + 1:03d}.jpg"
        )

        cv2.imwrite(
            str(output_file),
            rendered,
        )

        saved += 1

    print(
        f"{output_subdir}: {saved} imágenes"
    )

    return saved


# ============================================================
# PERSONAS CON ALTA CONFIANZA
# ============================================================

def analyze_person_high_confidence(df):

    data = df[
        (df["class_name"] == "person")
        & (df["confidence"] >= HIGH_CONF)
    ].sort_values(
        "confidence",
        ascending=False,
    )

    return save_detection_group(
        data,
        "person/high_confidence",
        "person_high_conf",
    )


# ============================================================
# PERSONAS CON BAJA CONFIANZA
# ============================================================

def analyze_person_low_confidence(df):

    data = df[
        (df["class_name"] == "person")
        & (df["confidence"] >= LOW_CONF_MIN)
        & (df["confidence"] < LOW_CONF_MAX)
    ].sort_values(
        "confidence",
        ascending=True,
    )

    return save_detection_group(
        data,
        "person/low_confidence",
        "person_low_conf",
    )


# ============================================================
# PERSONAS MUY PEQUEÑAS
# ============================================================

def analyze_person_small(df):

    data = df[
        (df["class_name"] == "person")
        & (df["bbox_area"] > 0)
    ].sort_values(
        "bbox_area",
        ascending=True,
    )

    data = data[data["bbox_area"] <= SMALL_AREA]

    return save_detection_group(
        data,
        "person/very_small",
        "person_small",
    )


# ============================================================
# ESCENAS CON MUCHAS PERSONAS
# ============================================================

def analyze_crowded_person_scenes(df):

    persons = df[
        df["class_name"] == "person"
    ]

    counts = (
        persons.groupby("image")
        .size()
        .sort_values(
            ascending=False
        )
    )

    records = []

    for image_path, count in counts.items():

        records.append({
            "image": image_path,
            "person_count": int(count),
        })

    crowded = pd.DataFrame(records)

    if crowded.empty:
        return 0

    top_images = crowded.head(TOP_N)

    selected = df[
        df["image"].isin(
            top_images["image"]
        )
    ]

    saved = save_detection_group(
        selected,
        "person/crowded",
        "person_crowded",
    )

    return saved


# ============================================================
# VEHICLES
# ============================================================

def analyze_vehicle_high_confidence(df):

    data = df[
        (df["class_name"] == "vehicle")
        & (df["confidence"] >= HIGH_CONF)
    ].sort_values(
        "confidence",
        ascending=False,
    )

    return save_detection_group(
        data,
        "vehicle/high_confidence",
        "vehicle_high_conf",
    )


def analyze_vehicle_low_confidence(df):

    data = df[
        (df["class_name"] == "vehicle")
        & (df["confidence"] >= LOW_CONF_MIN)
        & (df["confidence"] < LOW_CONF_MAX)
    ].sort_values(
        "confidence",
        ascending=True,
    )

    return save_detection_group(
        data,
        "vehicle/low_confidence",
        "vehicle_low_conf",
    )


# ============================================================
# ESCENAS CON MAYOR NÚMERO DE DETECCIONES
# ============================================================

def analyze_crowded_scenes(df):

    counts = (
        df.groupby("image")
        .size()
        .sort_values(
            ascending=False
        )
    )

    records = []

    for image_path, count in counts.items():

        records.append({
            "image": image_path,
            "detections": int(count),
        })

    crowded = pd.DataFrame(records)

    if crowded.empty:
        return 0

    top_images = crowded.head(TOP_N)

    selected = df[
        df["image"].isin(
            top_images["image"]
        )
    ]

    return save_detection_group(
        selected,
        "crowded_scenes",
        "crowded_scene",
    )


# ============================================================
# INFORME
# ============================================================

def generate_report(df):

    report_path = (
        OUTPUT_DIR
        / "reports"
        / "visual_analysis_summary.txt"
    )

    person = df[
        df["class_name"] == "person"
    ]

    vehicle = df[
        df["class_name"] == "vehicle"
    ]

    report = []

    report.append(
        "SARC-DRONE - VISUAL ERROR ANALYSIS"
    )
    report.append("=" * 80)
    report.append("")

    report.append(
        f"Predicciones analizadas: {len(df):,}"
    )

    report.append("")

    report.append(
        "PERSON"
    )
    report.append("-" * 80)

    report.append(
        f"Predicciones: {len(person):,}"
    )

    if not person.empty:

        report.append(
            f"Confianza media: "
            f"{person['confidence'].mean():.4f}"
        )

        report.append(
            f"Confianza P10: "
            f"{person['confidence'].quantile(.10):.4f}"
        )

        report.append(
            f"Confianza P50: "
            f"{person['confidence'].quantile(.50):.4f}"
        )

        report.append(
            f"Confianza P90: "
            f"{person['confidence'].quantile(.90):.4f}"
        )

        report.append(
            f"Área bbox media: "
            f"{person['bbox_area'].mean():.2f}"
        )

        report.append(
            f"Área bbox P10: "
            f"{person['bbox_area'].quantile(.10):.2f}"
        )

        report.append(
            f"Área bbox P50: "
            f"{person['bbox_area'].quantile(.50):.2f}"
        )

        report.append(
            f"Área bbox P90: "
            f"{person['bbox_area'].quantile(.90):.2f}"
        )

    report.append("")

    report.append(
        "VEHICLE"
    )
    report.append("-" * 80)

    report.append(
        f"Predicciones: {len(vehicle):,}"
    )

    if not vehicle.empty:

        report.append(
            f"Confianza media: "
            f"{vehicle['confidence'].mean():.4f}"
        )

        report.append(
            f"Confianza P50: "
            f"{vehicle['confidence'].quantile(.50):.4f}"
        )

        report.append(
            f"Área bbox media: "
            f"{vehicle['bbox_area'].mean():.2f}"
        )

    report.append("")

    report.append(
        "CONFIGURACIÓN DEL ANÁLISIS"
    )
    report.append("-" * 80)

    report.append(
        f"TOP_N: {TOP_N}"
    )

    report.append(
        f"Alta confianza: >= {HIGH_CONF}"
    )

    report.append(
        f"Baja confianza: "
        f"{LOW_CONF_MIN} - {LOW_CONF_MAX}"
    )

    report.append(
        f"Área pequeña: <= {SMALL_AREA} px²"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    create_dirs()

    df = load_predictions()

    print()
    print("Generando análisis visual...")
    print()

    analyze_person_high_confidence(df)

    analyze_person_low_confidence(df)

    analyze_person_small(df)

    analyze_crowded_person_scenes(df)

    analyze_vehicle_high_confidence(df)

    analyze_vehicle_low_confidence(df)

    analyze_crowded_scenes(df)

    report = generate_report(df)

    print()
    print("=" * 80)
    print("ANÁLISIS VISUAL COMPLETADO")
    print("=" * 80)
    print()
    print(f"Resultados:")
    print(OUTPUT_DIR)
    print()
    print(f"Informe:")
    print(report)
    print()


if __name__ == "__main__":
    main()