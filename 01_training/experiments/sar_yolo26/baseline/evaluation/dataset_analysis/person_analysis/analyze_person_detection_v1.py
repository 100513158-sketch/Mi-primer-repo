"""
SAR YOLO26 - PERSON DETECTION ANALYSIS V1

Objetivo:
    Analizar específicamente la clase PERSON (class_id=0) del dataset limpio
    para determinar por qué su rendimiento es inferior al de VEHICLE.

El script SOLO diagnostica. No modifica el dataset.

Analiza:
    - distribución de tamaños de PERSON en píxeles
    - áreas normalizadas
    - ancho/alto de bbox
    - aspect ratio
    - personas por imagen
    - densidad/crowding
    - personas cerca del borde
    - bboxes parcialmente fuera
    - imágenes con mayor concentración de personas pequeñas
    - estadísticas por split

Salida:
    .../evaluation/dataset_analysis/person_analysis/person_analysis_v1/
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "ERROR: Pillow no está instalado. Ejecuta: pip install pillow"
    )


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar"
    r"\cleaned\VisDrone_SAR_2CLASS_V1"
)

OUTPUT_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\person_analysis"
    r"\person_analysis_v1"
)

CLASS_PERSON = 0
CLASS_VEHICLE = 1

SPLITS = ("train", "val", "test_dev")

# Umbrales de área en píxeles cuadrados.
AREA_BINS = (
    ("tiny_<16", 0, 16),
    ("very_small_16_32", 16, 32),
    ("small_32_64", 32, 64),
    ("medium_64_256", 64, 256),
    ("large_256_1024", 256, 1024),
    ("very_large_>=1024", 1024, float("inf")),
)

# Proximidad al borde: porcentaje del ancho/alto de la imagen.
BORDER_MARGIN = 0.02


# ============================================================
# UTILIDADES
# ============================================================

def area_bin(area_px: float) -> str:
    for name, low, high in AREA_BINS:
        if low <= area_px < high:
            return name
    return "unknown"


def safe_float(value: str) -> float:
    return float(value.strip())


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def bbox_from_yolo(cx, cy, w, h, img_w, img_h):
    """
    YOLO normalized -> píxeles.
    """
    bw = w * img_w
    bh = h * img_h
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h

    return x1, y1, x2, y2, bw, bh


def is_partial_bbox(x1, y1, x2, y2, img_w, img_h):
    return x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h


def is_near_border(x1, y1, x2, y2, img_w, img_h):
    mx = img_w * BORDER_MARGIN
    my = img_h * BORDER_MARGIN

    return (
        x1 <= mx
        or y1 <= my
        or x2 >= img_w - mx
        or y2 >= img_h - my
    )


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return values[int(k)]

    return values[f] + (values[c] - values[f]) * (k - f)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("# SAR YOLO26 - PERSON DETECTION ANALYSIS V1")
    print("=" * 72)
    print()
    print("Dataset:")
    print(DATASET_ROOT)
    print()
    print("Output:")
    print(OUTPUT_ROOT)
    print()

    if not DATASET_ROOT.exists():
        raise SystemExit(f"ERROR: Dataset no encontrado: {DATASET_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    objects_csv = OUTPUT_ROOT / "person_objects_v1.csv"
    images_csv = OUTPUT_ROOT / "person_images_v1.csv"
    split_csv = OUTPUT_ROOT / "person_split_statistics_v1.csv"
    summary_txt = OUTPUT_ROOT / "PERSON_ANALYSIS_V1_SUMMARY.txt"

    object_rows = []
    image_rows = []

    global_area = []
    global_width = []
    global_height = []
    global_ratio = []
    global_norm_area = []

    split_stats = {}

    total_images = 0
    total_persons = 0

    for split in SPLITS:
        split_root = DATASET_ROOT / split
        image_dir = split_root / "images"
        label_dir = split_root / "labels"

        if not image_dir.exists():
            print(f"[INFO] Split no encontrado: {split}")
            continue

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        print(f"## Analizando: {split}")
        print(f"Imágenes encontradas: {len(images)}")

        split_persons = 0
        split_person_images = 0
        split_areas = []
        split_tiny16 = 0
        split_tiny32 = 0
        split_tiny64 = 0
        split_border = 0
        split_partial = 0
        split_crowded10 = 0
        split_crowded20 = 0
        split_crowded50 = 0
        split_crowded100 = 0

        for idx, image_path in enumerate(images, start=1):
            label_path = label_dir / f"{image_path.stem}.txt"

            try:
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
            except Exception as exc:
                print(f"[WARN] Imagen no legible: {image_path} -> {exc}")
                continue

            persons_in_image = 0
            persons_small16 = 0
            persons_small32 = 0
            persons_small64 = 0
            persons_border = 0
            persons_partial = 0

            if label_path.exists():
                try:
                    lines = label_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                except Exception:
                    lines = []

                for line_no, line in enumerate(lines, start=1):
                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    try:
                        cls = int(parts[0])
                        cx = safe_float(parts[1])
                        cy = safe_float(parts[2])
                        bw_n = safe_float(parts[3])
                        bh_n = safe_float(parts[4])
                    except ValueError:
                        continue

                    if cls != CLASS_PERSON:
                        continue

                    persons_in_image += 1
                    split_persons += 1
                    total_persons += 1

                    x1, y1, x2, y2, bw_px, bh_px = bbox_from_yolo(
                        cx, cy, bw_n, bh_n, img_w, img_h
                    )

                    area_px = max(0.0, bw_px * bh_px)
                    image_area = float(img_w * img_h)
                    norm_area = area_px / image_area if image_area else 0.0
                    aspect = bw_px / bh_px if bh_px > 0 else 0.0

                    partial = is_partial_bbox(
                        x1, y1, x2, y2, img_w, img_h
                    )
                    border = is_near_border(
                        x1, y1, x2, y2, img_w, img_h
                    )

                    if area_px < 16:
                        persons_small16 += 1
                        split_tiny16 += 1

                    if area_px < 32:
                        persons_small32 += 1
                        split_tiny32 += 1

                    if area_px < 64:
                        persons_small64 += 1
                        split_tiny64 += 1

                    if border:
                        persons_border += 1
                        split_border += 1

                    if partial:
                        persons_partial += 1
                        split_partial += 1

                    split_areas.append(area_px)
                    global_area.append(area_px)
                    global_width.append(bw_px)
                    global_height.append(bh_px)
                    global_ratio.append(aspect)
                    global_norm_area.append(norm_area)

                    object_rows.append({
                        "split": split,
                        "image": image_path.name,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "line": line_no,
                        "image_width": img_w,
                        "image_height": img_h,
                        "bbox_width_px": round(bw_px, 4),
                        "bbox_height_px": round(bh_px, 4),
                        "bbox_area_px2": round(area_px, 4),
                        "bbox_area_percent_image": round(norm_area * 100, 6),
                        "aspect_ratio": round(aspect, 4),
                        "area_bin": area_bin(area_px),
                        "near_border": int(border),
                        "partial_bbox": int(partial),
                    })

            if persons_in_image > 0:
                split_person_images += 1

            if persons_in_image >= 10:
                split_crowded10 += 1
            if persons_in_image >= 20:
                split_crowded20 += 1
            if persons_in_image >= 50:
                split_crowded50 += 1
            if persons_in_image >= 100:
                split_crowded100 += 1

            image_rows.append({
                "split": split,
                "image": image_path.name,
                "image_path": str(image_path),
                "image_width": img_w,
                "image_height": img_h,
                "persons": persons_in_image,
                "person_tiny16": persons_small16,
                "person_tiny32": persons_small32,
                "person_tiny64": persons_small64,
                "person_near_border": persons_border,
                "person_partial_bbox": persons_partial,
                "person_density_per_mp": round(
                    persons_in_image / (img_w * img_h / 1_000_000),
                    4
                ) if img_w and img_h else 0,
            })

            total_images += 1

            if idx % 1000 == 0:
                print(f"Procesadas: {idx:,}/{len(images):,}")

        split_stats[split] = {
            "images": len(images),
            "person_images": split_person_images,
            "persons": split_persons,
            "persons_per_image_with_person": (
                split_persons / split_person_images
                if split_person_images else 0
            ),
            "tiny16": split_tiny16,
            "tiny32": split_tiny32,
            "tiny64": split_tiny64,
            "near_border": split_border,
            "partial_bbox": split_partial,
            "crowded10": split_crowded10,
            "crowded20": split_crowded20,
            "crowded50": split_crowded50,
            "crowded100": split_crowded100,
            "area_mean": (
                sum(split_areas) / len(split_areas)
                if split_areas else 0
            ),
            "area_p50": percentile(split_areas, 50),
            "area_p90": percentile(split_areas, 90),
            "area_p95": percentile(split_areas, 95),
        }

        print()

    # ========================================================
    # TOP IMÁGENES PROBLEMÁTICAS
    # ========================================================

    for row in image_rows:
        persons = row["persons"]

        if persons:
            small_ratio = row["person_tiny64"] / persons
            border_ratio = row["person_near_border"] / persons
        else:
            small_ratio = 0
            border_ratio = 0

        # Score diagnóstico, NO es una decisión de limpieza.
        row["difficulty_score"] = round(
            persons
            + row["person_tiny16"] * 4
            + row["person_tiny32"] * 2
            + row["person_tiny64"]
            + row["person_near_border"] * 0.5
            + row["person_partial_bbox"] * 3,
            3,
        )
        row["tiny64_ratio"] = round(small_ratio, 4)
        row["border_ratio"] = round(border_ratio, 4)

    top_images = sorted(
        [r for r in image_rows if r["persons"] > 0],
        key=lambda r: r["difficulty_score"],
        reverse=True,
    )[:100]

    # ========================================================
    # CSV OBJECTOS
    # ========================================================

    if object_rows:
        with objects_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=object_rows[0].keys())
            writer.writeheader()
            writer.writerows(object_rows)

    # ========================================================
    # CSV IMÁGENES
    # ========================================================

    if image_rows:
        with images_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=image_rows[0].keys())
            writer.writeheader()
            writer.writerows(image_rows)

    # ========================================================
    # CSV SPLITS
    # ========================================================

    if split_stats:
        fields = [
            "split", "images", "person_images", "persons",
            "persons_per_image_with_person",
            "tiny16", "tiny32", "tiny64",
            "near_border", "partial_bbox",
            "crowded10", "crowded20", "crowded50", "crowded100",
            "area_mean", "area_p50", "area_p90", "area_p95",
        ]

        with split_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for split, stats in split_stats.items():
                row = {"split": split}
                row.update(stats)
                writer.writerow(row)

    # ========================================================
    # SUMMARY
    # ========================================================

    counts = Counter(area_bin(a) for a in global_area)

    tiny16 = sum(a < 16 for a in global_area)
    tiny32 = sum(a < 32 for a in global_area)
    tiny64 = sum(a < 64 for a in global_area)

    border_total = sum(
        int(r["person_near_border"]) for r in image_rows
    )
    partial_total = sum(
        int(r["person_partial_bbox"]) for r in image_rows
    )

    person_images = sum(
        1 for r in image_rows if r["persons"] > 0
    )

    summary = []
    summary.append("=" * 72)
    summary.append("SAR YOLO26 - PERSON DETECTION ANALYSIS V1")
    summary.append("=" * 72)
    summary.append("")
    summary.append(f"Dataset: {DATASET_ROOT}")
    summary.append(f"Imágenes analizadas: {total_images:,}")
    summary.append(f"Imágenes con PERSON: {person_images:,}")
    summary.append(f"PERSON objects: {total_persons:,}")
    summary.append("")

    summary.append("DISTRIBUCIÓN DE TAMAÑOS PERSON")
    summary.append("-" * 72)

    for name, _, _ in AREA_BINS:
        n = counts[name]
        pct = (100 * n / total_persons) if total_persons else 0
        summary.append(f"{name:<24}: {n:>10,} ({pct:6.2f} %)")

    summary.append("")
    summary.append("PERCENTILES AREA BBOX")
    summary.append("-" * 72)
    summary.append(
        f"Mean : {sum(global_area)/len(global_area):.2f} px²"
        if global_area else "Mean : 0"
    )
    summary.append(f"P50  : {percentile(global_area, 50):.2f} px²")
    summary.append(f"P75  : {percentile(global_area, 75):.2f} px²")
    summary.append(f"P90  : {percentile(global_area, 90):.2f} px²")
    summary.append(f"P95  : {percentile(global_area, 95):.2f} px²")
    summary.append(f"P99  : {percentile(global_area, 99):.2f} px²")
    summary.append("")

    summary.append("OBJETOS PEQUEÑOS")
    summary.append("-" * 72)
    summary.append(
        f"<16 px² : {tiny16:,} "
        f"({100*tiny16/total_persons:.2f} %)"
        if total_persons else "<16 px² : 0"
    )
    summary.append(
        f"<32 px² : {tiny32:,} "
        f"({100*tiny32/total_persons:.2f} %)"
        if total_persons else "<32 px² : 0"
    )
    summary.append(
        f"<64 px² : {tiny64:,} "
        f"({100*tiny64/total_persons:.2f} %)"
        if total_persons else "<64 px² : 0"
    )
    summary.append("")

    summary.append("BORDES")
    summary.append("-" * 72)
    summary.append(
        f"PERSON cerca del borde : {border_total:,} "
        f"({100*border_total/total_persons:.2f} %)"
        if total_persons else "PERSON cerca del borde : 0"
    )
    summary.append(
        f"PERSON bbox parcial     : {partial_total:,} "
        f"({100*partial_total/total_persons:.2f} %)"
        if total_persons else "PERSON bbox parcial : 0"
    )
    summary.append("")

    summary.append("CROWDED POR PERSON")
    summary.append("-" * 72)

    for threshold in (10, 20, 50, 100):
        n = sum(
            1 for r in image_rows
            if r["persons"] >= threshold
        )
        summary.append(
            f">= {threshold:3d} PERSON: {n:,} imágenes"
        )

    summary.append("")
    summary.append("ESTADÍSTICAS POR SPLIT")
    summary.append("-" * 72)

    for split, s in split_stats.items():
        summary.append(
            f"{split}: images={s['images']:,} | "
            f"person_images={s['person_images']:,} | "
            f"persons={s['persons']:,} | "
            f"mean_area={s['area_mean']:.2f} | "
            f"P50={s['area_p50']:.2f} | "
            f"P95={s['area_p95']:.2f}"
        )

    summary.append("")
    summary.append("TOP 20 IMÁGENES DIAGNÓSTICO")
    summary.append("-" * 72)

    for i, row in enumerate(top_images[:20], start=1):
        summary.append(
            f"{i:02d}. {row['split']:<8} "
            f"persons={row['persons']:>3} "
            f"tiny16={row['person_tiny16']:>3} "
            f"tiny32={row['person_tiny32']:>3} "
            f"tiny64={row['person_tiny64']:>3} "
            f"border={row['person_near_border']:>3} "
            f"score={row['difficulty_score']:>8.2f} "
            f"{row['image']}"
        )

    summary.append("")
    summary.append("INTERPRETACIÓN")
    summary.append("-" * 72)
    summary.append(
        "Este análisis NO demuestra por sí solo qué objetos son difíciles "
        "para el modelo, porque todavía no cruza las etiquetas con "
        "predicciones del modelo."
    )
    summary.append(
        "Su objetivo es identificar las características del dataset que "
        "pueden explicar el bajo rendimiento de PERSON."
    )
    summary.append(
        "El siguiente análisis recomendado es cruzar estas características "
        "con las predicciones del baseline YOLO26s para obtener recall/AP "
        "por tamaño, densidad y posición."
    )
    summary.append("")
    summary.append("IMPORTANTE: este script SOLO diagnostica.")
    summary.append("No modifica imágenes ni labels.")

    summary_txt.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("=" * 72)
    print("# RESULTADO PERSON ANALYSIS V1")
    print("=" * 72)
    print()
    print(f"Imágenes:              {total_images:,}")
    print(f"Imágenes con PERSON:   {person_images:,}")
    print(f"Personas:              {total_persons:,}")
    print()
    print("PERSON PEQUEÑAS")
    print(
        f"<16 px²:              {tiny16:,} "
        f"({100*tiny16/total_persons:.2f} %)"
        if total_persons else "<16 px²: 0"
    )
    print(
        f"<32 px²:              {tiny32:,} "
        f"({100*tiny32/total_persons:.2f} %)"
        if total_persons else "<32 px²: 0"
    )
    print(
        f"<64 px²:              {tiny64:,} "
        f"({100*tiny64/total_persons:.2f} %)"
        if total_persons else "<64 px²: 0"
    )
    print()
    print(f"[OK] {objects_csv}")
    print(f"[OK] {images_csv}")
    print(f"[OK] {split_csv}")
    print(f"[OK] {summary_txt}")
    print()
    print("IMPORTANTE: el dataset NO ha sido modificado.")
    print("=" * 72)


if __name__ == "__main__":
    main()
