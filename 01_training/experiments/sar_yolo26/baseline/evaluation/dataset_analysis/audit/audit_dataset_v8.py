#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAR YOLO26 - DATASET AUDIT V8

Objetivo
--------
Auditoría avanzada del dataset antes del entrenamiento.

V8 añade respecto a V7:

- análisis completo por split
- análisis por clase
- análisis de tamaños
- análisis de bordes
- análisis de densidad
- detección de duplicados de anotaciones
- detección de imágenes duplicadas mediante hash
- detección de posibles fugas entre splits
- scoring de calidad por imagen
- clasificación KEEP / REVIEW / EXCLUDE_CANDIDATE / CRITICAL
- recomendación final del dataset

IMPORTANTE
----------
Este script SOLO diagnostica.

NO elimina:
    - imágenes
    - labels
    - carpetas
    - anotaciones

NO modifica el dataset original.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\VisDrone_SAR_2CLASS"
)

WORK_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
)

OUTPUT_ROOT = (
    WORK_ROOT
    / "evaluation"
    / "dataset_analysis"
    / "audit"
    / "audit_dataset_v8"
)

REPORTS_DIR = OUTPUT_ROOT / "reports"
EXAMPLES_DIR = OUTPUT_ROOT / "examples"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SPLITS = [
    "train",
    "val",
    "test",
    "test_dev",
]

CLASS_NAMES = {
    0: "person",
    1: "vehicle",
}

# Umbrales de tamaño
TINY16 = 16.0
TINY32 = 32.0
TINY64 = 64.0
SMALL100 = 100.0

# Umbrales de borde
BORDER_MARGIN = 0.02

# Crowded
CROWDED_100 = 100
CROWDED_200 = 200
CROWDED_300 = 300
CROWDED_500 = 500

# Scoring
REVIEW_SCORE = 15.0
EXCLUDE_SCORE = 50.0
CRITICAL_SCORE = 100.0


# ============================================================
# UTILIDADES
# ============================================================

def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def image_files(split_dir: Path) -> List[Path]:
    images_dir = split_dir / "images"

    if not images_dir.exists():
        return []

    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_from_image(image_path: Path, split_dir: Path) -> Path:
    """
    Convierte:

    split/images/foo/bar.jpg

    en:

    split/labels/foo/bar.txt
    """

    relative = image_path.relative_to(split_dir / "images")

    return split_dir / "labels" / relative.with_suffix(".txt")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def is_valid_number(value: float) -> bool:
    return math.isfinite(value)


# ============================================================
# ANOTACIÓN
# ============================================================

def parse_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> Dict:

    result = {
        "labels": 0,
        "invalid_labels": 0,
        "invalid_coordinates": 0,
        "invalid_bboxes": 0,
        "invalid_classes": 0,
        "duplicate_annotations": 0,
        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "small100": 0,
        "partial_bbox": 0,
        "outside_bbox": 0,
        "border_objects": 0,
        "persons": 0,
        "vehicles": 0,
        "person_tiny16": 0,
        "person_tiny32": 0,
        "person_tiny64": 0,
        "person_partial": 0,
        "person_border": 0,
        "vehicle_tiny16": 0,
        "vehicle_tiny32": 0,
        "vehicle_tiny64": 0,
        "vehicle_partial": 0,
        "vehicle_border": 0,
        "annotation_keys": [],
    }

    if not label_path.exists():
        return result

    try:
        lines = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except Exception:
        result["invalid_labels"] += 1
        return result

    seen = Counter()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            result["invalid_labels"] += 1
            continue

        try:
            cls = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            result["invalid_labels"] += 1
            continue

        result["labels"] += 1

        if cls not in CLASS_NAMES:
            result["invalid_classes"] += 1

        values = [xc, yc, bw, bh]

        if not all(is_valid_number(v) for v in values):
            result["invalid_coordinates"] += 1
            continue

        # Coordenadas YOLO
        if not (
            0.0 <= xc <= 1.0
            and 0.0 <= yc <= 1.0
            and 0.0 <= bw <= 1.0
            and 0.0 <= bh <= 1.0
        ):
            result["invalid_coordinates"] += 1
            continue

        if bw <= 0 or bh <= 0:
            result["invalid_bboxes"] += 1
            continue

        # Bounding box normalizada
        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        # Área en píxeles
        area = bw * image_width * bh * image_height

        if area < TINY16:
            result["tiny16"] += 1

        if area < TINY32:
            result["tiny32"] += 1

        if area < TINY64:
            result["tiny64"] += 1

        if area < SMALL100:
            result["small100"] += 1

        partial = (
            x1 < 0.0
            or y1 < 0.0
            or x2 > 1.0
            or y2 > 1.0
        )

        outside = (
            x2 <= 0.0
            or y2 <= 0.0
            or x1 >= 1.0
            or y1 >= 1.0
        )

        if partial:
            result["partial_bbox"] += 1

        if outside:
            result["outside_bbox"] += 1

        border = (
            x1 <= BORDER_MARGIN
            or y1 <= BORDER_MARGIN
            or x2 >= 1.0 - BORDER_MARGIN
            or y2 >= 1.0 - BORDER_MARGIN
        )

        if border:
            result["border_objects"] += 1

        # Clase
        if cls == 0:
            result["persons"] += 1

            if area < TINY16:
                result["person_tiny16"] += 1

            if area < TINY32:
                result["person_tiny32"] += 1

            if area < TINY64:
                result["person_tiny64"] += 1

            if partial:
                result["person_partial"] += 1

            if border:
                result["person_border"] += 1

        elif cls == 1:
            result["vehicles"] += 1

            if area < TINY16:
                result["vehicle_tiny16"] += 1

            if area < TINY32:
                result["vehicle_tiny32"] += 1

            if area < TINY64:
                result["vehicle_tiny64"] += 1

            if partial:
                result["vehicle_partial"] += 1

            if border:
                result["vehicle_border"] += 1

        # Duplicados exactos
        key = (
            cls,
            round(xc, 6),
            round(yc, 6),
            round(bw, 6),
            round(bh, 6),
        )

        seen[key] += 1

    duplicates = sum(
        count - 1
        for count in seen.values()
        if count > 1
    )

    result["duplicate_annotations"] = duplicates
    result["annotation_keys"] = list(seen.keys())

    return result


# ============================================================
# SCORING
# ============================================================

def calculate_score(data: Dict) -> Tuple[float, List[str], str]:

    score = 0.0
    reasons = []

    objects = data["objects"]

    if data["tiny16"] > 0:
        score += data["tiny16"] * 0.50
        reasons.append(f"tiny16={data['tiny16']}")

    if data["tiny32"] > 0:
        score += data["tiny32"] * 0.10
        reasons.append(f"tiny32={data['tiny32']}")

    if data["partial_bbox"] > 0:
        score += data["partial_bbox"] * 2.0
        reasons.append("partial_bbox")

    if data["outside_bbox"] > 0:
        score += data["outside_bbox"] * 10.0
        reasons.append("outside_bbox")

    if data["border_objects"] > 0:
        score += data["border_objects"] * 0.25
        reasons.append("border_objects")

    if objects >= CROWDED_100:
        score += 2.0
        reasons.append("crowded100")

    if objects >= CROWDED_200:
        score += 4.0
        reasons.append("crowded200")

    if objects >= CROWDED_300:
        score += 8.0
        reasons.append("crowded300")

    if objects >= CROWDED_500:
        score += 15.0
        reasons.append("crowded500")

    if data["duplicate_annotations"] > 0:
        score += data["duplicate_annotations"] * 5.0
        reasons.append(
            f"duplicate_annotations={data['duplicate_annotations']}"
        )

    if data["invalid_labels"] > 0:
        score += data["invalid_labels"] * 20.0
        reasons.append("invalid_labels")

    if data["invalid_coordinates"] > 0:
        score += data["invalid_coordinates"] * 20.0
        reasons.append("invalid_coordinates")

    if data["invalid_bboxes"] > 0:
        score += data["invalid_bboxes"] * 20.0
        reasons.append("invalid_bboxes")

    if data["invalid_classes"] > 0:
        score += data["invalid_classes"] * 20.0
        reasons.append("invalid_classes")

    if data["corrupt"]:
        score += 1000.0
        reasons.append("corrupt_image")

    # Decisión
    if data["corrupt"]:
        decision = "CRITICAL"

    elif score >= CRITICAL_SCORE:
        decision = "CRITICAL"

    elif score >= EXCLUDE_SCORE:
        decision = "EXCLUDE_CANDIDATE"

    elif score >= REVIEW_SCORE:
        decision = "REVIEW"

    else:
        decision = "KEEP"

    return score, reasons, decision


# ============================================================
# AUDITORÍA PRINCIPAL
# ============================================================

def analyze_split(split: str) -> List[Dict]:

    split_dir = DATASET_ROOT / split

    if not split_dir.exists():
        print(f"[INFO] Split no encontrado: {split}")
        return []

    images = image_files(split_dir)

    print()
    print(f"## Analizando: {split}")
    print()
    print(f"Imágenes encontradas: {len(images)}")

    rows = []

    for index, image_path in enumerate(images, start=1):

        corrupt = False

        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Verificación básica
                img.verify()

        except Exception:
            corrupt = True
            width = 0
            height = 0

        label_path = label_path_from_image(
            image_path,
            split_dir,
        )

        annotation = parse_label_file(
            label_path,
            width,
            height,
        )

        objects = annotation["labels"]

        row = {
            "split": split,
            "image": str(image_path),
            "label": str(label_path),
            "width": width,
            "height": height,
            "objects": objects,
            "persons": annotation["persons"],
            "vehicles": annotation["vehicles"],
            "tiny16": annotation["tiny16"],
            "tiny32": annotation["tiny32"],
            "tiny64": annotation["tiny64"],
            "small100": annotation["small100"],
            "partial_bbox": annotation["partial_bbox"],
            "outside_bbox": annotation["outside_bbox"],
            "border_objects": annotation["border_objects"],
            "invalid_labels": annotation["invalid_labels"],
            "invalid_coordinates": annotation["invalid_coordinates"],
            "invalid_bboxes": annotation["invalid_bboxes"],
            "invalid_classes": annotation["invalid_classes"],
            "duplicate_annotations": annotation["duplicate_annotations"],
            "corrupt": corrupt,
            "image_hash": "",
        }

        score, reasons, decision = calculate_score(row)

        row["score"] = round(score, 2)
        row["decision"] = decision
        row["reasons"] = ";".join(reasons)

        rows.append(row)

        if index % 1000 == 0:
            print(f"Procesadas: {index:,}/{len(images):,}")

    return rows


# ============================================================
# HASH DE IMÁGENES
# ============================================================

def calculate_image_hashes(rows: List[Dict]) -> None:

    print()
    print("Calculando hashes SHA256...")

    for index, row in enumerate(rows, start=1):

        path = Path(row["image"])

        try:
            row["image_hash"] = sha256_file(path)
        except Exception:
            row["image_hash"] = ""

        if index % 1000 == 0:
            print(f"Hashes: {index:,}/{len(rows):,}")


def detect_duplicate_images(rows: List[Dict]) -> List[Dict]:

    groups = defaultdict(list)

    for row in rows:

        h = row["image_hash"]

        if h:
            groups[h].append(row)

    duplicates = []

    for h, group in groups.items():

        if len(group) > 1:

            splits = sorted(
                set(r["split"] for r in group)
            )

            for row in group:

                duplicates.append({
                    "hash": h,
                    "split": row["split"],
                    "image": row["image"],
                    "duplicate_count": len(group),
                    "splits": "|".join(splits),
                    "cross_split": len(splits) > 1,
                })

    return duplicates


# ============================================================
# RESUMEN
# ============================================================

def summarize(rows: List[Dict]) -> Dict:

    summary = {
        "images": len(rows),
        "persons": 0,
        "vehicles": 0,
        "objects": 0,
        "tiny16": 0,
        "tiny32": 0,
        "tiny64": 0,
        "small100": 0,
        "partial_bbox": 0,
        "outside_bbox": 0,
        "border_objects": 0,
        "invalid_labels": 0,
        "invalid_coordinates": 0,
        "invalid_bboxes": 0,
        "invalid_classes": 0,
        "duplicates": 0,
        "corrupt": 0,
        "decisions": Counter(),
    }

    for row in rows:

        summary["persons"] += row["persons"]
        summary["vehicles"] += row["vehicles"]
        summary["objects"] += row["objects"]

        summary["tiny16"] += row["tiny16"]
        summary["tiny32"] += row["tiny32"]
        summary["tiny64"] += row["tiny64"]
        summary["small100"] += row["small100"]

        summary["partial_bbox"] += row["partial_bbox"]
        summary["outside_bbox"] += row["outside_bbox"]
        summary["border_objects"] += row["border_objects"]

        summary["invalid_labels"] += row["invalid_labels"]
        summary["invalid_coordinates"] += row["invalid_coordinates"]
        summary["invalid_bboxes"] += row["invalid_bboxes"]
        summary["invalid_classes"] += row["invalid_classes"]

        summary["duplicates"] += row["duplicate_annotations"]

        if row["corrupt"]:
            summary["corrupt"] += 1

        summary["decisions"][row["decision"]] += 1

    return summary


# ============================================================
# CROWDED
# ============================================================

def crowded_counts(rows: List[Dict]) -> Dict:

    thresholds = [
        CROWDED_100,
        CROWDED_200,
        CROWDED_300,
        CROWDED_500,
    ]

    result = {}

    for threshold in thresholds:

        result[threshold] = sum(
            1
            for row in rows
            if row["objects"] >= threshold
        )

    return result


# ============================================================
# CSV
# ============================================================

def write_csv(
    filename: str,
    rows: List[Dict],
) -> None:

    path = REPORTS_DIR / filename

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RESUMEN POR SPLIT
# ============================================================

def write_split_summary(rows: List[Dict]) -> None:

    output = []

    for split in sorted(set(r["split"] for r in rows)):

        split_rows = [
            r for r in rows
            if r["split"] == split
        ]

        s = summarize(split_rows)

        output.append({
            "split": split,
            "images": s["images"],
            "persons": s["persons"],
            "vehicles": s["vehicles"],
            "objects": s["objects"],
            "objects_per_image": round(
                s["objects"] / s["images"],
                4
            ) if s["images"] else 0,
            "tiny16": s["tiny16"],
            "tiny32": s["tiny32"],
            "tiny64": s["tiny64"],
            "partial_bbox": s["partial_bbox"],
            "border_objects": s["border_objects"],
            "duplicates": s["duplicates"],
            "corrupt": s["corrupt"],
            "KEEP": s["decisions"]["KEEP"],
            "REVIEW": s["decisions"]["REVIEW"],
            "EXCLUDE_CANDIDATE": s["decisions"]["EXCLUDE_CANDIDATE"],
            "CRITICAL": s["decisions"]["CRITICAL"],
        })

    write_csv(
        "split_summary.csv",
        output,
    )


# ============================================================
# DISTRIBUCIÓN DE DECISIONES
# ============================================================

def write_decision_csv(rows: List[Dict]) -> None:

    decisions = Counter(
        row["decision"]
        for row in rows
    )

    total = len(rows)

    output = []

    for decision in [
        "KEEP",
        "REVIEW",
        "EXCLUDE_CANDIDATE",
        "CRITICAL",
    ]:

        count = decisions[decision]

        output.append({
            "decision": decision,
            "images": count,
            "percentage": round(
                100.0 * count / total,
                4
            ) if total else 0,
        })

    write_csv(
        "decision_summary.csv",
        output,
    )


# ============================================================
# TOP PROBLEMÁTICAS
# ============================================================

def write_top_review(rows: List[Dict]) -> None:

    ordered = sorted(
        rows,
        key=lambda r: r["score"],
        reverse=True,
    )

    top = ordered[:100]

    write_csv(
        "top_100_review.csv",
        top,
    )


# ============================================================
# INFORME FINAL
# ============================================================

def write_report(
    rows: List[Dict],
    duplicate_images: List[Dict],
) -> None:

    summary = summarize(rows)
    crowded = crowded_counts(rows)

    total = summary["images"]
    objects = summary["objects"]

    decisions = summary["decisions"]

    report_path = (
        REPORTS_DIR
        / "AUDIT_V8_RECOMMENDATION.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAR YOLO26 - DATASET AUDIT V8\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            "DATASET\n"
        )

        f.write(
            f"{DATASET_ROOT}\n\n"
        )

        f.write(
            "RESUMEN GENERAL\n"
        )

        f.write(
            f"Imágenes: {total:,}\n"
        )

        f.write(
            f"Personas: {summary['persons']:,}\n"
        )

        f.write(
            f"Vehículos: {summary['vehicles']:,}\n"
        )

        f.write(
            f"Objetos: {objects:,}\n"
        )

        if total:
            f.write(
                f"Objetos/imagen: "
                f"{objects / total:.2f}\n"
            )

        f.write("\n")

        f.write(
            "DECISIONES\n"
        )

        for decision in [
            "KEEP",
            "REVIEW",
            "EXCLUDE_CANDIDATE",
            "CRITICAL",
        ]:

            count = decisions[decision]

            pct = (
                100.0 * count / total
                if total
                else 0
            )

            f.write(
                f"{decision:22}: "
                f"{count:8,} "
                f"({pct:6.2f} %)\n"
            )

        f.write("\n")

        f.write(
            "OBJETOS PEQUEÑOS\n"
        )

        f.write(
            f"<16 px² : {summary['tiny16']:,}\n"
        )

        f.write(
            f"<32 px² : {summary['tiny32']:,}\n"
        )

        f.write(
            f"<64 px² : {summary['tiny64']:,}\n"
        )

        f.write(
            f"<100 px²: {summary['small100']:,}\n"
        )

        f.write("\n")

        f.write(
            "BORDES\n"
        )

        f.write(
            f"BBox parcialmente fuera: "
            f"{summary['partial_bbox']:,}\n"
        )

        f.write(
            f"BBox completamente fuera: "
            f"{summary['outside_bbox']:,}\n"
        )

        f.write(
            f"Cerca del borde: "
            f"{summary['border_objects']:,}\n"
        )

        f.write("\n")

        f.write(
            "INTEGRIDAD\n"
        )

        f.write(
            f"Labels inválidos: "
            f"{summary['invalid_labels']:,}\n"
        )

        f.write(
            f"Coordenadas inválidas: "
            f"{summary['invalid_coordinates']:,}\n"
        )

        f.write(
            f"BBoxes inválidas: "
            f"{summary['invalid_bboxes']:,}\n"
        )

        f.write(
            f"Clases inválidas: "
            f"{summary['invalid_classes']:,}\n"
        )

        f.write(
            f"Duplicados de anotación: "
            f"{summary['duplicates']:,}\n"
        )

        f.write(
            f"Imágenes corruptas: "
            f"{summary['corrupt']:,}\n"
        )

        f.write("\n")

        f.write(
            "CROWDED\n"
        )

        for threshold in [
            100,
            200,
            300,
            500,
        ]:

            f.write(
                f">= {threshold:3} objetos: "
                f"{crowded[threshold]:,} imágenes\n"
            )

        f.write("\n")

        f.write(
            "DUPLICADOS DE IMAGEN\n"
        )

        f.write(
            f"Grupos detectados: "
            f"{len(duplicate_images):,}\n"
        )

        cross_split = [
            d for d in duplicate_images
            if d["cross_split"]
        ]

        f.write(
            f"Posibles fugas entre splits: "
            f"{len(cross_split):,}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # RECOMENDACIÓN
        # ----------------------------------------------------

        f.write(
            "=" * 72 + "\n"
        )

        f.write(
            "RECOMENDACIÓN\n"
        )

        f.write(
            "=" * 72 + "\n\n"
        )

        critical = decisions["CRITICAL"]
        exclude = decisions["EXCLUDE_CANDIDATE"]
        review = decisions["REVIEW"]

        if critical > 0:

            f.write(
                "NO ENTRENAR TODAVÍA.\n\n"
                "Se han detectado imágenes CRITICAL. "
                "Deben investigarse antes del entrenamiento.\n"
            )

        elif cross_split:

            f.write(
                "REVISAR SPLITS ANTES DE ENTRENAR.\n\n"
                "Se han detectado posibles imágenes "
                "duplicadas entre train/val/test_dev.\n"
                "Esto puede provocar data leakage.\n"
            )

        elif exclude > 0:

            f.write(
                "SE RECOMIENDA UNA REVISIÓN MANUAL.\n\n"
            )

            f.write(
                f"Hay {exclude:,} imágenes "
                "EXCLUDE_CANDIDATE.\n"
            )

            f.write(
                "No deben eliminarse automáticamente. "
                "Primero deben visualizarse.\n"
            )

        else:

            f.write(
                "EL DATASET ES APTO PARA CONTINUAR "
                "CON LAS PRUEBAS DE ENTRENAMIENTO.\n\n"
            )

        f.write("\n")

        f.write(
            "IMPORTANTE\n"
        )

        f.write(
            "Este informe NO significa que las imágenes "
            "EXCLUDE_CANDIDATE deban eliminarse.\n"
        )

        f.write(
            "Una imagen crowded o con objetos tiny puede "
            "ser perfectamente válida para un detector SAR.\n"
        )

        f.write(
            "La decisión final debe basarse en inspección "
            "visual y en los resultados de entrenamiento.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("# SAR YOLO26 - DATASET AUDIT V8")
    print()
    print("Dataset:")
    print(DATASET_ROOT)
    print()
    print("Output:")
    print(OUTPUT_ROOT)
    print()

    if not DATASET_ROOT.exists():

        print(
            "[ERROR] No existe DATASET_ROOT:"
        )

        print(DATASET_ROOT)

        return

    ensure_dirs()

    all_rows = []

    # --------------------------------------------------------
    # ANALIZAR SPLITS
    # --------------------------------------------------------

    for split in SPLITS:

        rows = analyze_split(split)

        all_rows.extend(rows)

    if not all_rows:

        print(
            "[ERROR] No se encontraron imágenes."
        )

        return

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    calculate_image_hashes(
        all_rows
    )

    # --------------------------------------------------------
    # DUPLICADOS
    # --------------------------------------------------------

    duplicate_images = detect_duplicate_images(
        all_rows
    )

    # --------------------------------------------------------
    # REPORTES
    # --------------------------------------------------------

    write_csv(
        "image_audit.csv",
        all_rows,
    )

    write_split_summary(
        all_rows
    )

    write_decision_csv(
        all_rows
    )

    write_top_review(
        all_rows
    )

    write_csv(
        "duplicate_images.csv",
        duplicate_images,
    )

    # --------------------------------------------------------
    # OBJETOS
    # --------------------------------------------------------

    object_rows = []

    for row in all_rows:

        object_rows.append({
            "split": row["split"],
            "image": row["image"],
            "objects": row["objects"],
            "persons": row["persons"],
            "vehicles": row["vehicles"],
            "tiny16": row["tiny16"],
            "tiny32": row["tiny32"],
            "tiny64": row["tiny64"],
            "partial_bbox": row["partial_bbox"],
            "outside_bbox": row["outside_bbox"],
            "border_objects": row["border_objects"],
            "score": row["score"],
            "decision": row["decision"],
            "reasons": row["reasons"],
        })

    write_csv(
        "object_distribution.csv",
        object_rows,
    )

    # --------------------------------------------------------
    # INFORME
    # --------------------------------------------------------

    write_report(
        all_rows,
        duplicate_images,
    )

    # --------------------------------------------------------
    # CONSOLA
    # --------------------------------------------------------

    summary = summarize(
        all_rows
    )

    crowded = crowded_counts(
        all_rows
    )

    total = summary["images"]

    print()
    print(
        f"Imágenes:              "
        f"{summary['images']:,}"
    )

    print(
        f"Personas:              "
        f"{summary['persons']:,}"
    )

    print(
        f"Vehículos:             "
        f"{summary['vehicles']:,}"
    )

    print(
        f"Objetos:               "
        f"{summary['objects']:,}"
    )

    if total:

        print(
            f"Objetos/imagen:        "
            f"{summary['objects'] / total:.2f}"
        )

    print()

    print("DECISIONES")

    for decision in [
        "KEEP",
        "REVIEW",
        "EXCLUDE_CANDIDATE",
        "CRITICAL",
    ]:

        count = summary["decisions"][decision]

        pct = (
            100.0 * count / total
            if total
            else 0
        )

        print(
            f"{decision:22}: "
            f"{count:8,} "
            f"({pct:6.2f} %)"
        )

    print()

    print("OBJETOS PEQUEÑOS")

    print(
        f"<16 px²:              "
        f"{summary['tiny16']:,}"
    )

    print(
        f"<32 px²:              "
        f"{summary['tiny32']:,}"
    )

    print(
        f"<64 px²:              "
        f"{summary['tiny64']:,}"
    )

    print()

    print("BORDES")

    print(
        f"BBox parcialmente fuera: "
        f"{summary['partial_bbox']:,}"
    )

    print(
        f"BBox completamente fuera: "
        f"{summary['outside_bbox']:,}"
    )

    print(
        f"Cerca del borde:          "
        f"{summary['border_objects']:,}"
    )

    print()

    print("INTEGRIDAD")

    print(
        f"Labels inválidos:      "
        f"{summary['invalid_labels']:,}"
    )

    print(
        f"Coordenadas inválidas: "
        f"{summary['invalid_coordinates']:,}"
    )

    print(
        f"BBoxes inválidas:      "
        f"{summary['invalid_bboxes']:,}"
    )

    print(
        f"Clases inválidas:      "
        f"{summary['invalid_classes']:,}"
    )

    print(
        f"Duplicados:            "
        f"{summary['duplicates']:,}"
    )

    print()

    print("CROWDED")

    for threshold in [
        100,
        200,
        300,
        500,
    ]:

        print(
            f">= {threshold:3} objetos: "
            f"{crowded[threshold]:,} imágenes"
        )

    print()

    print("DUPLICADOS DE IMAGEN")

    print(
        f"Grupos duplicados:     "
        f"{len(duplicate_images):,}"
    )

    cross_split = [
        d
        for d in duplicate_images
        if d["cross_split"]
    ]

    print(
        f"Posibles data leakage: "
        f"{len(cross_split):,}"
    )

    print()

    print("Reports:")
    print(REPORTS_DIR)

    print()

    print(
        "Informe:"
    )

    print(
        REPORTS_DIR
        / "AUDIT_V8_RECOMMENDATION.txt"
    )

    print()

    print(
        "IMPORTANTE: este script SOLO diagnostica."
    )

    print(
        "No elimina ni modifica imágenes o labels."
    )

    print()


if __name__ == "__main__":
    main()