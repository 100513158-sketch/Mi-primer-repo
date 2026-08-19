from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path


# ============================================================
# SAR YOLO26 - DATASET CLEANER V1
# ============================================================
#
# OBJETIVO
# --------
# Crear una versión limpia/candidata del dataset a partir
# del dataset original.
#
# IMPORTANTE:
#   - NO modifica el dataset original.
#   - NO elimina imágenes.
#   - NO elimina labels.
#   - NO modifica annotations.
#   - Solo COPIA los elementos clasificados como KEEP.
#   - REVIEW y EXCLUDE_CANDIDATE quedan registrados.
#
# ============================================================


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

DATASET_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\VisDrone_SAR_2CLASS"
)

CLEANED_ROOT = Path(
    r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1"
)

# Auditoría V11
AUDIT_V11_ROOT = Path(
    r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline"
    r"\evaluation\dataset_analysis\audit\audit_dataset_v11"
)

AUDIT_V11_REPORTS = AUDIT_V11_ROOT / "reports"


# Splits que vamos a conservar
SPLITS = [
    "train",
    "val",
    "test_dev",
]


# Extensiones de imagen admitidas
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# UTILIDADES
# ============================================================

def print_header(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    """
    Busca la imagen correspondiente a un label.

    Primero intenta las extensiones habituales.
    Si no la encuentra, hace una búsqueda por stem.
    """

    stem = label_path.stem

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"

        if candidate.exists():
            return candidate

    # Fallback
    matches = [
        p
        for p in images_dir.glob(f"{stem}.*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if matches:
        return matches[0]

    return None


def get_label_for_image(image_path: Path, labels_dir: Path) -> Path | None:
    """
    Busca el label correspondiente a una imagen.
    """

    label_path = labels_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        return label_path

    return None


def copy_file(src: Path, dst: Path) -> None:
    """
    Copia conservando metadata básica.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)


# ============================================================
# DETECCIÓN DE CSV DE V11
# ============================================================

def find_decision_csv() -> Path | None:
    """
    Busca automáticamente el CSV de decisiones generado
    por audit_dataset_v11.py.

    Se prueban varios nombres para hacer el cleaner robusto
    frente a pequeñas diferencias entre versiones.
    """

    candidates = [
        AUDIT_V11_REPORTS / "image_audit.csv",
        AUDIT_V11_REPORTS / "IMAGE_AUDIT.csv",
        AUDIT_V11_REPORTS / "audit_images.csv",
        AUDIT_V11_REPORTS / "dataset_audit.csv",
        AUDIT_V11_REPORTS / "AUDIT_V11.csv",
        AUDIT_V11_ROOT / "image_audit.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Búsqueda genérica
    if AUDIT_V11_REPORTS.exists():
        csv_files = list(AUDIT_V11_REPORTS.glob("*.csv"))

        for csv_file in csv_files:
            name = csv_file.name.lower()

            if (
                "image" in name
                or "audit" in name
                or "decision" in name
            ):
                return csv_file

    return None


# ============================================================
# LECTURA DE DECISIONES
# ============================================================

def normalize_decision(value: str) -> str:
    """
    Normaliza la decisión del audit.
    """

    value = value.strip().upper()

    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    return value


def normalize_path(value: str) -> Path:
    """
    Convierte una ruta almacenada en CSV en Path.
    """

    return Path(value.strip().strip('"'))


def find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """
    Busca una columna de forma tolerante.
    """

    normalized = {
        field.lower().strip(): field
        for field in fieldnames
    }

    for candidate in candidates:
        key = candidate.lower().strip()

        if key in normalized:
            return normalized[key]

    return None


def load_decisions(csv_path: Path) -> dict[str, dict]:
    """
    Lee el CSV de V11 y devuelve:

        {
            ruta_imagen: {
                "decision": "...",
                ...
            }
        }

    La ruta absoluta es la clave principal.
    """

    decisions = {}

    print()
    print(f"CSV de auditoría:")
    print(csv_path)

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(
                "El CSV no contiene cabecera."
            )

        fieldnames = list(reader.fieldnames)

        path_column = find_column(
            fieldnames,
            [
                "image_path",
                "image",
                "path",
                "filepath",
                "file",
                "image_file",
            ],
        )

        decision_column = find_column(
            fieldnames,
            [
                "decision",
                "status",
                "classification",
                "category",
                "risk",
            ],
        )

        if path_column is None:
            raise RuntimeError(
                "No se encontró una columna de ruta de imagen "
                f"en {csv_path}.\n"
                f"Columnas encontradas: {fieldnames}"
            )

        if decision_column is None:
            raise RuntimeError(
                "No se encontró una columna de decisión "
                f"en {csv_path}.\n"
                f"Columnas encontradas: {fieldnames}"
            )

        for row in reader:

            image_value = row.get(path_column, "").strip()
            decision_value = row.get(decision_column, "").strip()

            if not image_value:
                continue

            image_path = normalize_path(image_value)

            decisions[str(image_path.resolve()).lower()] = {
                "decision": normalize_decision(decision_value),
                "row": row,
            }

    return decisions


# ============================================================
# RECORRIDO DIRECTO DEL DATASET
# ============================================================

def collect_dataset_images() -> list[tuple[str, Path, Path | None]]:
    """
    Recorre el dataset original directamente.

    Devuelve:

        [
            (
                split,
                image_path,
                label_path
            )
        ]
    """

    items = []

    for split in SPLITS:

        images_dir = DATASET_ROOT / split / "images"
        labels_dir = DATASET_ROOT / split / "labels"

        print()
        print(f"## Analizando split: {split}")

        if not images_dir.exists():
            print(
                f"[WARNING] No existe: {images_dir}"
            )
            continue

        images = sorted(
            p
            for p in images_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(
            f"Imágenes encontradas: {len(images):,}"
        )

        for image_path in images:

            label_path = get_label_for_image(
                image_path,
                labels_dir,
            )

            items.append(
                (
                    split,
                    image_path,
                    label_path,
                )
            )

    return items


# ============================================================
# CREACIÓN DE ESTRUCTURA
# ============================================================

def create_output_structure() -> None:

    print()
    print("Creando estructura de salida...")

    for split in SPLITS:

        ensure_directory(
            CLEANED_ROOT / split / "images"
        )

        ensure_directory(
            CLEANED_ROOT / split / "labels"
        )

    ensure_directory(
        CLEANED_ROOT / "audit"
    )

    print(
        f"[OK] {CLEANED_ROOT}"
    )


# ============================================================
# CSV DE AUDITORÍA
# ============================================================

def write_decision_csv(
    filename: str,
    rows: list[dict],
) -> None:

    output = CLEANED_ROOT / "audit" / filename

    if not rows:

        # Crear igualmente el fichero con cabecera.
        fieldnames = [
            "split",
            "image_path",
            "label_path",
            "decision",
            "exists_image",
            "exists_label",
        ]

    else:

        base_fields = [
            "split",
            "image_path",
            "label_path",
            "decision",
            "exists_image",
            "exists_label",
        ]

        extra_fields = []

        for row in rows:

            for key in row:

                if key not in base_fields and key not in extra_fields:
                    extra_fields.append(key)

        fieldnames = base_fields + extra_fields

    with output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"[OK] {output}"
    )


# ============================================================
# RESUMEN
# ============================================================

def write_summary(
    total_images: int,
    keep_images: int,
    review_images: int,
    exclude_images: int,
    critical_images: int,
    copied_images: int,
    missing_labels: int,
    missing_audit_decisions: int,
    duplicate_destinations: int,
) -> None:

    summary_path = (
        CLEANED_ROOT
        / "audit"
        / "CLEANING_SUMMARY.txt"
    )

    keep_pct = (
        keep_images / total_images * 100
        if total_images
        else 0
    )

    review_pct = (
        review_images / total_images * 100
        if total_images
        else 0
    )

    exclude_pct = (
        exclude_images / total_images * 100
        if total_images
        else 0
    )

    critical_pct = (
        critical_images / total_images * 100
        if total_images
        else 0
    )

    lines = [

        "SAR YOLO26 - DATASET CLEANER V1",
        "",
        "=" * 70,
        "",
        "DATASET ORIGINAL",
        str(DATASET_ROOT),
        "",
        "DATASET GENERADO",
        str(CLEANED_ROOT),
        "",
        "=" * 70,
        "",
        "RESUMEN",
        "",
        f"Imágenes analizadas:        {total_images:,}",
        f"KEEP:                        {keep_images:,} ({keep_pct:.2f} %)",
        f"REVIEW:                      {review_images:,} ({review_pct:.2f} %)",
        f"EXCLUDE_CANDIDATE:           {exclude_images:,} ({exclude_pct:.2f} %)",
        f"CRITICAL:                    {critical_images:,} ({critical_pct:.2f} %)",
        "",
        f"Imágenes copiadas:           {copied_images:,}",
        f"Labels inexistentes:         {missing_labels:,}",
        f"Sin decisión en auditoría:   {missing_audit_decisions:,}",
        f"Colisiones destino:          {duplicate_destinations:,}",
        "",
        "=" * 70,
        "",
        "REGLA V1",
        "",
        "Solo las imágenes clasificadas como KEEP se copian",
        "al dataset limpio.",
        "",
        "REVIEW y EXCLUDE_CANDIDATE NO se eliminan.",
        "Quedan registrados en los CSV de auditoría.",
        "",
        "El dataset original NO ha sido modificado.",
        "",
        "=" * 70,
    ]

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[OK] {summary_path}"
    )


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main() -> None:

    print_header(
        "SAR YOLO26 - DATASET CLEANER V1"
    )

    print()
    print("Dataset original:")
    print(DATASET_ROOT)

    print()
    print("Dataset limpio:")
    print(CLEANED_ROOT)

    print()
    print("Auditoría:")
    print(AUDIT_V11_ROOT)

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise SystemExit(
            "\n[ERROR] No existe DATASET_ROOT:\n"
            f"{DATASET_ROOT}"
        )

    if not AUDIT_V11_ROOT.exists():

        raise SystemExit(
            "\n[ERROR] No existe AUDIT_V11_ROOT:\n"
            f"{AUDIT_V11_ROOT}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = find_decision_csv()

    if csv_path is None:

        raise SystemExit(
            "\n[ERROR] No se encontró el CSV de decisiones "
            "de audit_dataset_v11.py.\n\n"
            f"Revisa:\n{AUDIT_V11_REPORTS}"
        )

    decisions = load_decisions(csv_path)

    print(
        f"Decisiones cargadas: {len(decisions):,}"
    )

    # --------------------------------------------------------
    # ESTRUCTURA
    # --------------------------------------------------------

    create_output_structure()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset_items = collect_dataset_images()

    total_images = len(dataset_items)

    print()
    print(
        f"Total imágenes encontradas: {total_images:,}"
    )

    # --------------------------------------------------------
    # CONTADORES
    # --------------------------------------------------------

    counters = Counter()

    keep_rows = []
    review_rows = []
    exclude_rows = []

    copied_images = 0
    missing_labels = 0
    missing_audit_decisions = 0
    duplicate_destinations = 0

    processed = 0

    destinations = set()

    # --------------------------------------------------------
    # PROCESAMIENTO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESANDO DECISIONES")
    print("=" * 70)

    for split, image_path, label_path in dataset_items:

        processed += 1

        if processed % 1000 == 0:

            print(
                f"Procesadas: "
                f"{processed:,}/{total_images:,}"
            )

        key = str(
            image_path.resolve()
        ).lower()

        audit_entry = decisions.get(key)

        if audit_entry is None:

            # ------------------------------------------------
            # Fallback:
            # intentar comparar por nombre de archivo.
            # ------------------------------------------------

            matches = [
                entry
                for entry_key, entry in decisions.items()
                if Path(entry_key).name.lower()
                == image_path.name.lower()
            ]

            if len(matches) == 1:
                audit_entry = matches[0]

        if audit_entry is None:

            decision = "UNKNOWN"

            missing_audit_decisions += 1

        else:

            decision = normalize_decision(
                audit_entry["decision"]
            )

        # ----------------------------------------------------
        # RUTA DESTINO
        # ----------------------------------------------------

        relative_image = image_path.relative_to(
            DATASET_ROOT / split / "images"
        )

        destination_image = (
            CLEANED_ROOT
            / split
            / "images"
            / relative_image
        )

        destination_label = (
            CLEANED_ROOT
            / split
            / "labels"
            / f"{image_path.stem}.txt"
        )

        destination_key = str(
            destination_image.resolve()
        ).lower()

        if destination_key in destinations:

            duplicate_destinations += 1

        destinations.add(destination_key)

        # ----------------------------------------------------
        # INFORMACIÓN CSV
        # ----------------------------------------------------

        row = {
            "split": split,
            "image_path": str(image_path),
            "label_path": (
                str(label_path)
                if label_path
                else ""
            ),
            "decision": decision,
            "exists_image": image_path.exists(),
            "exists_label": (
                label_path.exists()
                if label_path
                else False
            ),
        }

        # ----------------------------------------------------
        # KEEP
        # ----------------------------------------------------

        if decision == "KEEP":

            counters["KEEP"] += 1

            if label_path is None:

                missing_labels += 1

                row["decision"] = (
                    "KEEP_MISSING_LABEL"
                )

                keep_rows.append(row)

                continue

            # -----------------------------------------------
            # Copiar imagen
            # -----------------------------------------------

            copy_file(
                image_path,
                destination_image,
            )

            # -----------------------------------------------
            # Copiar label
            # -----------------------------------------------

            copy_file(
                label_path,
                destination_label,
            )

            copied_images += 1

            keep_rows.append(row)

        # ----------------------------------------------------
        # REVIEW
        # ----------------------------------------------------

        elif decision == "REVIEW":

            counters["REVIEW"] += 1

            review_rows.append(row)

        # ----------------------------------------------------
        # EXCLUDE CANDIDATE
        # ----------------------------------------------------

        elif decision == "EXCLUDE_CANDIDATE":

            counters["EXCLUDE_CANDIDATE"] += 1

            exclude_rows.append(row)

        # ----------------------------------------------------
        # CRITICAL
        # ----------------------------------------------------

        elif decision == "CRITICAL":

            counters["CRITICAL"] += 1

            exclude_rows.append(row)

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        else:

            counters["UNKNOWN"] += 1

            review_rows.append(row)

    # ========================================================
    # CSV
    # ========================================================

    write_decision_csv(
        "KEEP.csv",
        keep_rows,
    )

    write_decision_csv(
        "REVIEW.csv",
        review_rows,
    )

    write_decision_csv(
        "EXCLUDE_CANDIDATE.csv",
        exclude_rows,
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    write_summary(
        total_images=total_images,
        keep_images=counters["KEEP"],
        review_images=counters["REVIEW"],
        exclude_images=counters["EXCLUDE_CANDIDATE"],
        critical_images=counters["CRITICAL"],
        copied_images=copied_images,
        missing_labels=missing_labels,
        missing_audit_decisions=missing_audit_decisions,
        duplicate_destinations=duplicate_destinations,
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    print_header(
        "RESULTADO CLEANER V1"
    )

    print(
        f"KEEP:              {counters['KEEP']:,}"
    )

    print(
        f"REVIEW:            {counters['REVIEW']:,}"
    )

    print(
        f"EXCLUDE_CANDIDATE: {counters['EXCLUDE_CANDIDATE']:,}"
    )

    print(
        f"CRITICAL:          {counters['CRITICAL']:,}"
    )

    print(
        f"UNKNOWN:           {counters['UNKNOWN']:,}"
    )

    print()
    print(
        f"Imágenes copiadas: {copied_images:,}"
    )

    print()
    print("Dataset generado:")
    print(CLEANED_ROOT)

    print()
    print(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()