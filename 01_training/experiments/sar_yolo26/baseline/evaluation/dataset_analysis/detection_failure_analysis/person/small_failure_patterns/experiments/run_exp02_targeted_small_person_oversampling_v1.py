from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26
# EXP02 - TARGETED SMALL PERSON OVERSAMPLING V1
# ============================================================================
#
# OBJETIVO
# --------
# Aumentar la exposición del modelo a imágenes que contienen PERSON pequeñas.
#
# INTERVENCIÓN ÚNICA
# ------------------
# Imagen sin PERSON < 256 px²  -> 1 aparición
# Imagen con PERSON < 256 px²  -> 2 apariciones
#
# TODO LO DEMÁS SE MANTIENE:
#   model  = YOLO26s
#   imgsz  = 640
#   epochs = 100
#   batch  = 8
#   workers = 8
#   seed   = 42
#   device = 0
#
# EVALUACIÓN:
#   test_dev
#   imgsz = 1536
#   IoU referencia = 0.50
#
# IMPORTANTE
# ----------
# - NO modifica el dataset original.
# - NO modifica train/images.
# - NO modifica train/labels.
# - NO modifica val.
# - NO modifica test_dev.
# - NO modifica el YAML oficial.
# - Genera únicamente artefactos propios de EXP02.
#
# ============================================================================


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

EXPERIMENT_NAME = (
    "exp02_targeted_small_person_oversampling_v1"
)

PERSON_CLASS_ID = 0
VEHICLE_CLASS_ID = 1

SMALL_AREA_THRESHOLD = 256.0

# Factor de oversampling de imágenes con al menos
# una PERSON pequeña.
OVERSAMPLING_FACTOR = 2

# ---------------------------------------------------------------------------
# ENTRENAMIENTO: igual que baseline
# ---------------------------------------------------------------------------

TRAIN_IMAGE_SIZE = 640
EPOCHS = 100
BATCH = 8
WORKERS = 8
DEVICE = 0
SEED = 42
AMP = True
PATIENCE = 20
CACHE = False

# ---------------------------------------------------------------------------
# EVALUACIÓN
# ---------------------------------------------------------------------------

EVAL_IMAGE_SIZE = 1536
EVAL_CONF_THRESHOLD = 0.25
EVAL_MATCH_IOU = 0.50

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================================
# SCRIPT / PROYECTO
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root() -> Path:
    """
    Localiza C:\\SARC-Drone.
    """

    for directory in [
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
    ]:

        if directory.name.lower() == "sarc-drone":
            return directory

    raise RuntimeError(
        "No se pudo localizar C:\\SARC-Drone.\n"
        f"Script:\n{SCRIPT_PATH}"
    )


PROJECT_ROOT = find_project_root()


# ============================================================================
# RUTAS
# ============================================================================

BASELINE_DIR = (
    PROJECT_ROOT
    / "01_training"
    / "experiments"
    / "sar_yolo26"
    / "baseline"
)


DATASET_ROOT = (
    PROJECT_ROOT
    / "00_datasets"
    / "SAR_DATASET_STUDIO"
    / "processed"
    / "sar"
    / "cleaned"
    / "VisDrone_SAR_2CLASS_V1"
)


TRAIN_IMAGES_DIR = (
    DATASET_ROOT
    / "train"
    / "images"
)

TRAIN_LABELS_DIR = (
    DATASET_ROOT
    / "train"
    / "labels"
)

VAL_IMAGES_DIR = (
    DATASET_ROOT
    / "val"
    / "images"
)

TEST_IMAGES_DIR = (
    DATASET_ROOT
    / "test_dev"
    / "images"
)

TEST_LABELS_DIR = (
    DATASET_ROOT
    / "test_dev"
    / "labels"
)


# ============================================================================
# MODELO PRETRAINED
# ============================================================================

MODEL_CANDIDATES = [

    BASELINE_DIR
    / "yolo26s.pt",

    BASELINE_DIR
    / "training"
    / "models"
    / "pretrained"
    / "yolo26s.pt",

    PROJECT_ROOT
    / "01_training"
    / "models"
    / "pretrained"
    / "yolo26s.pt",

    PROJECT_ROOT
    / "yolo26s.pt",
]


# ============================================================================
# DIRECTORIOS EXP02
# ============================================================================

EXPERIMENT_ROOT = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / EXPERIMENT_NAME
)


RUNS_DIR = (
    EXPERIMENT_ROOT
    / "runs"
)


EXPERIMENT_REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / EXPERIMENT_NAME
    / "reports"
)


# ============================================================================
# ARCHIVOS EXP02
# ============================================================================

TRAIN_MANIFEST = (
    EXPERIMENT_ROOT
    / "train_oversampled.txt"
)


TEMP_DATA_YAML = (
    EXPERIMENT_ROOT
    / "exp02_dataset.yaml"
)


OVERSAMPLING_CSV = (
    EXPERIMENT_REPORTS_DIR
    / "exp02_oversampling_statistics_v1.csv"
)


COMPARE_CSV = (
    EXPERIMENT_REPORTS_DIR
    / "exp02_vs_exp01_configuration_v1.csv"
)


EVAL_SUMMARY_CSV = (
    EXPERIMENT_REPORTS_DIR
    / "exp02_evaluation_summary_v1.csv"
)


TRAINING_SUMMARY = (
    EXPERIMENT_REPORTS_DIR
    / "EXP02_TRAINING_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def image_area_from_yolo(
    width_norm: float,
    height_norm: float,
    image_width: int,
    image_height: int,
) -> float:

    return (
        width_norm
        * height_norm
        * image_width
        * image_height
    )


def find_model() -> Path:

    for candidate in MODEL_CANDIDATES:

        if candidate.is_file():

            return candidate

    # Búsqueda controlada si no está en las ubicaciones conocidas.

    candidates = []

    for root in [
        PROJECT_ROOT / "01_training",
        PROJECT_ROOT,
    ]:

        if not root.exists():
            continue

        try:

            for candidate in root.rglob(
                "yolo26s.pt"
            ):

                if candidate.is_file():

                    candidates.append(
                        candidate
                    )

        except PermissionError:

            continue

    candidates = sorted(
        set(candidates),
        key=lambda p: str(p).lower(),
    )

    if len(candidates) == 1:

        return candidates[0]

    if len(candidates) > 1:

        lines = "\n".join(
            f"  - {path}"
            for path in candidates
        )

        raise RuntimeError(
            "Se encontraron varias copias de yolo26s.pt:\n"
            f"{lines}\n\n"
            "No se seleccionará ninguna arbitrariamente."
        )

    raise FileNotFoundError(
        "No se encontró yolo26s.pt."
    )


def validate_structure() -> None:

    print()
    print("=" * 72)
    print("VALIDANDO ESTRUCTURA EXP02")
    print("=" * 72)
    print()

    required_paths = {

        "PROJECT_ROOT":
            PROJECT_ROOT,

        "BASELINE_DIR":
            BASELINE_DIR,

        "DATASET_ROOT":
            DATASET_ROOT,

        "TRAIN_IMAGES_DIR":
            TRAIN_IMAGES_DIR,

        "TRAIN_LABELS_DIR":
            TRAIN_LABELS_DIR,

        "VAL_IMAGES_DIR":
            VAL_IMAGES_DIR,

        "TEST_IMAGES_DIR":
            TEST_IMAGES_DIR,

        "TEST_LABELS_DIR":
            TEST_LABELS_DIR,
    }

    for name, path in required_paths.items():

        if not path.exists():

            raise FileNotFoundError(
                f"No se encontró {name}:\n{path}"
            )

        print(
            f"[OK] {name}"
        )

        print(
            f"     {path}"
        )

    model_path = find_model()

    print()
    print(
        "[OK] PRETRAINED MODEL"
    )

    print(
        f"     {model_path}"
    )


def ensure_output_dirs() -> None:

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# SMALL PERSON POR IMAGEN
# ============================================================================

def count_small_persons_in_label(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> int:
    """
    Cuenta PERSON pequeñas de una imagen.
    """

    if not label_path.exists():

        return 0

    count = 0

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except UnicodeDecodeError:

        lines = label_path.read_text(
            encoding="latin-1"
        ).splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:

            class_id = int(
                float(parts[0])
            )

            width_norm = float(
                parts[3]
            )

            height_norm = float(
                parts[4]
            )

        except ValueError:

            continue

        if class_id != PERSON_CLASS_ID:
            continue

        if width_norm <= 0:
            continue

        if height_norm <= 0:
            continue

        area = image_area_from_yolo(
            width_norm,
            height_norm,
            image_width,
            image_height,
        )

        if area < SMALL_AREA_THRESHOLD:

            count += 1

    return count


# ============================================================================
# MANIFEST
# ============================================================================

def build_oversampled_manifest() -> dict:

    print()
    print("=" * 72)
    print("CONSTRUYENDO MANIFEST EXP02")
    print("=" * 72)
    print()

    image_files = sorted(
        [
            path
            for path in TRAIN_IMAGES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:

        raise RuntimeError(
            "No se encontraron imágenes de TRAIN."
        )

    original_images = len(
        image_files
    )

    targeted_images = 0
    normal_images = 0
    small_person_gt = 0

    manifest_lines = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        label_path = (
            TRAIN_LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        try:

            with Image.open(
                image_path
            ) as image:

                image_width = image.width
                image_height = image.height

        except Exception as exc:

            raise RuntimeError(
                "\nNo se pudo leer:\n"
                f"{image_path}\n"
                f"Error: {exc}"
            ) from exc

        small_count = (
            count_small_persons_in_label(
                label_path,
                image_width,
                image_height,
            )
        )

        small_person_gt += small_count

        image_string = str(
            image_path.resolve()
        )

        if small_count > 0:

            targeted_images += 1

            repeats = (
                OVERSAMPLING_FACTOR
            )

        else:

            normal_images += 1

            repeats = 1

        for _ in range(repeats):

            manifest_lines.append(
                image_string
            )

        if (
            index % 500 == 0
            or index == original_images
        ):

            print(
                f"Analizadas: "
                f"{index:,}/{original_images:,}"
            )

    if not manifest_lines:

        raise RuntimeError(
            "El manifest quedó vacío."
        )

    TRAIN_MANIFEST.write_text(
        "\n".join(manifest_lines),
        encoding="utf-8",
    )

    manifest_images = len(
        manifest_lines
    )

    print()
    print(
        f"Imágenes originales:       {original_images:,}"
    )

    print(
        f"Imágenes con small person: {targeted_images:,}"
    )

    print(
        f"Imágenes sin small person: {normal_images:,}"
    )

    print(
        f"PERSON small GT train:     {small_person_gt:,}"
    )

    print(
        f"Manifest appearances:      {manifest_images:,}"
    )

    print(
        f"Factor oversampling:       "
        f"{OVERSAMPLING_FACTOR}x"
    )

    print()
    print(
        "[OK] Manifest generado:"
    )

    print(
        f"     {TRAIN_MANIFEST}"
    )

    return {

        "original_images":
            original_images,

        "targeted_images":
            targeted_images,

        "normal_images":
            normal_images,

        "small_person_gt":
            small_person_gt,

        "manifest_images":
            manifest_images,
    }


# ============================================================================
# YAML EXPERIMENTAL
# ============================================================================

def create_experiment_yaml() -> None:
    """
    Genera un YAML exclusivo de EXP02.

    NO modifica el YAML oficial.
    """

    content = f"""path: {DATASET_ROOT.as_posix()}

train: {TRAIN_MANIFEST.as_posix()}
val: {VAL_IMAGES_DIR.as_posix()}
test: {TEST_IMAGES_DIR.as_posix()}

names:
  0: person
  1: vehicle
"""

    TEMP_DATA_YAML.write_text(
        content,
        encoding="utf-8",
    )

    print()
    print(
        "[OK] YAML experimental generado:"
    )

    print(
        f"     {TEMP_DATA_YAML}"
    )

    print(
        "[INFO] El YAML oficial NO ha sido modificado."
    )


# ============================================================================
# REPORT OVERSAMPLING
# ============================================================================

def write_oversampling_report(
    statistics: dict,
) -> None:

    rows = [

        {
            "parameter":
                "original_images",

            "value":
                statistics[
                    "original_images"
                ],
        },

        {
            "parameter":
                "targeted_images",

            "value":
                statistics[
                    "targeted_images"
                ],
        },

        {
            "parameter":
                "normal_images",

            "value":
                statistics[
                    "normal_images"
                ],
        },

        {
            "parameter":
                "small_person_gt_train",

            "value":
                statistics[
                    "small_person_gt"
                ],
        },

        {
            "parameter":
                "manifest_images",

            "value":
                statistics[
                    "manifest_images"
                ],
        },

        {
            "parameter":
                "oversampling_factor",

            "value":
                OVERSAMPLING_FACTOR,
        },

        {
            "parameter":
                "small_area_threshold_px2",

            "value":
                SMALL_AREA_THRESHOLD,
        },
    ]

    with OVERSAMPLING_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "value",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================================
# COMPARACIÓN EXP01 vs EXP02
# ============================================================================

def write_comparison_report() -> None:

    rows = [

        {
            "parameter":
                "model",

            "exp01":
                "YOLO26s",

            "exp02":
                "YOLO26s",

            "changed":
                "NO",
        },

        {
            "parameter":
                "train_imgsz",

            "exp01":
                640,

            "exp02":
                TRAIN_IMAGE_SIZE,

            "changed":
                "NO",
        },

        {
            "parameter":
                "epochs",

            "exp01":
                100,

            "exp02":
                EPOCHS,

            "changed":
                "NO",
        },

        {
            "parameter":
                "batch",

            "exp01":
                8,

            "exp02":
                BATCH,

            "changed":
                "NO",
        },

        {
            "parameter":
                "workers",

            "exp01":
                8,

            "exp02":
                WORKERS,

            "changed":
                "NO",
        },

        {
            "parameter":
                "seed",

            "exp01":
                42,

            "exp02":
                SEED,

            "changed":
                "NO",
        },

        {
            "parameter":
                "small_oversampling",

            "exp01":
                "1x",

            "exp02":
                f"{OVERSAMPLING_FACTOR}x",

            "changed":
                "YES",
        },

        {
            "parameter":
                "eval_imgsz",

            "exp01":
                1536,

            "exp02":
                EVAL_IMAGE_SIZE,

            "changed":
                "NO",
        },

        {
            "parameter":
                "eval_iou_reference",

            "exp01":
                0.50,

            "exp02":
                EVAL_MATCH_IOU,

            "changed":
                "NO",
        },
    ]

    with COMPARE_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "exp01",
                "exp02",
                "changed",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_exp02(
    model_path: Path,
) -> Path:

    print()
    print("=" * 72)
    print("ENTRENAMIENTO EXP02")
    print("=" * 72)
    print()

    print(
        f"Pretrained model: {model_path}"
    )

    print(
        f"Data YAML:        {TEMP_DATA_YAML}"
    )

    print(
        f"Train imgsz:      {TRAIN_IMAGE_SIZE}"
    )

    print(
        f"Epochs:           {EPOCHS}"
    )

    print(
        f"Batch:            {BATCH}"
    )

    print(
        f"Workers:          {WORKERS}"
    )

    print(
        f"Device:           {DEVICE}"
    )

    print(
        f"Seed:             {SEED}"
    )

    print()

    model = YOLO(
        str(model_path)
    )

    results = model.train(

        data=str(
            TEMP_DATA_YAML
        ),

        epochs=EPOCHS,

        imgsz=TRAIN_IMAGE_SIZE,

        batch=BATCH,

        workers=WORKERS,

        device=DEVICE,

        seed=SEED,

        amp=AMP,

        cache=CACHE,

        patience=PATIENCE,

        project=str(
            RUNS_DIR
        ),

        name="exp02_small_person_oversampling",

        pretrained=True,

        save=True,

        plots=True,

        verbose=True,
    )

    save_dir = Path(
        results.save_dir
    )

    best_path = (
        save_dir
        / "weights"
        / "best.pt"
    )

    if not best_path.exists():

        raise FileNotFoundError(
            "\nEl entrenamiento terminó pero "
            "no se encontró best.pt:\n"
            f"{best_path}"
        )

    return best_path


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_exp02(
    best_path: Path,
) -> dict:

    print()
    print("=" * 72)
    print("EVALUACIÓN EXP02 SOBRE TEST_DEV")
    print("=" * 72)
    print()

    model = YOLO(
        str(best_path)
    )

    print(
        f"Model: {best_path}"
    )

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Evaluation imgsz: "
        f"{EVAL_IMAGE_SIZE}"
    )

    print(
        f"Confidence: "
        f"{EVAL_CONF_THRESHOLD}"
    )

    metrics = model.val(

        data=str(
            TEMP_DATA_YAML
        ),

        split="test",

        imgsz=EVAL_IMAGE_SIZE,

        conf=EVAL_CONF_THRESHOLD,

        device=DEVICE,

        workers=WORKERS,

        plots=True,

        save_json=False,

        project=str(
            RUNS_DIR
        ),

        name="exp02_small_person_oversampling_test",

        verbose=True,
    )

    result = {

        "experiment":
            EXPERIMENT_NAME,

        "best_pt":
            str(best_path),

        "eval_imgsz":
            EVAL_IMAGE_SIZE,

        "conf_threshold":
            EVAL_CONF_THRESHOLD,

        "match_iou_reference":
            EVAL_MATCH_IOU,
    }

    try:

        result["precision"] = float(
            metrics.box.mp
        )

    except Exception:

        result["precision"] = ""

    try:

        result["recall"] = float(
            metrics.box.mr
        )

    except Exception:

        result["recall"] = ""

    try:

        result["map50"] = float(
            metrics.box.map50
        )

    except Exception:

        result["map50"] = ""

    try:

        result["map50_95"] = float(
            metrics.box.map
        )

    except Exception:

        result["map50_95"] = ""

    with EVAL_SUMMARY_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                result.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            result
        )

    return result


# ============================================================================
# SUMMARY
# ============================================================================

def write_training_summary(
    statistics: dict,
    best_path: Path,
    evaluation: dict,
) -> None:

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "SAR YOLO26 - EXP02 TARGETED SMALL PERSON OVERSAMPLING V1"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "OBJETIVO"
    )

    lines.append(
        "Aumentar la exposición del modelo a "
        "imágenes con PERSON pequeñas."
    )

    lines.append("")

    lines.append(
        "INTERVENCIÓN"
    )

    lines.append(
        f"Oversampling factor: "
        f"{OVERSAMPLING_FACTOR}x"
    )

    lines.append(
        f"Threshold: "
        f"< {SMALL_AREA_THRESHOLD} px²"
    )

    lines.append("")

    lines.append(
        "DATASET"
    )

    lines.append(
        str(DATASET_ROOT)
    )

    lines.append("")

    lines.append(
        "MODELO PRETRAINED"
    )

    lines.append(
        str(
            find_model()
        )
    )

    lines.append("")

    lines.append(
        "BEST MODEL"
    )

    lines.append(
        str(best_path)
    )

    lines.append("")

    lines.append(
        "TRAINING"
    )

    lines.append(
        f"imgsz={TRAIN_IMAGE_SIZE}"
    )

    lines.append(
        f"epochs={EPOCHS}"
    )

    lines.append(
        f"batch={BATCH}"
    )

    lines.append(
        f"workers={WORKERS}"
    )

    lines.append(
        f"device={DEVICE}"
    )

    lines.append(
        f"seed={SEED}"
    )

    lines.append(
        f"amp={AMP}"
    )

    lines.append(
        f"patience={PATIENCE}"
    )

    lines.append("")

    lines.append(
        "OVERSAMPLING"
    )

    lines.append(
        f"Original images: "
        f"{statistics['original_images']:,}"
    )

    lines.append(
        f"Targeted images: "
        f"{statistics['targeted_images']:,}"
    )

    lines.append(
        f"Normal images: "
        f"{statistics['normal_images']:,}"
    )

    lines.append(
        f"Small PERSON GT: "
        f"{statistics['small_person_gt']:,}"
    )

    lines.append(
        f"Manifest appearances: "
        f"{statistics['manifest_images']:,}"
    )

    lines.append("")

    lines.append(
        "EVALUATION"
    )

    lines.append(
        f"test_dev images: "
        f"{TEST_IMAGES_DIR}"
    )

    lines.append(
        f"imgsz={EVAL_IMAGE_SIZE}"
    )

    lines.append(
        f"conf={EVAL_CONF_THRESHOLD}"
    )

    lines.append(
        f"IoU reference={EVAL_MATCH_IOU}"
    )

    lines.append("")

    for key in [
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]:

        if evaluation.get(
            key,
            "",
        ) != "":

            lines.append(
                f"{key}="
                f"{evaluation[key]}"
            )

    lines.append("")

    lines.append(
        "CONTROL EXPERIMENTAL"
    )

    lines.append(
        "EXP02 modifica únicamente el muestreo de entrenamiento."
    )

    lines.append(
        "No modifica la arquitectura."
    )

    lines.append(
        "No modifica la resolución de entrenamiento."
    )

    lines.append(
        "No modifica batch, epochs, workers ni seed."
    )

    lines.append(
        "No modifica el dataset original."
    )

    lines.append("")

    lines.append(
        "SIGUIENTE PASO"
    )

    lines.append(
        "Comparar EXP02 contra EXP01."
    )

    lines.append(
        "La métrica prioritaria es SMALL PERSON RECALL."
    )

    lines.append("")

    lines.append(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    lines.append(
        "IMPORTANTE: el YAML oficial NO ha sido modificado."
    )

    TRAINING_SUMMARY.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - EXP02 TARGETED SMALL PERSON OVERSAMPLING V1"
    )
    print("=" * 72)

    print()
    print(
        "SCRIPT:"
    )
    print(
        f"  {SCRIPT_PATH}"
    )

    print()
    print(
        "PROJECT ROOT:"
    )
    print(
        f"  {PROJECT_ROOT}"
    )

    print()
    print(
        "DATASET:"
    )
    print(
        f"  {DATASET_ROOT}"
    )

    print()
    print(
        "EXPERIMENT:"
    )
    print(
        f"  {EXPERIMENT_ROOT}"
    )

    # ----------------------------------------------------------------------
    # VALIDACIÓN
    # ----------------------------------------------------------------------

    validate_structure()

    # ----------------------------------------------------------------------
    # OUTPUTS
    # ----------------------------------------------------------------------

    ensure_output_dirs()

    # ----------------------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------------------

    model_path = find_model()

    print()
    print(
        "[OK] YOLO26s pretrained:"
    )

    print(
        f"     {model_path}"
    )

    # ----------------------------------------------------------------------
    # MANIFEST
    # ----------------------------------------------------------------------

    statistics = (
        build_oversampled_manifest()
    )

    # ----------------------------------------------------------------------
    # YAML EXPERIMENTAL
    # ----------------------------------------------------------------------

    create_experiment_yaml()

    # ----------------------------------------------------------------------
    # REPORTS DE CONFIGURACIÓN
    # ----------------------------------------------------------------------

    write_oversampling_report(
        statistics
    )

    write_comparison_report()

    # ----------------------------------------------------------------------
    # ENTRENAMIENTO
    # ----------------------------------------------------------------------

    best_path = train_exp02(
        model_path
    )

    print()
    print(
        "[OK] BEST MODEL:"
    )

    print(
        f"     {best_path}"
    )

    # ----------------------------------------------------------------------
    # EVALUACIÓN
    # ----------------------------------------------------------------------

    evaluation = evaluate_exp02(
        best_path
    )

    # ----------------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------------

    write_training_summary(
        statistics=statistics,
        best_path=best_path,
        evaluation=evaluation,
    )

    # ----------------------------------------------------------------------
    # RESULTADO
    # ----------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "# RESULTADO EXP02 TARGETED SMALL PERSON OVERSAMPLING V1"
    )
    print("=" * 72)

    print()

    print(
        f"Imágenes originales:       "
        f"{statistics['original_images']:,}"
    )

    print(
        f"Imágenes con small person: "
        f"{statistics['targeted_images']:,}"
    )

    print(
        f"Factor oversampling:       "
        f"{OVERSAMPLING_FACTOR}x"
    )

    print(
        f"Manifest appearances:      "
        f"{statistics['manifest_images']:,}"
    )

    print()

    print(
        "BEST MODEL:"
    )

    print(
        f"  {best_path}"
    )

    print()

    print(
        "REPORTS:"
    )

    print(
        f"[OK] {OVERSAMPLING_CSV}"
    )

    print(
        f"[OK] {COMPARE_CSV}"
    )

    print(
        f"[OK] {EVAL_SUMMARY_CSV}"
    )

    print(
        f"[OK] {TRAINING_SUMMARY}"
    )

    print()

    print(
        "IMPORTANTE: el dataset original NO ha sido modificado."
    )

    print(
        "IMPORTANTE: el YAML oficial NO ha sido modificado."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "[CANCELADO] EXP02 interrumpido por el usuario."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print("[ERROR EXP02]")
        print("=" * 72)
        print()

        print(
            str(exc)
        )

        print()

        sys.exit(1)