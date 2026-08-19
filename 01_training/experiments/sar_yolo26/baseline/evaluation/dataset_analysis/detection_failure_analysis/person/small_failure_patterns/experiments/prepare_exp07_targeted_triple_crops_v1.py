from __future__ import annotations

import csv
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


# ============================================================================
# SAR YOLO26 - EXP07 PREPARE TARGETED TRIPLE CROPS V2
# ============================================================================
#
# CORRECCIÓN V2:
# - Busca imágenes y labels recursivamente.
# - No asume estructura plana.
# - Valida que cada GT index exista en el label origen.
# - Registra motivos de descarte.
# - Máximo 3500 crops.
# - Máximo 8 crops por imagen.
#
# NO modifica dataset original.
# NO modifica labels originales.
# NO modifica YAML oficial.
# ============================================================================


SEED = 42

TARGET_CROPS = 3500
MAX_CROPS_PER_SOURCE_IMAGE = 8

CROP_WIDTH_RATIO = 0.40
CROP_HEIGHT_RATIO = 0.40
MIN_BOX_VISIBILITY = 0.35

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
# LOCALIZACIÓN
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_baseline_dir() -> Path:

    for parent in [
        SCRIPT_DIR,
        *SCRIPT_DIR.parents,
    ]:

        if parent.name.lower() == "baseline":
            return parent

    raise RuntimeError(
        "No se pudo localizar baseline."
    )


BASELINE_DIR = find_baseline_dir()

PROJECT_ROOT = BASELINE_DIR.parents[3]


# ============================================================================
# DATASET ORIGINAL
# ============================================================================

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


# ============================================================================
# EXP04
# ============================================================================

EXP04_DENSE_IMAGES_DIR = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp04_dense_scene_targeted_crops_v1"
    / "dense_crops"
    / "images"
)


# ============================================================================
# EXP07
# ============================================================================

EXP07_ROOT = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp07_targeted_extreme_small_dense_neighbor_v1"
)

EXP07_CROPS_ROOT = (
    EXP07_ROOT
    / "triple_crops"
)

EXP07_IMAGES_DIR = (
    EXP07_CROPS_ROOT
    / "images"
)

EXP07_LABELS_DIR = (
    EXP07_CROPS_ROOT
    / "labels"
)

EXP07_MANIFEST = (
    EXP07_ROOT
    / "train_with_exp04_and_exp07.txt"
)

EXP07_DATA_YAML = (
    EXP07_ROOT
    / "exp07_dataset.yaml"
)


# ============================================================================
# INPUT REPORTS
# ============================================================================

INPUT_REPORT_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp07_triple_population_analysis_v1"
    / "reports"
)

TRIPLE_SOURCE_CSV = (
    INPUT_REPORT_DIR
    / "exp07_triple_population_objects_v1.csv"
)

SAMPLING_PLAN_CSV = (
    INPUT_REPORT_DIR
    / "exp07_triple_population_stratified_sampling_plan_v1.csv"
)


# ============================================================================
# OUTPUT REPORTS
# ============================================================================

REPORTS_DIR = (
    BASELINE_DIR
    / "evaluation"
    / "dataset_analysis"
    / "detection_failure_analysis"
    / "person"
    / "small_failure_patterns"
    / "experiments"
    / "exp07_targeted_extreme_small_dense_neighbor_v1"
    / "reports"
)

SELECTION_CSV = (
    REPORTS_DIR
    / "exp07_selected_triple_targets_v2.csv"
)

IMAGE_STATS_CSV = (
    REPORTS_DIR
    / "exp07_selected_targets_by_image_v2.csv"
)

STRATA_STATS_CSV = (
    REPORTS_DIR
    / "exp07_selected_targets_by_stratum_v2.csv"
)

FAILURES_CSV = (
    REPORTS_DIR
    / "exp07_crop_generation_failures_v2.csv"
)

SUMMARY_TXT = (
    REPORTS_DIR
    / "EXP07_TARGETED_TRIPLE_CROPS_PREPARATION_V2_SUMMARY.txt"
)


# ============================================================================
# UTILIDADES
# ============================================================================

def safe_div(
    a: float,
    b: float,
) -> float:

    if b == 0:
        return 0.0

    return a / b


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )

        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# BUCKETS
# ============================================================================

def get_size_bucket(
    value: float,
) -> str:

    if value < 8:
        return "<8"

    if value < 10:
        return "8-10"

    if value < 12:
        return "10-12"

    if value < 14:
        return "12-14"

    if value < 16:
        return "14-16"

    return ">=16"


def get_density_bucket(
    value: int,
) -> str:

    if value < 50:
        return "25-49"

    if value < 100:
        return "50-99"

    if value < 200:
        return "100-199"

    return ">=200"


# ============================================================================
# INDEXAR IMÁGENES Y LABELS
# ============================================================================

def build_image_index() -> dict[str, Path]:

    print()
    print("Indexando TRAIN images...")

    index = {}

    for path in TRAIN_IMAGES_DIR.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            key = path.name.lower()

            if key in index:

                raise RuntimeError(
                    "Hay imágenes duplicadas con el mismo nombre:\n"
                    f"{index[key]}\n"
                    f"{path}"
                )

            index[key] = path

    print(
        f"[OK] TRAIN images indexadas: "
        f"{len(index):,}"
    )

    return index


def build_label_index() -> dict[str, Path]:

    print()
    print("Indexando TRAIN labels...")

    index = {}

    for path in TRAIN_LABELS_DIR.rglob("*.txt"):

        if not path.is_file():
            continue

        key = path.name.lower()

        if key in index:

            raise RuntimeError(
                "Hay labels duplicados con el mismo nombre:\n"
                f"{index[key]}\n"
                f"{path}"
            )

        index[key] = path

    print(
        f"[OK] TRAIN labels indexados: "
        f"{len(index):,}"
    )

    return index


# ============================================================================
# CARGA DE POBLACIÓN TRIPLE
# ============================================================================

def load_triple_population() -> list[dict]:

    if not TRIPLE_SOURCE_CSV.exists():

        raise FileNotFoundError(
            f"No se encontró:\n{TRIPLE_SOURCE_CSV}"
        )

    with TRIPLE_SOURCE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:

        raise RuntimeError(
            "El CSV TRIPLE está vacío."
        )

    output = []

    for row in rows:

        output.append(
            {
                "image":
                    row["image"],

                "gt_index":
                    int(
                        row["gt_index"]
                    ),

                "area":
                    float(
                        row["area"]
                    ),

                "size_sqrt":
                    float(
                        row["size_sqrt"]
                    ),

                "person_count":
                    int(
                        row["person_count"]
                    ),

                "nearest_distance":
                    (
                        float(
                            row[
                                "nearest_distance"
                            ]
                        )
                        if row[
                            "nearest_distance"
                        ]
                        not in ("", None)
                        else math.inf
                    ),

                "size_bucket":
                    get_size_bucket(
                        float(
                            row["size_sqrt"]
                        )
                    ),

                "density_bucket":
                    get_density_bucket(
                        int(
                            row["person_count"]
                        )
                    ),
            }
        )

    return output


# ============================================================================
# PLAN DE MUESTREO
# ============================================================================

def load_sampling_plan() -> dict:

    if not SAMPLING_PLAN_CSV.exists():

        raise FileNotFoundError(
            f"No se encontró:\n{SAMPLING_PLAN_CSV}"
        )

    plan = {}

    with SAMPLING_PLAN_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            key = (
                row["size_bucket"],
                row["density_bucket"],
            )

            plan[key] = int(
                row["proposed_samples"]
            )

    return plan


def select_targets(
    candidates: list[dict],
    plan: dict,
) -> list[dict]:

    rng = random.Random(
        SEED
    )

    strata = defaultdict(list)

    for row in candidates:

        strata[
            (
                row["size_bucket"],
                row["density_bucket"],
            )
        ].append(row)

    for rows in strata.values():
        rng.shuffle(rows)

    selected = []

    per_image = Counter()

    # ---------------------------------------------
    # Primera pasada según cuotas estratificadas
    # ---------------------------------------------

    for key, quota in plan.items():

        rows = strata.get(
            key,
            [],
        )

        count = 0

        for row in rows:

            if len(selected) >= TARGET_CROPS:
                break

            image = row["image"]

            if (
                per_image[image]
                >=
                MAX_CROPS_PER_SOURCE_IMAGE
            ):
                continue

            selected.append(
                row
            )

            per_image[image] += 1
            count += 1

            if count >= quota:
                break

    # ---------------------------------------------
    # Relleno hasta 3500
    # ---------------------------------------------

    if len(selected) < TARGET_CROPS:

        selected_keys = {
            (
                r["image"],
                r["gt_index"],
            )
            for r in selected
        }

        remaining = [
            r
            for r in candidates
            if (
                r["image"],
                r["gt_index"],
            )
            not in selected_keys
        ]

        rng.shuffle(
            remaining
        )

        for row in remaining:

            if len(selected) >= TARGET_CROPS:
                break

            image = row["image"]

            if (
                per_image[image]
                >=
                MAX_CROPS_PER_SOURCE_IMAGE
            ):
                continue

            selected.append(
                row
            )

            per_image[image] += 1

    return selected


# ============================================================================
# LABELS ORIGINALES
# ============================================================================

def load_label_objects(
    label_path: Path,
) -> list[dict]:

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except UnicodeDecodeError:

        lines = label_path.read_text(
            encoding="latin-1"
        ).splitlines()

    objects = []

    for index, line in enumerate(
        lines
    ):

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:

            class_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:

            continue

        if w <= 0 or h <= 0:
            continue

        objects.append(
            {
                "index":
                    index,

                "class_id":
                    class_id,

                "xc":
                    xc,

                "yc":
                    yc,

                "w":
                    w,

                "h":
                    h,
            }
        )

    return objects


# ============================================================================
# BOXES
# ============================================================================

def yolo_to_xyxy(
    obj: dict,
    iw: int,
    ih: int,
) -> list[float]:

    cx = obj["xc"] * iw
    cy = obj["yc"] * ih

    bw = obj["w"] * iw
    bh = obj["h"] * ih

    return [
        cx - bw / 2,
        cy - bh / 2,
        cx + bw / 2,
        cy + bh / 2,
    ]


def clip_box(
    box: list[float],
    crop: tuple[int, int, int, int],
) -> tuple[list[float], float]:

    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop

    ix1 = max(
        x1,
        cx1,
    )

    iy1 = max(
        y1,
        cy1,
    )

    ix2 = min(
        x2,
        cx2,
    )

    iy2 = min(
        y2,
        cy2,
    )

    original_area = (
        max(
            0.0,
            x2 - x1,
        )
        *
        max(
            0.0,
            y2 - y1,
        )
    )

    intersection_area = (
        max(
            0.0,
            ix2 - ix1,
        )
        *
        max(
            0.0,
            iy2 - iy1,
        )
    )

    visibility = safe_div(
        intersection_area,
        original_area,
    )

    return (
        [
            ix1,
            iy1,
            ix2,
            iy2,
        ],
        visibility,
    )


def box_to_crop_yolo(
    box: list[float],
    crop: tuple[int, int, int, int],
) -> tuple[
    float,
    float,
    float,
    float,
]:

    cx1, cy1, cx2, cy2 = crop

    cw = cx2 - cx1
    ch = cy2 - cy1

    x1 = max(
        0,
        min(
            cw,
            box[0] - cx1,
        )
    )

    y1 = max(
        0,
        min(
            ch,
            box[1] - cy1,
        )
    )

    x2 = max(
        0,
        min(
            cw,
            box[2] - cx1,
        )
    )

    y2 = max(
        0,
        min(
            ch,
            box[3] - cy1,
        )
    )

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    xc = (
        x1 + w / 2
    ) / cw

    yc = (
        y1 + h / 2
    ) / ch

    wn = w / cw
    hn = h / ch

    return (
        xc,
        yc,
        wn,
        hn,
    )


# ============================================================================
# CROP
# ============================================================================

def compute_crop(
    target_box: list[float],
    iw: int,
    ih: int,
) -> tuple[int, int, int, int]:

    target_cx = (
        target_box[0]
        +
        target_box[2]
    ) / 2

    target_cy = (
        target_box[1]
        +
        target_box[3]
    ) / 2

    cw = min(
        iw,
        int(
            round(
                iw
                *
                CROP_WIDTH_RATIO
            )
        ),
    )

    ch = min(
        ih,
        int(
            round(
                ih
                *
                CROP_HEIGHT_RATIO
            )
        ),
    )

    x1 = int(
        round(
            target_cx
            -
            cw / 2
        )
    )

    y1 = int(
        round(
            target_cy
            -
            ch / 2
        )
    )

    x1 = max(
        0,
        min(
            x1,
            iw - cw,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            ih - ch,
        ),
    )

    return (
        x1,
        y1,
        x1 + cw,
        y1 + ch,
    )


# ============================================================================
# GENERAR CROPS
# ============================================================================

def generate_crops(
    selected: list[dict],
    image_index: dict,
    label_index: dict,
) -> tuple[list[dict], list[dict]]:

    if EXP07_CROPS_ROOT.exists():

        shutil.rmtree(
            EXP07_CROPS_ROOT
        )

    EXP07_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXP07_LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = []
    generated = []

    grouped = defaultdict(
        list
    )

    for row in selected:

        grouped[
            row["image"]
        ].append(row)

    for image_number, (
        image_name,
        targets,
    ) in enumerate(
        grouped.items(),
        start=1,
    ):

        image_key = (
            Path(
                image_name
            ).name.lower()
        )

        image_path = image_index.get(
            image_key
        )

        if image_path is None:

            failures.append(
                {
                    "image":
                        image_name,

                    "reason":
                        "IMAGE_NOT_FOUND",
                }
            )

            continue

        label_name = (
            image_path.stem
            +
            ".txt"
        ).lower()

        label_path = label_index.get(
            label_name
        )

        if label_path is None:

            failures.append(
                {
                    "image":
                        image_name,

                    "reason":
                        "LABEL_NOT_FOUND",
                }
            )

            continue

        try:

            with Image.open(
                image_path
            ) as image:

                iw = image.width
                ih = image.height

                source_objects = (
                    load_label_objects(
                        label_path
                    )
                )

                source_by_index = {
                    obj["index"]:
                        obj
                    for obj
                    in source_objects
                }

                for target in targets:

                    target_obj = (
                        source_by_index.get(
                            target[
                                "gt_index"
                            ]
                        )
                    )

                    if target_obj is None:

                        failures.append(
                            {
                                "image":
                                    image_name,

                                "gt_index":
                                    target[
                                        "gt_index"
                                    ],

                                "reason":
                                    "GT_INDEX_NOT_FOUND",
                            }
                        )

                        continue

                    target_box = (
                        yolo_to_xyxy(
                            target_obj,
                            iw,
                            ih,
                        )
                    )

                    crop_box = (
                        compute_crop(
                            target_box,
                            iw,
                            ih,
                        )
                    )

                    crop_image = (
                        image.crop(
                            crop_box
                        )
                    )

                    crop_name = (
                        f"{image_path.stem}"
                        f"__triple_"
                        f"{target['gt_index']}.jpg"
                    )

                    output_image = (
                        EXP07_IMAGES_DIR
                        /
                        crop_name
                    )

                    crop_image.save(
                        output_image,
                        quality=95,
                    )

                    label_lines = []

                    retained = 0

                    for obj in source_objects:

                        obj_box = (
                            yolo_to_xyxy(
                                obj,
                                iw,
                                ih,
                            )
                        )

                        clipped, visibility = (
                            clip_box(
                                obj_box,
                                crop_box,
                            )
                        )

                        if (
                            visibility
                            <
                            MIN_BOX_VISIBILITY
                        ):
                            continue

                        xc, yc, wn, hn = (
                            box_to_crop_yolo(
                                clipped,
                                crop_box,
                            )
                        )

                        if (
                            wn <= 0
                            or
                            hn <= 0
                        ):
                            continue

                        label_lines.append(
                            f"{obj['class_id']} "
                            f"{xc:.8f} "
                            f"{yc:.8f} "
                            f"{wn:.8f} "
                            f"{hn:.8f}"
                        )

                        retained += 1

                    if not label_lines:

                        output_image.unlink(
                            missing_ok=True
                        )

                        failures.append(
                            {
                                "image":
                                    image_name,

                                "gt_index":
                                    target[
                                        "gt_index"
                                    ],

                                "reason":
                                    "NO_RETAINED_LABELS",
                            }
                        )

                        continue

                    output_label = (
                        EXP07_LABELS_DIR
                        /
                        (
                            Path(
                                crop_name
                            ).stem
                            +
                            ".txt"
                        )
                    )

                    output_label.write_text(
                        "\n".join(
                            label_lines
                        )
                        +
                        "\n",
                        encoding="utf-8",
                    )

                    generated.append(
                        {
                            "source_image":
                                image_name,

                            "gt_index":
                                target[
                                    "gt_index"
                                ],

                            "crop_image":
                                crop_name,

                            "size_bucket":
                                target[
                                    "size_bucket"
                                ],

                            "density_bucket":
                                target[
                                    "density_bucket"
                                ],

                            "retained_labels":
                                retained,
                        }
                    )

        except Exception as exc:

            failures.append(
                {
                    "image":
                        image_name,

                    "reason":
                        "IMAGE_PROCESSING_ERROR",

                    "detail":
                        str(exc),
                }
            )

        if (
            image_number % 50 == 0
            or
            image_number
            ==
            len(grouped)
        ):

            print(
                f"Imágenes: "
                f"{image_number:,}/"
                f"{len(grouped):,} "
                f"| crops={len(generated):,} "
                f"| fallos={len(failures):,}"
            )

    return (
        generated,
        failures,
    )


# ============================================================================
# MANIFEST
# ============================================================================

def create_manifest() -> int:

    original = [
        p
        for p in TRAIN_IMAGES_DIR.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    dense = [
        p
        for p in EXP04_DENSE_IMAGES_DIR.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    triple = [
        p
        for p in EXP07_IMAGES_DIR.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    if not original:
        raise RuntimeError(
            "No existen imágenes TRAIN originales."
        )

    if not dense:
        raise RuntimeError(
            "No existen dense crops de EXP04."
        )

    if not triple:
        raise RuntimeError(
            "No existen triple crops EXP07."
        )

    lines = [
        str(
            p.resolve()
        )
        for p in original
    ]

    lines.extend(
        str(
            p.resolve()
        )
        for p in dense
    )

    lines.extend(
        str(
            p.resolve()
        )
        for p in triple
    )

    EXP07_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXP07_MANIFEST.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        "MANIFEST"
    )

    print(
        f"TRAIN original: {len(original):,}"
    )

    print(
        f"EXP04 dense:    {len(dense):,}"
    )

    print(
        f"EXP07 triple:   {len(triple):,}"
    )

    print(
        f"TOTAL:          {len(lines):,}"
    )

    return len(lines)


# ============================================================================
# YAML EXPERIMENTAL
# ============================================================================

def create_yaml() -> None:

    content = (
        f"path: "
        f"{DATASET_ROOT.as_posix()}\n\n"
        f"train: "
        f"{EXP07_MANIFEST.as_posix()}\n"
        f"val: "
        f"{VAL_IMAGES_DIR.as_posix()}\n"
        f"test: "
        f"{TEST_IMAGES_DIR.as_posix()}\n\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle\n"
    )

    EXP07_DATA_YAML.write_text(
        content,
        encoding="utf-8",
    )

    print()
    print(
        "[OK] YAML experimental:"
    )

    print(
        f"     {EXP07_DATA_YAML}"
    )

    print(
        "[INFO] YAML oficial NO modificado."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 72)
    print(
        "# SAR YOLO26 - EXP07 PREPARACIÓN CONTROLADA V2"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = (
        load_triple_population()
    )

    plan = (
        load_sampling_plan()
    )

    print()
    print(
        f"[OK] TRIPLE disponibles: "
        f"{len(candidates):,}"
    )

    selected = select_targets(
        candidates,
        plan,
    )

    print(
        f"[OK] Targets seleccionados: "
        f"{len(selected):,}"
    )

    write_csv(
        SELECTION_CSV,
        selected,
    )

    print()
    print(
        "Construyendo índices recursivos..."
    )

    image_index = (
        build_image_index()
    )

    label_index = (
        build_label_index()
    )

    print()
    print(
        "Generando crops..."
    )

    generated, failures = (
        generate_crops(
            selected,
            image_index,
            label_index,
        )
    )

    print()
    print(
        f"[RESULTADO] "
        f"Crops generados: "
        f"{len(generated):,}"
    )

    print(
        f"[RESULTADO] "
        f"Fallos: "
        f"{len(failures):,}"
    )

    if failures:

        write_csv(
            FAILURES_CSV,
            failures,
        )

        print()
        print(
            "[WARNING] "
            f"Se registraron {len(failures):,} fallos."
        )

    if not generated:

        raise RuntimeError(
            "No se generó ningún crop. "
            "Revisa el report:\n"
            f"{FAILURES_CSV}"
        )

    # ------------------------------------------
    # Estadísticas
    # ------------------------------------------

    image_counts = Counter(
        row["source_image"]
        for row in generated
    )

    image_stats = [
        {
            "source_image":
                image,

            "generated_crops":
                count,
        }
        for image, count
        in sorted(
            image_counts.items()
        )
    ]

    strata_counts = Counter(
        (
            row["size_bucket"],
            row["density_bucket"],
        )
        for row in generated
    )

    strata_stats = [
        {
            "size_bucket":
                key[0],

            "density_bucket":
                key[1],

            "generated_crops":
                count,
        }
        for key, count
        in sorted(
            strata_counts.items()
        )
    ]

    write_csv(
        IMAGE_STATS_CSV,
        image_stats,
    )

    write_csv(
        STRATA_STATS_CSV,
        strata_stats,
    )

    manifest_count = (
        create_manifest()
    )

    create_yaml()

    max_per_image = (
        max(
            image_counts.values()
        )
        if image_counts
        else 0
    )

    summary_lines = [
        "=" * 72,
        "SAR YOLO26 - EXP07 TARGETED TRIPLE CROPS PREPARATION V2",
        "=" * 72,
        "",
        "TRIPLE AVAILABLE:",
        "13,849",
        "",
        "SELECTED:",
        f"{len(selected):,}",
        "",
        "GENERATED:",
        f"{len(generated):,}",
        "",
        "FAILED:",
        f"{len(failures):,}",
        "",
        "UNIQUE SOURCE IMAGES:",
        f"{len(image_counts):,}",
        "",
        "MAX CROPS PER IMAGE:",
        f"{max_per_image}",
        "",
        "MANIFEST:",
        f"{manifest_count:,}",
        "",
        "IMPORTANTE:",
        "Dataset original NO modificado.",
        "Labels originales NO modificados.",
        "YAML oficial NO modificado.",
        "",
        "EXP07 YAML:",
        str(EXP07_DATA_YAML),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(
        "# EXP07 PREPARACIÓN FINALIZADA"
    )
    print("=" * 72)

    print(
        f"Targets:          "
        f"{len(selected):,}"
    )

    print(
        f"Crops generados:  "
        f"{len(generated):,}"
    )

    print(
        f"Fallos:           "
        f"{len(failures):,}"
    )

    print(
        f"Max/img:          "
        f"{max_per_image}"
    )

    print(
        f"Manifest:         "
        f"{manifest_count:,}"
    )

    print()
    print(
        f"[OK] {SELECTION_CSV}"
    )

    print(
        f"[OK] {IMAGE_STATS_CSV}"
    )

    print(
        f"[OK] {STRATA_STATS_CSV}"
    )

    if failures:
        print(
            f"[OK] {FAILURES_CSV}"
        )

    print(
        f"[OK] {SUMMARY_TXT}"
    )

    print()
    print(
        f"[OK] {EXP07_DATA_YAML}"
    )

    print()
    print(
        "NO se entrenó el modelo."
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n[CANCELADO]"
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "=" * 72
        )

        print(
            "[ERROR EXP07 PREPARATION V2]"
        )

        print()

        print(
            str(exc)
        )

        print()

        sys.exit(1)