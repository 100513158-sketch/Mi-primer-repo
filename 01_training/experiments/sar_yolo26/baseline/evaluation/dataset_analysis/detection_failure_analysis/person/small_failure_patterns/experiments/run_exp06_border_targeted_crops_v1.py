from __future__ import annotations

import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26 - EXP06 BORDER TARGETED CROPS V1
# ============================================================================
# Base: EXP04 (imgsz=960 + dense-scene crops)
# New intervention: additional crops around SMALL PERSON near image borders.
# Original dataset and official YAML are NEVER modified.
# ============================================================================

EXPERIMENT_NAME = "exp06_border_targeted_crops_v1"

PERSON_CLASS_ID = 0
SMALL_AREA_THRESHOLD = 256.0

# Border criterion: any side of the GT box inside 10% of image border.
EDGE_MARGIN_RATIO = 0.10

# Targeted crop size.
BORDER_CROP_WIDTH_RATIO = 0.40
BORDER_CROP_HEIGHT_RATIO = 0.40
MIN_BOX_VISIBILITY = 0.35

# Keep the positive resolution found in EXP03/EXP04.
TRAIN_IMAGE_SIZE = 960

# Same heavy-training configuration that completed EXP05 after HVCI was off.
EPOCHS = 100
BATCH = 8
WORKERS = 8
DEVICE = 0
SEED = 42
AMP = True
PATIENCE = 20
CACHE = False

# Evaluation protocol kept constant across experiments.
EVAL_IMAGE_SIZE = 1536
EVAL_CONF_THRESHOLD = 0.25
EVAL_MATCH_IOU = 0.50

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root() -> Path:
    for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if p.name.lower() == "sarc-drone":
            return p
    raise RuntimeError(
        "No se pudo localizar C:\\SARC-Drone."
    )


PROJECT_ROOT = find_project_root()

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

TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
VAL_IMAGES_DIR = DATASET_ROOT / "val" / "images"
TEST_IMAGES_DIR = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test_dev" / "labels"

EXP04_ROOT = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / "exp04_dense_scene_targeted_crops_v1"
)
EXP04_DENSE_IMAGES_DIR = EXP04_ROOT / "dense_crops" / "images"

EXPERIMENT_ROOT = (
    BASELINE_DIR
    / "training"
    / "experiments"
    / EXPERIMENT_NAME
)
RUNS_DIR = EXPERIMENT_ROOT / "runs"

BORDER_CROPS_DIR = EXPERIMENT_ROOT / "border_crops"
BORDER_IMAGES_DIR = BORDER_CROPS_DIR / "images"
BORDER_LABELS_DIR = BORDER_CROPS_DIR / "labels"

TRAIN_MANIFEST = (
    EXPERIMENT_ROOT
    / "train_with_dense_and_border_crops.txt"
)
TEMP_DATA_YAML = EXPERIMENT_ROOT / "exp06_dataset.yaml"

REPORTS_DIR = (
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

BORDER_STATS_CSV = REPORTS_DIR / "exp06_border_crop_statistics_v1.csv"
TRAIN_CONFIG_CSV = REPORTS_DIR / "exp06_training_configuration_v1.csv"
EVAL_CSV = REPORTS_DIR / "exp06_small_person_recall_v1.csv"
SIZE_CSV = REPORTS_DIR / "exp06_small_person_recall_by_size_v1.csv"
OBJECTS_CSV = REPORTS_DIR / "exp06_small_person_objects_v1.csv"
COMPARISON_CSV = REPORTS_DIR / "exp06_vs_exp01_exp02_exp03_exp04_v1.csv"
SUMMARY_TXT = REPORTS_DIR / "EXP06_BORDER_TARGETED_CROPS_V1_SUMMARY.txt"

MODEL_CANDIDATES = [
    BASELINE_DIR / "yolo26s.pt",
    BASELINE_DIR / "training" / "models" / "pretrained" / "yolo26s.pt",
    PROJECT_ROOT / "01_training" / "models" / "pretrained" / "yolo26s.pt",
    PROJECT_ROOT / "yolo26s.pt",
]


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def xywhn_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    iw: int,
    ih: int,
) -> List[float]:
    cx, cy = xc * iw, yc * ih
    bw, bh = w * iw, h * ih

    return [
        max(0.0, cx - bw / 2.0),
        max(0.0, cy - bh / 2.0),
        min(float(iw), cx + bw / 2.0),
        min(float(ih), cy + bh / 2.0),
    ]


def center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_pretrained_model() -> Path:
    for p in MODEL_CANDIDATES:
        if p.is_file():
            return p

    found = []
    for root in [PROJECT_ROOT / "01_training", PROJECT_ROOT]:
        if not root.exists():
            continue
        try:
            found.extend(
                p for p in root.rglob("yolo26s.pt") if p.is_file()
            )
        except PermissionError:
            pass

    found = sorted(set(found), key=lambda x: str(x).lower())

    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(
            "Se encontraron varias copias de yolo26s.pt:\n"
            + "\n".join(f"  - {p}" for p in found)
        )
    raise FileNotFoundError("No se encontró yolo26s.pt.")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_structure() -> None:
    print("=" * 72)
    print("VALIDANDO ESTRUCTURA EXP06")
    print("=" * 72)

    required = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "BASELINE_DIR": BASELINE_DIR,
        "DATASET_ROOT": DATASET_ROOT,
        "TRAIN_IMAGES_DIR": TRAIN_IMAGES_DIR,
        "TRAIN_LABELS_DIR": TRAIN_LABELS_DIR,
        "VAL_IMAGES_DIR": VAL_IMAGES_DIR,
        "TEST_IMAGES_DIR": TEST_IMAGES_DIR,
        "TEST_LABELS_DIR": TEST_LABELS_DIR,
        "EXP04_DENSE_IMAGES_DIR": EXP04_DENSE_IMAGES_DIR,
    }

    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"No se encontró {name}:\n{p}")
        print(f"[OK] {name}\n     {p}")

    dense = [
        p for p in EXP04_DENSE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not dense:
        raise RuntimeError("EXP04 no contiene dense crops utilizables.")

    print(f"[OK] EXP04 DENSE CROPS: {len(dense):,}")


# ============================================================================
# LABEL LOADING
# ============================================================================

def load_person_boxes(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Dict]:
    result = []

    if not label_path.exists():
        return result

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()

    for gt_index, line in enumerate(lines):
        p = line.strip().split()
        if len(p) < 5:
            continue

        try:
            cls = int(float(p[0]))
            xc, yc, w, h = map(float, p[1:5])
        except ValueError:
            continue

        if cls != PERSON_CLASS_ID or w <= 0 or h <= 0:
            continue

        box = xywhn_to_xyxy(
            xc, yc, w, h,
            image_width, image_height
        )
        area = w * h * image_width * image_height

        result.append(
            {
                "gt_index": gt_index,
                "box": box,
                "area": area,
                "size_sqrt": math.sqrt(max(area, 0.0)),
            }
        )

    return result


# ============================================================================
# BORDER LOGIC
# ============================================================================

def border_sides(
    box: List[float],
    iw: int,
    ih: int,
) -> List[str]:
    x1, y1, x2, y2 = box

    mx = iw * EDGE_MARGIN_RATIO
    my = ih * EDGE_MARGIN_RATIO

    sides = []

    if x1 <= mx:
        sides.append("LEFT")
    if y1 <= my:
        sides.append("TOP")
    if iw - x2 <= mx:
        sides.append("RIGHT")
    if ih - y2 <= my:
        sides.append("BOTTOM")

    return sides


def crop_for_target(
    target: Dict,
    iw: int,
    ih: int,
) -> Tuple[int, int, int, int]:
    cx, cy = center(target["box"])

    cw = min(iw, int(round(iw * BORDER_CROP_WIDTH_RATIO)))
    ch = min(ih, int(round(ih * BORDER_CROP_HEIGHT_RATIO)))

    x1 = int(round(cx - cw / 2.0))
    y1 = int(round(cy - ch / 2.0))

    x1 = int(clamp(x1, 0, iw - cw))
    y1 = int(clamp(y1, 0, ih - ch))

    return x1, y1, x1 + cw, y1 + ch


def clip_box(
    box: List[float],
    crop: Tuple[int, int, int, int],
) -> Tuple[List[float], float]:
    bx1, by1, bx2, by2 = box
    cx1, cy1, cx2, cy2 = crop

    x1, y1 = max(bx1, cx1), max(by1, cy1)
    x2, y2 = min(bx2, cx2), min(by2, cy2)

    original_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    clipped_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    visibility = safe_div(clipped_area, original_area)
    return [x1, y1, x2, y2], visibility


def box_to_crop_yolo(
    box: List[float],
    crop: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    cx1, cy1, cx2, cy2 = crop
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    x1, y1, x2, y2 = box

    x1 = clamp(x1 - cx1, 0, crop_w)
    y1 = clamp(y1 - cy1, 0, crop_h)
    x2 = clamp(x2 - cx1, 0, crop_w)
    y2 = clamp(y2 - cy1, 0, crop_h)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        (x1 + w / 2.0) / crop_w,
        (y1 + h / 2.0) / crop_h,
        w / crop_w,
        h / crop_h,
    )


# ============================================================================
# CROP GENERATION
# ============================================================================

def generate_border_crops() -> Dict:
    print()
    print("=" * 72)
    print("GENERANDO BORDER TARGETED CROPS")
    print("=" * 72)

    if BORDER_CROPS_DIR.exists():
        shutil.rmtree(BORDER_CROPS_DIR)

    BORDER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    BORDER_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in TRAIN_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    totals = {
        "images": 0,
        "small": 0,
        "small_border": 0,
        "crops": 0,
        "small_in_crops": 0,
        "errors": 0,
        "left": 0,
        "top": 0,
        "right": 0,
        "bottom": 0,
    }

    crop_rows = []

    for i, image_path in enumerate(image_files, 1):
        label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        try:
            with Image.open(image_path) as image:
                iw, ih = image.width, image.height
                persons = load_person_boxes(label_path, iw, ih)
                totals["images"] += 1

                small = [
                    p for p in persons
                    if p["area"] < SMALL_AREA_THRESHOLD
                ]
                totals["small"] += len(small)

                for target in small:
                    sides = border_sides(target["box"], iw, ih)
                    if not sides:
                        continue

                    totals["small_border"] += 1
                    for side in sides:
                        totals[side.lower()] += 1

                    crop = crop_for_target(target, iw, ih)
                    cropped = image.crop(crop)

                    crop_name = (
                        f"{image_path.stem}"
                        f"__border_{target['gt_index']}.jpg"
                    )
                    crop_path = BORDER_IMAGES_DIR / crop_name
                    cropped.save(crop_path, quality=95)

                    label_lines = []
                    small_here = 0

                    for person in persons:
                        clipped, visibility = clip_box(
                            person["box"], crop
                        )
                        if visibility < MIN_BOX_VISIBILITY:
                            continue

                        xc, yc, w, h = box_to_crop_yolo(clipped, crop)
                        if w <= 0 or h <= 0:
                            continue

                        crop_area = (
                            w * h *
                            (crop[2] - crop[0]) *
                            (crop[3] - crop[1])
                        )
                        if crop_area < SMALL_AREA_THRESHOLD:
                            small_here += 1

                        label_lines.append(
                            f"{PERSON_CLASS_ID} "
                            f"{xc:.8f} {yc:.8f} "
                            f"{w:.8f} {h:.8f}"
                        )

                    if not label_lines:
                        crop_path.unlink(missing_ok=True)
                        continue

                    out_label = BORDER_LABELS_DIR / (
                        f"{image_path.stem}"
                        f"__border_{target['gt_index']}.txt"
                    )
                    out_label.write_text(
                        "\n".join(label_lines) + "\n",
                        encoding="utf-8",
                    )

                    totals["crops"] += 1
                    totals["small_in_crops"] += small_here

                    crop_rows.append(
                        {
                            "source_image": image_path.name,
                            "target_gt_index": target["gt_index"],
                            "source_person_count": len(persons),
                            "border_sides": "+".join(sides),
                            "target_area": round(target["area"], 6),
                            "target_size_sqrt": round(
                                target["size_sqrt"], 6
                            ),
                            "crop_image": crop_name,
                            "crop_width": crop[2] - crop[0],
                            "crop_height": crop[3] - crop[1],
                            "persons_in_crop": len(label_lines),
                            "small_persons_in_crop": small_here,
                            "is_dense_scene": int(len(persons) >= 25),
                        }
                    )

        except Exception as exc:
            totals["errors"] += 1
            print(f"[WARNING] {image_path.name}: {exc}")

        if i % 500 == 0 or i == len(image_files):
            print(
                f"Procesadas: {i:,}/{len(image_files):,} "
                f"| Small: {totals['small']:,} "
                f"| Small+border: {totals['small_border']:,} "
                f"| Crops: {totals['crops']:,}"
            )

    print()
    print(f"TRAIN images procesadas: {totals['images']:,}")
    print(f"SMALL PERSON:             {totals['small']:,}")
    print(f"SMALL PERSON + BORDER:    {totals['small_border']:,}")
    print(f"BORDER CROPS:             {totals['crops']:,}")
    print(f"SMALL IN BORDER CROPS:    {totals['small_in_crops']:,}")
    print(f"IMAGE ERRORS:             {totals['errors']:,}")

    print()
    print("BORDER DISTRIBUTION:")
    print(f"  LEFT:   {totals['left']:,}")
    print(f"  TOP:    {totals['top']:,}")
    print(f"  RIGHT:  {totals['right']:,}")
    print(f"  BOTTOM: {totals['bottom']:,}")

    if not crop_rows:
        raise RuntimeError(
            "No se generó ningún border crop."
        )

    write_csv(BORDER_STATS_CSV, crop_rows)
    return totals


# ============================================================================
# MANIFEST + YAML
# ============================================================================

def create_manifest() -> int:
    original = sorted(
        p for p in TRAIN_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    dense = sorted(
        p for p in EXP04_DENSE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    border = sorted(
        p for p in BORDER_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not original:
        raise RuntimeError("TRAIN original vacío.")
    if not dense:
        raise RuntimeError("EXP04 dense crops no encontrados.")
    if not border:
        raise RuntimeError("EXP06 border crops no encontrados.")

    lines = [
        str(p.resolve()) for p in original
    ]
    lines.extend(
        str(p.resolve()) for p in dense
    )
    lines.extend(
        str(p.resolve()) for p in border
    )

    TRAIN_MANIFEST.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("MANIFEST:")
    print(f"  Original TRAIN: {len(original):,}")
    print(f"  EXP04 crops:    {len(dense):,}")
    print(f"  EXP06 crops:    {len(border):,}")
    print(f"  TOTAL:          {len(lines):,}")

    return len(lines)


def create_yaml() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

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
    print("[OK] YAML EXP06:")
    print(f"     {TEMP_DATA_YAML}")
    print("[INFO] YAML oficial NO modificado.")


# ============================================================================
# TRAIN
# ============================================================================

def train_exp06(model_path: Path) -> Path:
    print()
    print("=" * 72)
    print("ENTRENAMIENTO EXP06")
    print("=" * 72)

    model = YOLO(str(model_path))

    results = model.train(
        data=str(TEMP_DATA_YAML),
        epochs=EPOCHS,
        imgsz=TRAIN_IMAGE_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=SEED,
        amp=AMP,
        cache=CACHE,
        patience=PATIENCE,
        project=str(RUNS_DIR),
        name="exp06_border_targeted_crops",
        pretrained=True,
        save=True,
        save_period=1,
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"

    if not best.exists():
        raise FileNotFoundError(
            f"No se encontró best.pt:\n{best}"
        )

    print()
    print("[OK] BEST MODEL:")
    print(f"     {best}")

    return best


# ============================================================================
# EVALUATION
# ============================================================================

def load_small_person_gt(
    label_path: Path,
    iw: int,
    ih: int,
) -> List[Dict]:
    persons = load_person_boxes(
        label_path,
        iw,
        ih,
    )

    return [
        p for p in persons
        if p["area"] < SMALL_AREA_THRESHOLD
    ]


def match_predictions(
    gt: List[Dict],
    pred_boxes: List[List[float]],
    pred_conf: List[float],
) -> List[Dict]:
    output = []
    used = set()

    for g in gt:
        best_iou = 0.0
        best_idx = None

        for j, box in enumerate(pred_boxes):
            if j in used:
                continue

            current = iou_xyxy(
                g["box"],
                box,
            )

            if current > best_iou:
                best_iou = current
                best_idx = j

        matched = (
            best_idx is not None
            and
            best_iou >= EVAL_MATCH_IOU
        )

        confidence = 0.0

        if matched:
            used.add(best_idx)
            confidence = pred_conf[best_idx]

        output.append(
            {
                "status": "TP" if matched else "FN",
                "iou": best_iou,
                "confidence": confidence,
            }
        )

    return output


def process_test_image(
    model: YOLO,
    image_path: Path,
) -> List[Dict]:
    label_path = (
        TEST_LABELS_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        return []

    try:
        with Image.open(image_path) as image:
            iw, ih = image.width, image.height
    except Exception as exc:
        print(
            f"[WARNING] No se pudo abrir "
            f"{image_path.name}: {exc}"
        )
        return []

    gt = load_small_person_gt(
        label_path,
        iw,
        ih,
    )

    if not gt:
        return []

    try:
        result = model.predict(
            source=str(image_path),
            imgsz=EVAL_IMAGE_SIZE,
            conf=EVAL_CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
            save=False,
        )[0]
    except Exception as exc:
        print(
            f"[ERROR] Inferencia fallida "
            f"{image_path.name}: {exc}"
        )
        return []

    pred_boxes = []
    pred_conf = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()

        for box, conf, cls in zip(
            boxes,
            confs,
            classes,
        ):
            if int(cls) != PERSON_CLASS_ID:
                continue

            pred_boxes.append(box)
            pred_conf.append(float(conf))

    matches = match_predictions(
        gt,
        pred_boxes,
        pred_conf,
    )

    rows = []

    for g, m in zip(gt, matches):
        s = g["size_sqrt"]

        if s < 16:
            bucket = "<16"
        elif s < 32:
            bucket = "16-32"
        elif s < 64:
            bucket = "32-64"
        elif s < 128:
            bucket = "64-128"
        elif s < 256:
            bucket = "128-256"
        else:
            bucket = ">=256"

        rows.append(
            {
                "image": image_path.name,
                "gt_index": g["gt_index"],
                "area": round(g["area"], 6),
                "size_sqrt": round(s, 6),
                "size_bucket": bucket,
                "status": m["status"],
                "iou": round(m["iou"], 6),
                "confidence": round(m["confidence"], 6),
            }
        )

    return rows


def metrics(rows: List[Dict]) -> Dict:
    gt = len(rows)
    tp = sum(
        1 for r in rows if r["status"] == "TP"
    )
    fn = gt - tp
    recall = safe_div(tp, gt)

    return {
        "gt": gt,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "recall_percentage": recall * 100.0,
    }


def size_metrics(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)

    for row in rows:
        groups[row["size_bucket"]].append(row)

    order = [
        "<16",
        "16-32",
        "32-64",
        "64-128",
        "128-256",
        ">=256",
    ]

    result = []

    for bucket in order:
        m = metrics(
            groups.get(
                bucket,
                [],
            )
        )
        result.append(
            {
                "size_bucket": bucket,
                **m,
            }
        )

    return result


# ============================================================================
# REPORTS
# ============================================================================

def write_training_config(
    model_path: Path,
    manifest_count: int,
    crop_stats: Dict,
) -> None:
    rows = [
        {"parameter": "model", "value": "YOLO26s"},
        {"parameter": "pretrained_model", "value": str(model_path)},
        {"parameter": "train_imgsz", "value": TRAIN_IMAGE_SIZE},
        {"parameter": "epochs", "value": EPOCHS},
        {"parameter": "batch", "value": BATCH},
        {"parameter": "workers", "value": WORKERS},
        {"parameter": "seed", "value": SEED},
        {"parameter": "edge_margin_ratio", "value": EDGE_MARGIN_RATIO},
        {
            "parameter": "border_crop_width_ratio",
            "value": BORDER_CROP_WIDTH_RATIO,
        },
        {
            "parameter": "border_crop_height_ratio",
            "value": BORDER_CROP_HEIGHT_RATIO,
        },
        {
            "parameter": "minimum_box_visibility",
            "value": MIN_BOX_VISIBILITY,
        },
        {"parameter": "manifest_images", "value": manifest_count},
        {
            "parameter": "generated_border_crops",
            "value": crop_stats["crops"],
        },
        {
            "parameter": "small_persons_in_border_crops",
            "value": crop_stats["small_in_crops"],
        },
    ]

    write_csv(
        TRAIN_CONFIG_CSV,
        rows,
    )


def write_comparison(exp06: Dict) -> None:
    rows = [
        {
            "experiment": "EXP01",
            "train_imgsz": 640,
            "intervention": "baseline",
            "recall_percentage": 29.68,
        },
        {
            "experiment": "EXP02",
            "train_imgsz": 640,
            "intervention": "oversampling_2x",
            "recall_percentage": 29.62,
        },
        {
            "experiment": "EXP03",
            "train_imgsz": 960,
            "intervention": "high_resolution",
            "recall_percentage": 30.33,
        },
        {
            "experiment": "EXP04",
            "train_imgsz": 960,
            "intervention": "dense_scene_targeted_crops",
            "recall_percentage": 32.72,
        },
        {
            "experiment": "EXP05",
            "train_imgsz": 960,
            "intervention": "dense_plus_neighbor_crops",
            "recall_percentage": 30.47,
        },
        {
            "experiment": "EXP06",
            "train_imgsz": 960,
            "intervention": "dense_plus_border_crops",
            "recall_percentage": exp06["recall_percentage"],
        },
    ]

    for row in rows:
        row["delta_vs_exp01_pp"] = (
            row["recall_percentage"] - 29.68
        )
        row["delta_vs_exp04_pp"] = (
            row["recall_percentage"] - 32.72
        )

    write_csv(
        COMPARISON_CSV,
        rows,
    )


def write_summary(
    crop_stats: Dict,
    m: Dict,
    sm: List[Dict],
    best_path: Path,
) -> None:
    delta4 = m["recall_percentage"] - 32.72

    if delta4 > 1.0:
        interpretation = (
            "MEJORA FUERTE sobre EXP04."
        )
    elif delta4 > 0.5:
        interpretation = (
            "MEJORA MODERADA sobre EXP04."
        )
    elif delta4 >= -0.5:
        interpretation = (
            "SIN CAMBIO RELEVANTE respecto a EXP04."
        )
    else:
        interpretation = (
            "EMPEORAMIENTO respecto a EXP04."
        )

    lines = [
        "=" * 72,
        "SAR YOLO26 - EXP06 BORDER TARGETED CROPS V1",
        "=" * 72,
        "",
        "HIPÓTESIS",
        (
            "Las PERSON pequeñas próximas a los bordes "
            "pueden perder contexto y presentar menor recall."
        ),
        "",
        "CONFIGURACIÓN",
        f"imgsz={TRAIN_IMAGE_SIZE}",
        f"epochs={EPOCHS}",
        f"batch={BATCH}",
        f"workers={WORKERS}",
        f"edge_margin_ratio={EDGE_MARGIN_RATIO}",
        "",
        "CROPS EXP06",
        f"SMALL PERSON TRAIN:     {crop_stats['small']:,}",
        f"SMALL + BORDER:         {crop_stats['small_border']:,}",
        f"BORDER CROPS:           {crop_stats['crops']:,}",
        f"SMALL IN CROPS:         {crop_stats['small_in_crops']:,}",
        f"IMAGE ERRORS:            {crop_stats['errors']:,}",
        "",
        "BORDER DISTRIBUTION",
        f"LEFT:    {crop_stats['left']:,}",
        f"TOP:     {crop_stats['top']:,}",
        f"RIGHT:   {crop_stats['right']:,}",
        f"BOTTOM:  {crop_stats['bottom']:,}",
        "",
        "RESULTADO",
        f"SMALL PERSON GT:         {m['gt']:,}",
        f"SMALL PERSON TP:         {m['tp']:,}",
        f"SMALL PERSON FN:         {m['fn']:,}",
        f"SMALL PERSON Recall:     {m['recall_percentage']:.2f}%",
        "",
        "COMPARACIÓN",
        "EXP01: 29.68%",
        "EXP02: 29.62%",
        "EXP03: 30.33%",
        "EXP04: 32.72%",
        "EXP05: 30.47%",
        f"EXP06: {m['recall_percentage']:.2f}%",
        f"EXP06 - EXP04: {delta4:+.2f} pp",
        "",
        "INTERPRETACIÓN",
        interpretation,
        "",
        "DESGLOSE POR TAMAÑO",
        "-" * 72,
    ]

    for row in sm:
        lines.append(
            f"{row['size_bucket']:>8} "
            f"GT={row['gt']:>6,} "
            f"TP={row['tp']:>6,} "
            f"FN={row['fn']:>6,} "
            f"Recall={row['recall_percentage']:>7.2f}%"
        )

    lines += [
        "",
        "DECISIÓN",
        "-" * 72,
        (
            "CONSERVAR para fase combinada."
            if delta4 > 0.5
            else "NO CONSERVAR como intervención principal."
        ),
        "",
        "IMPORTANTE: dataset original NO modificado.",
        "IMPORTANTE: border crops solo dentro de EXP06.",
        "IMPORTANTE: YAML oficial NO modificado.",
        "",
        "BEST MODEL",
        str(best_path),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print()
    print("=" * 72)
    print("# SAR YOLO26 - EXP06 BORDER TARGETED CROPS V1")
    print("=" * 72)

    validate_structure()

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = find_pretrained_model()

    print()
    print("[OK] PRETRAINED:")
    print(f"     {model_path}")

    crop_stats = generate_border_crops()

    manifest_count = create_manifest()

    create_yaml()

    write_training_config(
        model_path,
        manifest_count,
        crop_stats,
    )

    best_path = train_exp06(
        model_path
    )

    print()
    print("=" * 72)
    print("EVALUACIÓN SMALL PERSON EXP06")
    print("=" * 72)

    model = YOLO(
        str(best_path)
    )

    print(
        f"[OK] Clases: {model.names}"
    )

    test_images = sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not test_images:
        raise RuntimeError("TEST_DEV sin imágenes.")

    rows = []

    for i, image_path in enumerate(
        test_images,
        start=1,
    ):
        rows.extend(
            process_test_image(
                model,
                image_path,
            )
        )

        if i % 100 == 0 or i == len(test_images):
            print(
                f"Analizadas: "
                f"{i:,}/{len(test_images):,} "
                f"| Small GT: {len(rows):,}"
            )

    if not rows:
        raise RuntimeError(
            "No se encontraron SMALL PERSON en TEST_DEV."
        )

    m = metrics(rows)
    sm = size_metrics(rows)

    write_csv(
        OBJECTS_CSV,
        rows,
    )

    write_csv(
        SIZE_CSV,
        sm,
    )

    write_csv(
        EVAL_CSV,
        [
            {"metric": "SMALL_PERSON_GT", "value": m["gt"]},
            {"metric": "SMALL_PERSON_TP", "value": m["tp"]},
            {"metric": "SMALL_PERSON_FN", "value": m["fn"]},
            {"metric": "SMALL_PERSON_RECALL", "value": m["recall"]},
            {
                "metric": "SMALL_PERSON_RECALL_PERCENTAGE",
                "value": m["recall_percentage"],
            },
            {"metric": "TRAIN_IMAGE_SIZE", "value": TRAIN_IMAGE_SIZE},
            {"metric": "EDGE_MARGIN_RATIO", "value": EDGE_MARGIN_RATIO},
            {"metric": "EVAL_IMAGE_SIZE", "value": EVAL_IMAGE_SIZE},
            {"metric": "CONF_THRESHOLD", "value": EVAL_CONF_THRESHOLD},
            {"metric": "MATCH_IOU", "value": EVAL_MATCH_IOU},
        ],
    )

    write_comparison(m)

    write_summary(
        crop_stats,
        m,
        sm,
        best_path,
    )

    print()
    print("=" * 72)
    print("# RESULTADO EXP06")
    print("=" * 72)
    print()
    print(
        f"SMALL PERSON + BORDER: {crop_stats['small_border']:,}"
    )
    print(
        f"BORDER CROPS:          {crop_stats['crops']:,}"
    )
    print()
    print(
        f"SMALL PERSON GT:       {m['gt']:,}"
    )
    print(
        f"SMALL PERSON TP:       {m['tp']:,}"
    )
    print(
        f"SMALL PERSON FN:       {m['fn']:,}"
    )
    print(
        f"SMALL PERSON Recall:   {m['recall_percentage']:.2f}%"
    )
    print()
    print("COMPARACIÓN")
    print("EXP01: 29.68%")
    print("EXP02: 29.62%")
    print("EXP03: 30.33%")
    print("EXP04: 32.72%")
    print("EXP05: 30.47%")
    print(
        f"EXP06: {m['recall_percentage']:.2f}%"
    )
    print()
    print("REPORTS:")
    for p in [
        BORDER_STATS_CSV,
        TRAIN_CONFIG_CSV,
        EVAL_CSV,
        SIZE_CSV,
        OBJECTS_CSV,
        COMPARISON_CSV,
        SUMMARY_TXT,
    ]:
        print(f"[OK] {p}")
    print()
    print(
        "IMPORTANTE: dataset original NO modificado."
    )
    print(
        "IMPORTANTE: YAML oficial NO modificado."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CANCELADO] EXP06 interrumpido.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 72)
        print("[ERROR EXP06]")
        print("=" * 72)
        print()
        print(str(exc))
        print()
        sys.exit(1)
