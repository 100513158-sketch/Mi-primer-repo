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
# SAR YOLO26 - EXP05 NEIGHBOR SEPARATION V1
# ============================================================================
# EXP05 = EXP04 (imgsz=960 + dense crops) + close-neighbor crops.
# The original dataset and official YAML are never modified.
# ============================================================================

PERSON_CLASS_ID = 0
SMALL_AREA_THRESHOLD = 256.0
DENSE_PERSON_THRESHOLD = 25
NEIGHBOR_DISTANCE_MULTIPLIER = 3.0
MIN_NEIGHBOR_REFERENCE_SIZE = 16.0
NEIGHBOR_CROP_WIDTH_RATIO = 0.40
NEIGHBOR_CROP_HEIGHT_RATIO = 0.40
MIN_BOX_VISIBILITY = 0.35

TRAIN_IMAGE_SIZE = 960
EPOCHS = 100
BATCH = 8
WORKERS = 8
DEVICE = 0
SEED = 42
AMP = True
PATIENCE = 20
CACHE = False

EVAL_IMAGE_SIZE = 1536
EVAL_CONF_THRESHOLD = 0.25
EVAL_MATCH_IOU = 0.50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def find_project_root() -> Path:
    for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if p.name.lower() == "sarc-drone":
            return p
    raise RuntimeError(f"No se pudo localizar C:\\SARC-Drone.\nScript: {SCRIPT_PATH}")


PROJECT_ROOT = find_project_root()
BASELINE_DIR = PROJECT_ROOT / "01_training" / "experiments" / "sar_yolo26" / "baseline"
DATASET_ROOT = PROJECT_ROOT / "00_datasets" / "SAR_DATASET_STUDIO" / "processed" / "sar" / "cleaned" / "VisDrone_SAR_2CLASS_V1"
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
VAL_IMAGES_DIR = DATASET_ROOT / "val" / "images"
TEST_IMAGES_DIR = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test_dev" / "labels"

EXPERIMENT_ROOT = BASELINE_DIR / "training" / "experiments" / "exp05_neighbor_separation_v1"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
NEIGHBOR_CROPS_DIR = EXPERIMENT_ROOT / "neighbor_crops"
NEIGHBOR_IMAGES_DIR = NEIGHBOR_CROPS_DIR / "images"
NEIGHBOR_LABELS_DIR = NEIGHBOR_CROPS_DIR / "labels"
TRAIN_MANIFEST = EXPERIMENT_ROOT / "train_with_dense_and_neighbor_crops.txt"
TEMP_DATA_YAML = EXPERIMENT_ROOT / "exp05_dataset.yaml"

EXP04_ROOT = BASELINE_DIR / "training" / "experiments" / "exp04_dense_scene_targeted_crops_v1"
EXP04_DENSE_IMAGES_DIR = EXP04_ROOT / "dense_crops" / "images"

REPORTS_DIR = BASELINE_DIR / "evaluation" / "dataset_analysis" / "detection_failure_analysis" / "person" / "small_failure_patterns" / "experiments" / "exp05_neighbor_separation_v1" / "reports"
NEIGHBOR_STATS_CSV = REPORTS_DIR / "exp05_neighbor_crop_statistics_v1.csv"
TRAIN_CONFIG_CSV = REPORTS_DIR / "exp05_training_configuration_v1.csv"
EVAL_CSV = REPORTS_DIR / "exp05_small_person_recall_v1.csv"
SIZE_CSV = REPORTS_DIR / "exp05_small_person_recall_by_size_v1.csv"
OBJECTS_CSV = REPORTS_DIR / "exp05_small_person_objects_v1.csv"
COMPARISON_CSV = REPORTS_DIR / "exp05_vs_exp01_exp02_exp03_exp04_v1.csv"
SUMMARY_TXT = REPORTS_DIR / "EXP05_NEIGHBOR_SEPARATION_V1_SUMMARY.txt"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return 0.0 if union <= 0 else inter / union


def center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_distance(a: List[float], b: List[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def xywhn_to_xyxy(xc, yc, w, h, iw, ih) -> List[float]:
    cx, cy = xc * iw, yc * ih
    bw, bh = w * iw, h * ih
    return [
        max(0.0, cx - bw / 2.0),
        max(0.0, cy - bh / 2.0),
        min(float(iw), cx + bw / 2.0),
        min(float(ih), cy + bh / 2.0),
    ]


def is_dense_scene(person_count: int) -> bool:
    return person_count >= DENSE_PERSON_THRESHOLD


def find_pretrained_model() -> Path:
    candidates = [
        BASELINE_DIR / "yolo26s.pt",
        BASELINE_DIR / "training" / "models" / "pretrained" / "yolo26s.pt",
        PROJECT_ROOT / "01_training" / "models" / "pretrained" / "yolo26s.pt",
        PROJECT_ROOT / "yolo26s.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    found = []
    for root in [PROJECT_ROOT / "01_training", PROJECT_ROOT]:
        if not root.exists():
            continue
        try:
            found.extend(root.rglob("yolo26s.pt"))
        except PermissionError:
            pass
    found = sorted({p for p in found if p.is_file()}, key=lambda p: str(p).lower())
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError("Se encontraron varias copias de yolo26s.pt:\n" + "\n".join(f"  - {p}" for p in found))
    raise FileNotFoundError("No se encontró yolo26s.pt.")


def validate_structure() -> Path:
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
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {name}:\n{path}")
        print(f"[OK] {name}\n     {path}")
    dense = [p for p in EXP04_DENSE_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not dense:
        raise RuntimeError(f"EXP04 no contiene dense crops válidos: {EXP04_DENSE_IMAGES_DIR}")
    model = find_pretrained_model()
    print(f"[OK] PRETRAINED MODEL\n     {model}")
    print(f"[OK] EXP04 DENSE CROPS: {len(dense):,}")
    return model


def load_person_boxes(label_path: Path, iw: int, ih: int) -> List[Dict]:
    rows = []
    if not label_path.exists():
        return rows
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0])); xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        except ValueError:
            continue
        if cls != PERSON_CLASS_ID or w <= 0 or h <= 0:
            continue
        box = xywhn_to_xyxy(xc, yc, w, h, iw, ih)
        area = w * h * iw * ih
        rows.append({"gt_index": idx, "box": box, "area": area, "size_sqrt": math.sqrt(max(area, 0.0))})
    return rows


def neighbor_threshold(target: Dict) -> float:
    return NEIGHBOR_DISTANCE_MULTIPLIER * max(float(target["size_sqrt"]), MIN_NEIGHBOR_REFERENCE_SIZE)


def find_close_neighbors(target: Dict, persons: List[Dict]) -> List[Tuple[Dict, float]]:
    threshold = neighbor_threshold(target)
    found = []
    for candidate in persons:
        if candidate["gt_index"] == target["gt_index"]:
            continue
        d = center_distance(target["box"], candidate["box"])
        if d <= threshold:
            found.append((candidate, d))
    found.sort(key=lambda x: x[1])
    return found


def group_crop_box(target: Dict, neighbors: List[Tuple[Dict, float]], iw: int, ih: int) -> Tuple[int, int, int, int]:
    members = [target] + [x[0] for x in neighbors]
    centers = [center(m["box"]) for m in members]
    cx = sum(x for x, _ in centers) / len(centers)
    cy = sum(y for _, y in centers) / len(centers)
    cw = min(int(round(iw * NEIGHBOR_CROP_WIDTH_RATIO)), iw)
    ch = min(int(round(ih * NEIGHBOR_CROP_HEIGHT_RATIO)), ih)
    x1 = int(clamp(round(cx - cw / 2.0), 0, iw - cw))
    y1 = int(clamp(round(cy - ch / 2.0), 0, ih - ch))
    return x1, y1, x1 + cw, y1 + ch


def clip_box(box: List[float], crop: Tuple[int, int, int, int]) -> Tuple[List[float], float]:
    bx1, by1, bx2, by2 = box; cx1, cy1, cx2, cy2 = crop
    x1, y1, x2, y2 = max(bx1, cx1), max(by1, cy1), min(bx2, cx2), min(by2, cy2)
    original = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    visible = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return [x1, y1, x2, y2], safe_div(visible, original)


def box_to_crop_yolo(box: List[float], crop: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    x1 = clamp(box[0] - cx1, 0, cw); y1 = clamp(box[1] - cy1, 0, ch)
    x2 = clamp(box[2] - cx1, 0, cw); y2 = clamp(box[3] - cy1, 0, ch)
    w, h = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0 or w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return ((x1 + w / 2) / cw, (y1 + h / 2) / ch, w / cw, h / ch)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def generate_neighbor_crops() -> Dict:
    print("\n" + "=" * 72)
    print("GENERANDO CROPS DE PERSONAS CON VECINOS PRÓXIMOS")
    print("=" * 72 + "\n")

    if NEIGHBOR_CROPS_DIR.exists():
        shutil.rmtree(NEIGHBOR_CROPS_DIR)
    NEIGHBOR_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    NEIGHBOR_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in TRAIN_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    small_total = small_neighbors = relations = crops = small_in_crops = processed = errors = 0
    stats = []

    for n, image_path in enumerate(images, 1):
        label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                iw, ih = image.width, image.height
                persons = load_person_boxes(label_path, iw, ih)
                processed += 1
                smalls = [p for p in persons if p["area"] < SMALL_AREA_THRESHOLD]
                small_total += len(smalls)
                for target in smalls:
                    neighbors = find_close_neighbors(target, persons)
                    if not neighbors:
                        continue
                    small_neighbors += 1
                    relations += len(neighbors)
                    crop = group_crop_box(target, neighbors, iw, ih)
                    crop_img = image.crop(crop)
                    crop_name = f"{image_path.stem}__neighbor_{target['gt_index']}.jpg"
                    crop_img_path = NEIGHBOR_IMAGES_DIR / crop_name
                    crop_img.save(crop_img_path, quality=95)
                    label_lines = []
                    small_here = 0
                    crop_w, crop_h = crop[2] - crop[0], crop[3] - crop[1]
                    for person in persons:
                        clipped, visibility = clip_box(person["box"], crop)
                        if visibility < MIN_BOX_VISIBILITY:
                            continue
                        xc, yc, w, h = box_to_crop_yolo(clipped, crop)
                        if w <= 0 or h <= 0:
                            continue
                        area = w * h * crop_w * crop_h
                        if area < SMALL_AREA_THRESHOLD:
                            small_here += 1
                        label_lines.append(f"{PERSON_CLASS_ID} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")
                    if not label_lines:
                        crop_img_path.unlink(missing_ok=True)
                        continue
                    (NEIGHBOR_LABELS_DIR / f"{image_path.stem}__neighbor_{target['gt_index']}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                    crops += 1
                    small_in_crops += small_here
                    stats.append({
                        "source_image": image_path.name,
                        "target_gt_index": target["gt_index"],
                        "source_person_count": len(persons),
                        "is_dense_scene": int(is_dense_scene(len(persons))),
                        "neighbor_count": len(neighbors),
                        "nearest_neighbor_distance_px": round(neighbors[0][1], 4),
                        "neighbor_threshold_px": round(neighbor_threshold(target), 4),
                        "target_size_sqrt": round(target["size_sqrt"], 4),
                        "crop_image": crop_name,
                        "crop_width": crop_w,
                        "crop_height": crop_h,
                        "persons_in_crop": len(label_lines),
                        "small_persons_in_crop": small_here,
                    })
        except Exception as exc:
            errors += 1
            print(f"[WARNING] {image_path.name}: {exc}")

        if n % 500 == 0 or n == len(images):
            print(f"Procesadas: {n:,}/{len(images):,} | Small: {small_total:,} | Small+neighbor: {small_neighbors:,} | Crops: {crops:,}")

    print(f"\nImágenes TRAIN procesadas: {processed:,}")
    print(f"SMALL PERSON: {small_total:,}")
    print(f"SMALL PERSON con vecinos: {small_neighbors:,}")
    print(f"Relaciones de vecindad: {relations:,}")
    print(f"Crops generados: {crops:,}")
    print(f"Small PERSON en crops: {small_in_crops:,}")
    print(f"Errores de imágenes: {errors:,}")

    if not stats:
        raise RuntimeError("No se generó ningún crop de vecinos.")

    write_csv(NEIGHBOR_STATS_CSV, stats)
    return {
        "small_persons": small_total,
        "small_with_neighbors": small_neighbors,
        "neighbor_relations": relations,
        "generated_crops": crops,
        "small_persons_in_crops": small_in_crops,
        "errors": errors,
    }


def create_manifest() -> int:
    originals = sorted(p for p in TRAIN_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    dense = sorted(p for p in EXP04_DENSE_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    neighbor = sorted(p for p in NEIGHBOR_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not originals or not dense or not neighbor:
        raise RuntimeError(f"Manifest incompleto: originals={len(originals)}, dense={len(dense)}, neighbor={len(neighbor)}")
    lines = [str(p.resolve()) for p in originals + dense + neighbor]
    TRAIN_MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Original TRAIN: {len(originals):,}")
    print(f"[OK] EXP04 dense crops: {len(dense):,}")
    print(f"[OK] EXP05 neighbor crops: {len(neighbor):,}")
    print(f"[OK] Manifest total: {len(lines):,}")
    return len(lines)


def create_yaml() -> None:
    content = f"""path: {DATASET_ROOT.as_posix()}\n\ntrain: {TRAIN_MANIFEST.as_posix()}\nval: {VAL_IMAGES_DIR.as_posix()}\ntest: {TEST_IMAGES_DIR.as_posix()}\n\nnames:\n  0: person\n  1: vehicle\n"""
    TEMP_DATA_YAML.write_text(content, encoding="utf-8")
    print(f"[OK] YAML EXP05: {TEMP_DATA_YAML}")
    print("[INFO] YAML oficial NO modificado.")


def train(model_path: Path) -> Path:
    print("\n" + "=" * 72)
    print("ENTRENAMIENTO EXP05")
    print("=" * 72)
    model = YOLO(str(model_path))
    try:
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
            name="exp05_neighbor_separation",
            pretrained=True,
            save=True,
            plots=True,
            verbose=True,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(f"EXP05 terminó por OOM con imgsz={TRAIN_IMAGE_SIZE}, batch={BATCH}. No se cambió automáticamente el batch.") from exc
        raise
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"No se encontró best.pt: {best}")
    return best


def load_small_gt(label_path: Path, iw: int, ih: int) -> List[Dict]:
    rows = []
    if not label_path.exists():
        return rows
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()
    for idx, line in enumerate(lines):
        p = line.strip().split()
        if len(p) < 5:
            continue
        try:
            cls = int(float(p[0])); xc = float(p[1]); yc = float(p[2]); w = float(p[3]); h = float(p[4])
        except ValueError:
            continue
        if cls != PERSON_CLASS_ID or w <= 0 or h <= 0:
            continue
        area = w * h * iw * ih
        if area >= SMALL_AREA_THRESHOLD:
            continue
        rows.append({"gt_index": idx, "box": xywhn_to_xyxy(xc, yc, w, h, iw, ih), "area": area, "size_sqrt": math.sqrt(max(area, 0.0))})
    return rows


def match_predictions(gts: List[Dict], boxes: List[List[float]], confs: List[float]) -> List[Dict]:
    out = []
    used = set()
    for gt in gts:
        best_iou = 0.0; best_idx = None
        for i, pred in enumerate(boxes):
            if i in used:
                continue
            val = iou_xyxy(gt["box"], pred)
            if val > best_iou:
                best_iou = val; best_idx = i
        matched = best_idx is not None and best_iou >= EVAL_MATCH_IOU
        conf = 0.0
        if matched:
            used.add(best_idx); conf = confs[best_idx]
        out.append({"matched": matched, "iou": best_iou, "confidence": conf})
    return out


def size_bucket(s: float) -> str:
    if s < 16: return "<16"
    if s < 32: return "16-32"
    if s < 64: return "32-64"
    if s < 128: return "64-128"
    if s < 256: return "128-256"
    return ">=256"


def process_test_image(model: YOLO, image_path: Path) -> List[Dict]:
    label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []
    try:
        with Image.open(image_path) as image:
            iw, ih = image.width, image.height
    except Exception as exc:
        print(f"[WARNING] {image_path.name}: {exc}")
        return []
    gts = load_small_gt(label_path, iw, ih)
    if not gts:
        return []
    try:
        result = model.predict(source=str(image_path), imgsz=EVAL_IMAGE_SIZE, conf=EVAL_CONF_THRESHOLD, device=DEVICE, verbose=False, save=False)[0]
    except Exception as exc:
        print(f"[ERROR] Inferencia {image_path.name}: {exc}")
        return []
    boxes, confs = [], []
    if result.boxes is not None:
        for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            if int(cls) == PERSON_CLASS_ID:
                boxes.append(box); confs.append(float(conf))
    matches = match_predictions(gts, boxes, confs)
    return [
        {
            "image": image_path.name,
            "gt_index": gt["gt_index"],
            "area": round(gt["area"], 6),
            "size_sqrt": round(gt["size_sqrt"], 6),
            "size_bucket": size_bucket(gt["size_sqrt"]),
            "status": "TP" if m["matched"] else "FN",
            "iou": round(m["iou"], 6),
            "confidence": round(m["confidence"], 6),
        }
        for gt, m in zip(gts, matches)
    ]


def metrics(rows: List[Dict]) -> Dict:
    gt = len(rows); tp = sum(1 for r in rows if r["status"] == "TP"); fn = gt - tp
    recall = safe_div(tp, gt)
    return {"gt": gt, "tp": tp, "fn": fn, "recall": recall, "recall_percentage": recall * 100.0}


def size_metrics(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for r in rows: groups[r["size_bucket"]].append(r)
    out = []
    for bucket in ["<16", "16-32", "32-64", "64-128", "128-256", ">=256"]:
        out.append({"size_bucket": bucket, **metrics(groups.get(bucket, []))})
    return out


def write_reports(crop_stats: Dict, exp05: Dict, sizes: List[Dict], best: Path) -> None:
    write_csv(EVAL_CSV, [
        {"metric": "SMALL_PERSON_GT", "value": exp05["gt"]},
        {"metric": "SMALL_PERSON_TP", "value": exp05["tp"]},
        {"metric": "SMALL_PERSON_FN", "value": exp05["fn"]},
        {"metric": "SMALL_PERSON_RECALL", "value": exp05["recall"]},
        {"metric": "SMALL_PERSON_RECALL_PERCENTAGE", "value": exp05["recall_percentage"]},
        {"metric": "TRAIN_IMAGE_SIZE", "value": TRAIN_IMAGE_SIZE},
        {"metric": "DENSE_PERSON_THRESHOLD", "value": DENSE_PERSON_THRESHOLD},
        {"metric": "NEIGHBOR_DISTANCE_MULTIPLIER", "value": NEIGHBOR_DISTANCE_MULTIPLIER},
        {"metric": "EVAL_IMAGE_SIZE", "value": EVAL_IMAGE_SIZE},
        {"metric": "CONF_THRESHOLD", "value": EVAL_CONF_THRESHOLD},
        {"metric": "MATCH_IOU", "value": EVAL_MATCH_IOU},
    ])
    write_csv(SIZE_CSV, sizes)
    write_csv(COMPARISON_CSV, [
        {"experiment":"EXP01", "train_imgsz":640, "intervention":"baseline", "recall_percentage":29.68, "delta_vs_exp04_pp":29.68-32.72},
        {"experiment":"EXP02", "train_imgsz":640, "intervention":"oversampling_2x", "recall_percentage":29.62, "delta_vs_exp04_pp":29.62-32.72},
        {"experiment":"EXP03", "train_imgsz":960, "intervention":"high_resolution", "recall_percentage":30.33, "delta_vs_exp04_pp":30.33-32.72},
        {"experiment":"EXP04", "train_imgsz":960, "intervention":"dense_scene_targeted_crops", "recall_percentage":32.72, "delta_vs_exp04_pp":0.0},
        {"experiment":"EXP05", "train_imgsz":960, "intervention":"dense_crops_plus_neighbor_crops", "recall_percentage":exp05["recall_percentage"], "delta_vs_exp04_pp":exp05["recall_percentage"]-32.72},
    ])
    rows = [
        {"parameter":"model", "value":"YOLO26s"},
        {"parameter":"train_imgsz", "value":TRAIN_IMAGE_SIZE},
        {"parameter":"epochs", "value":EPOCHS},
        {"parameter":"batch", "value":BATCH},
        {"parameter":"dense_person_threshold", "value":DENSE_PERSON_THRESHOLD},
        {"parameter":"neighbor_distance_multiplier", "value":NEIGHBOR_DISTANCE_MULTIPLIER},
        {"parameter":"neighbor_crop_width_ratio", "value":NEIGHBOR_CROP_WIDTH_RATIO},
        {"parameter":"neighbor_crop_height_ratio", "value":NEIGHBOR_CROP_HEIGHT_RATIO},
        {"parameter":"manifest_images", "value":crop_stats["manifest_images"]},
        {"parameter":"neighbor_crops", "value":crop_stats["generated_crops"]},
        {"parameter":"small_persons_in_neighbor_crops", "value":crop_stats["small_persons_in_crops"]},
    ]
    write_csv(TRAIN_CONFIG_CSV, rows)

    delta = exp05["recall_percentage"] - 32.72
    if delta > 1.0: decision = "MEJORA FUERTE sobre EXP04."
    elif delta > 0.5: decision = "MEJORA MODERADA sobre EXP04."
    elif delta >= -0.5: decision = "SIN CAMBIO RELEVANTE sobre EXP04."
    else: decision = "EMPEORAMIENTO respecto a EXP04."

    text = "\n".join([
        "=" * 72,
        "SAR YOLO26 - EXP05 NEIGHBOR SEPARATION V1",
        "=" * 72,
        "",
        f"Modelo best: {best}",
        f"Dataset: {DATASET_ROOT}",
        "",
        "CRITERIO PROXIMIDAD",
        f"distance <= {NEIGHBOR_DISTANCE_MULTIPLIER:.1f} * max(size_sqrt, {MIN_NEIGHBOR_REFERENCE_SIZE:.0f})",
        "",
        "CROPS",
        f"Small PERSON train: {crop_stats['small_persons']:,}",
        f"Small + neighbor: {crop_stats['small_with_neighbors']:,}",
        f"Neighbor relations: {crop_stats['neighbor_relations']:,}",
        f"Neighbor crops: {crop_stats['generated_crops']:,}",
        f"Small in crops: {crop_stats['small_persons_in_crops']:,}",
        "",
        "RESULTADO",
        f"SMALL PERSON GT: {exp05['gt']:,}",
        f"SMALL PERSON TP: {exp05['tp']:,}",
        f"SMALL PERSON FN: {exp05['fn']:,}",
        f"SMALL PERSON Recall: {exp05['recall_percentage']:.2f}%",
        "",
        "COMPARACIÓN",
        "EXP01: 29.68%",
        "EXP02: 29.62%",
        "EXP03: 30.33%",
        "EXP04: 32.72%",
        f"EXP05: {exp05['recall_percentage']:.2f}%",
        f"EXP05 - EXP04: {delta:+.2f} pp",
        "",
        "DECISIÓN",
        decision,
        "",
        "IMPORTANTE: dataset original NO modificado.",
        "IMPORTANTE: YAML oficial NO modificado.",
    ])
    SUMMARY_TXT.write_text(text, encoding="utf-8")


def main() -> None:
    print("\n" + "=" * 72)
    print("# SAR YOLO26 - EXP05 NEIGHBOR SEPARATION V1")
    print("=" * 72)
    print(f"\nSCRIPT:\n  {SCRIPT_PATH}")
    print(f"\nDATASET:\n  {DATASET_ROOT}")
    print(f"\nTRAIN IMG SIZE: {TRAIN_IMAGE_SIZE}")
    print(f"NEIGHBOR CRITERION: distance <= {NEIGHBOR_DISTANCE_MULTIPLIER:.1f} * max(size_sqrt, {MIN_NEIGHBOR_REFERENCE_SIZE:.0f})")

    model_path = validate_structure()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    crop_stats = generate_neighbor_crops()
    manifest_count = create_manifest()
    crop_stats["manifest_images"] = manifest_count
    create_yaml()

    best = train(model_path)
    print(f"\n[OK] BEST MODEL:\n     {best}")

    print("\n" + "=" * 72)
    print("EVALUACIÓN SMALL PERSON EXP05")
    print("=" * 72)

    yolo = YOLO(str(best))
    images = sorted(p for p in TEST_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError("No se encontraron imágenes TEST_DEV.")

    rows = []
    for n, image_path in enumerate(images, 1):
        rows.extend(process_test_image(yolo, image_path))
        if n % 100 == 0 or n == len(images):
            print(f"Analizadas: {n:,}/{len(images):,} | Small GT: {len(rows):,}")

    if not rows:
        raise RuntimeError("No se encontraron SMALL PERSON en TEST_DEV.")

    exp05 = metrics(rows)
    sizes = size_metrics(rows)

    write_csv(OBJECTS_CSV, rows)
    write_reports(crop_stats, exp05, sizes, best)

    print("\n" + "=" * 72)
    print("# RESULTADO EXP05")
    print("=" * 72)
    print(f"\nSMALL PERSON con vecinos: {crop_stats['small_with_neighbors']:,}")
    print(f"Crops de vecinos:         {crop_stats['generated_crops']:,}")
    print(f"SMALL PERSON GT:          {exp05['gt']:,}")
    print(f"SMALL PERSON TP:          {exp05['tp']:,}")
    print(f"SMALL PERSON FN:          {exp05['fn']:,}")
    print(f"SMALL PERSON Recall:      {exp05['recall_percentage']:.2f}%")
    print("\nCOMPARACIÓN")
    print("EXP01: 29.68%")
    print("EXP02: 29.62%")
    print("EXP03: 30.33%")
    print("EXP04: 32.72%")
    print(f"EXP05: {exp05['recall_percentage']:.2f}%")
    print("\nREPORTS:")
    for p in [NEIGHBOR_STATS_CSV, TRAIN_CONFIG_CSV, EVAL_CSV, SIZE_CSV, OBJECTS_CSV, COMPARISON_CSV, SUMMARY_TXT]:
        print(f"[OK] {p}")
    print("\nIMPORTANTE: dataset original NO modificado.")
    print("IMPORTANTE: YAML oficial NO modificado.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CANCELADO] EXP05 interrumpido.")
        sys.exit(130)
    except Exception as exc:
        print("\n" + "=" * 72)
        print("[ERROR EXP05]")
        print("=" * 72)
        print(f"\n{exc}\n")
        sys.exit(1)
