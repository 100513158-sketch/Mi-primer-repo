from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


# ============================================================================
# SAR YOLO26 - EXP06 VS EXP04 FACTOR COMPARISON V1
# ============================================================================
# No training. Same TEST_DEV. Same IoU/conf protocol.
# Main question: did EXP06 actually improve EDGE_LOCATION vs EXP04?
# ============================================================================

PERSON_CLASS_ID = 0
SMALL_AREA_THRESHOLD = 256.0
EDGE_MARGIN_RATIO = 0.10
DENSE_PERSON_COUNT = 25
NEIGHBOR_DISTANCE_FACTOR = 2.0
MATCH_IOU_THRESHOLD = 0.50
EVAL_IMAGE_SIZE = 1536
CONF_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent


def project_root() -> Path:
    for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if p.name.lower() == "sarc-drone":
            return p
    raise RuntimeError("No se pudo localizar C:\\SARC-Drone.")


PROJECT_ROOT = project_root()
BASELINE = PROJECT_ROOT / "01_training" / "experiments" / "sar_yolo26" / "baseline"
DATASET = (
    PROJECT_ROOT / "00_datasets" / "SAR_DATASET_STUDIO" / "processed" / "sar"
    / "cleaned" / "VisDrone_SAR_2CLASS_V1"
)
TEST_IMAGES = DATASET / "test_dev" / "images"
TEST_LABELS = DATASET / "test_dev" / "labels"

EXP04_MODEL = (
    BASELINE / "training" / "experiments"
    / "exp04_dense_scene_targeted_crops_v1" / "runs"
    / "exp04_dense_scene_targeted_crops" / "weights" / "best.pt"
)
EXP06_MODEL = (
    BASELINE / "training" / "experiments"
    / "exp06_border_targeted_crops_v1" / "runs"
    / "exp06_border_targeted_crops" / "weights" / "best.pt"
)

REPORTS = (
    BASELINE / "evaluation" / "dataset_analysis" / "detection_failure_analysis"
    / "person" / "small_failure_patterns" / "experiments"
    / "exp06_vs_exp04_factor_comparison_v1" / "reports"
)

OBJECTS_CSV = REPORTS / "exp06_vs_exp04_objects_v1.csv"
FACTORS_CSV = REPORTS / "exp06_vs_exp04_factor_metrics_v1.csv"
COMBOS_CSV = REPORTS / "exp06_vs_exp04_factor_combinations_v1.csv"
SUMMARY = REPORTS / "EXP06_VS_EXP04_FACTOR_COMPARISON_V1_SUMMARY.txt"


def safe_div(a, b):
    return 0.0 if b == 0 else a / b


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = aa + ab - inter
    return 0.0 if u <= 0 else inter / u


def box_xyxy(xc, yc, w, h, iw, ih):
    cx, cy = xc * iw, yc * ih
    bw, bh = w * iw, h * ih
    return [
        max(0.0, cx - bw / 2),
        max(0.0, cy - bh / 2),
        min(float(iw), cx + bw / 2),
        min(float(ih), cy + bh / 2),
    ]


def center_distance(a, b):
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def load_small_persons(label_path, iw, ih):
    out = []
    if not label_path.exists():
        return out
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()

    for idx, line in enumerate(lines):
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cls = int(float(p[0]))
            xc, yc, w, h = map(float, p[1:5])
        except ValueError:
            continue
        if cls != PERSON_CLASS_ID or w <= 0 or h <= 0:
            continue
        area = w * h * iw * ih
        if area >= SMALL_AREA_THRESHOLD:
            continue
        box = box_xyxy(xc, yc, w, h, iw, ih)
        out.append({
            "gt_index": idx,
            "box": box,
            "area": area,
            "size_sqrt": math.sqrt(area),
        })
    return out


def factor_flags(persons, target, iw, ih):
    x1, y1, x2, y2 = target["box"]
    mx, my = iw * EDGE_MARGIN_RATIO, ih * EDGE_MARGIN_RATIO
    edge = (
        x1 <= mx or y1 <= my
        or iw - x2 <= mx
        or ih - y2 <= my
    )
    extreme = target["size_sqrt"] < 16.0
    distances = [
        center_distance(target["box"], p["box"])
        for p in persons
        if p["gt_index"] != target["gt_index"]
    ]
    nearest = min(distances) if distances else float("inf")
    close = nearest <= NEIGHBOR_DISTANCE_FACTOR * max(target["size_sqrt"], 1.0)
    dense = len(persons) >= DENSE_PERSON_COUNT
    return {
        "EXTREME_SMALL": int(extreme),
        "EDGE_LOCATION": int(edge),
        "CLOSE_NEIGHBORS": int(close),
        "DENSE_SCENE": int(dense),
    }


def predict(model, image_path):
    r = model.predict(
        source=str(image_path),
        imgsz=EVAL_IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device=0,
        verbose=False,
        save=False,
    )
    if not r or r[0].boxes is None:
        return [], []
    b = r[0].boxes
    boxes = b.xyxy.cpu().tolist()
    confs = b.conf.cpu().tolist()
    classes = b.cls.cpu().tolist()
    keep = [
        (box, float(conf))
        for box, conf, cls in zip(boxes, confs, classes)
        if int(cls) == PERSON_CLASS_ID
    ]
    return [x[0] for x in keep], [x[1] for x in keep]


def match(gt, pred_boxes, pred_confs):
    used = set()
    out = []
    for g in gt:
        best_iou, best_idx = 0.0, None
        for j, pb in enumerate(pred_boxes):
            if j in used:
                continue
            sc = iou(g["box"], pb)
            if sc > best_iou:
                best_iou, best_idx = sc, j
        ok = best_idx is not None and best_iou >= MATCH_IOU_THRESHOLD
        conf = pred_confs[best_idx] if ok else 0.0
        if ok:
            used.add(best_idx)
        out.append({"tp": ok, "iou": best_iou, "conf": conf})
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def factor_metrics(rows):
    names = ["OVERALL", "EXTREME_SMALL", "EDGE_LOCATION", "CLOSE_NEIGHBORS", "DENSE_SCENE"]
    out = []
    for name in names:
        sub = rows if name == "OVERALL" else [r for r in rows if r[name] == 1]
        gt = len(sub)
        tp04 = sum(r["EXP04_TP"] for r in sub)
        tp06 = sum(r["EXP06_TP"] for r in sub)
        r04 = safe_div(tp04, gt) * 100
        r06 = safe_div(tp06, gt) * 100
        out.append({
            "factor": name,
            "gt": gt,
            "EXP04_TP": tp04,
            "EXP04_FN": gt - tp04,
            "EXP04_recall_pct": round(r04, 4),
            "EXP06_TP": tp06,
            "EXP06_FN": gt - tp06,
            "EXP06_recall_pct": round(r06, 4),
            "delta_pp": round(r06 - r04, 4),
            "TP_gain": tp06 - tp04,
        })
    return out


def combo_metrics(rows):
    combos = [
        ("EDGE_LOCATION + EXTREME_SMALL", ["EDGE_LOCATION", "EXTREME_SMALL"]),
        ("EDGE_LOCATION + CLOSE_NEIGHBORS", ["EDGE_LOCATION", "CLOSE_NEIGHBORS"]),
        ("EDGE_LOCATION + DENSE_SCENE", ["EDGE_LOCATION", "DENSE_SCENE"]),
        ("DENSE_SCENE + CLOSE_NEIGHBORS", ["DENSE_SCENE", "CLOSE_NEIGHBORS"]),
        (
            "EDGE_LOCATION + DENSE_SCENE + EXTREME_SMALL",
            ["EDGE_LOCATION", "DENSE_SCENE", "EXTREME_SMALL"],
        ),
    ]
    out = []
    for name, fs in combos:
        sub = [r for r in rows if all(r[f] == 1 for f in fs)]
        if not sub:
            continue
        gt = len(sub)
        tp04 = sum(r["EXP04_TP"] for r in sub)
        tp06 = sum(r["EXP06_TP"] for r in sub)
        r04 = safe_div(tp04, gt) * 100
        r06 = safe_div(tp06, gt) * 100
        out.append({
            "combination": name,
            "gt": gt,
            "EXP04_recall_pct": round(r04, 4),
            "EXP06_recall_pct": round(r06, 4),
            "delta_pp": round(r06 - r04, 4),
            "TP_gain": tp06 - tp04,
        })
    return out


def main():
    print()
    print("=" * 72)
    print("# SAR YOLO26 - EXP06 VS EXP04 FACTOR COMPARISON V1")
    print("=" * 72)

    REPORTS.mkdir(parents=True, exist_ok=True)

    for label, path in {
        "EXP04_MODEL": EXP04_MODEL,
        "EXP06_MODEL": EXP06_MODEL,
        "DATASET_ROOT": DATASET,
        "TEST_IMAGES": TEST_IMAGES,
        "TEST_LABELS": TEST_LABELS,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {label}:\n{path}")
        print(f"[OK] {label}\n     {path}")

    print("\nCargando modelos...")
    m04 = YOLO(str(EXP04_MODEL))
    m06 = YOLO(str(EXP06_MODEL))
    print("[OK] EXP04 cargado.")
    print("[OK] EXP06 cargado.")

    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError("TEST_DEV no contiene imágenes.")

    rows = []

    for i, image_path in enumerate(images, 1):
        label_path = TEST_LABELS / f"{image_path.stem}.txt"
        try:
            with Image.open(image_path) as im:
                iw, ih = im.width, im.height
        except Exception as exc:
            print(f"[WARNING] {image_path.name}: {exc}")
            continue

        gt = load_small_persons(label_path, iw, ih)
        if not gt:
            continue

        factors = [factor_flags(gt, g, iw, ih) for g in gt]

        p04, c04 = predict(m04, image_path)
        p06, c06 = predict(m06, image_path)

        x04 = match(gt, p04, c04)
        x06 = match(gt, p06, c06)

        for g, f, a, b in zip(gt, factors, x04, x06):
            rows.append({
                "image": image_path.name,
                "gt_index": g["gt_index"],
                "area": round(g["area"], 6),
                "size_sqrt": round(g["size_sqrt"], 6),
                "EXP04_TP": int(a["tp"]),
                "EXP04_iou": round(a["iou"], 6),
                "EXP04_conf": round(a["conf"], 6),
                "EXP06_TP": int(b["tp"]),
                "EXP06_iou": round(b["iou"], 6),
                "EXP06_conf": round(b["conf"], 6),
                "delta_tp": int(b["tp"]) - int(a["tp"]),
                **f,
            })

        if i % 100 == 0 or i == len(images):
            print(f"Analizadas: {i:,}/{len(images):,} | SMALL GT: {len(rows):,}")

    if not rows:
        raise RuntimeError("No se encontraron SMALL PERSON.")

    factors = factor_metrics(rows)
    combos = combo_metrics(rows)

    write_csv(OBJECTS_CSV, rows)
    write_csv(FACTORS_CSV, factors)
    write_csv(COMBOS_CSV, combos)

    overall = factors[0]
    edge = next(r for r in factors if r["factor"] == "EDGE_LOCATION")

    lines = [
        "=" * 72,
        "SAR YOLO26 - EXP06 VS EXP04 FACTOR COMPARISON V1",
        "=" * 72,
        "",
        f"TEST SMALL PERSON GT: {overall['gt']:,}",
        "",
        f"GLOBAL EXP04: {overall['EXP04_recall_pct']:.2f}%",
        f"GLOBAL EXP06: {overall['EXP06_recall_pct']:.2f}%",
        f"GLOBAL DELTA:  {overall['delta_pp']:+.2f} pp",
        f"GLOBAL TP GAIN:{overall['TP_gain']:+d}",
        "",
        f"EDGE EXP04:   {edge['EXP04_recall_pct']:.2f}%",
        f"EDGE EXP06:   {edge['EXP06_recall_pct']:.2f}%",
        f"EDGE DELTA:   {edge['delta_pp']:+.2f} pp",
        f"EDGE TP GAIN: {edge['TP_gain']:+d}",
        "",
        "FACTORES",
        "-" * 72,
    ]

    for r in factors:
        lines.append(
            f"{r['factor']:<20} GT={r['gt']:>6,} "
            f"EXP04={r['EXP04_recall_pct']:>7.2f}% "
            f"EXP06={r['EXP06_recall_pct']:>7.2f}% "
            f"Δ={r['delta_pp']:+7.2f} pp "
            f"TPgain={r['TP_gain']:+d}"
        )

    lines += ["", "COMBINACIONES", "-" * 72]
    for r in combos:
        lines.append(
            f"{r['combination']:<42} GT={r['gt']:>6,} "
            f"EXP04={r['EXP04_recall_pct']:>7.2f}% "
            f"EXP06={r['EXP06_recall_pct']:>7.2f}% "
            f"Δ={r['delta_pp']:+7.2f} pp"
        )

    lines += [
        "",
        "DECISIÓN EDGE_LOCATION",
        "-" * 72,
    ]

    if edge["delta_pp"] > 0.5:
        lines += [
            "EXP06 mejora claramente el factor EDGE_LOCATION.",
            "Border crops quedan respaldados para la fase combinada.",
        ]
    elif edge["delta_pp"] >= -0.5:
        lines += [
            "EXP06 queda prácticamente empatado con EXP04.",
            "No hay evidencia fuerte de mejora específica de EDGE_LOCATION.",
        ]
    else:
        lines += [
            "EXP06 empeora EDGE_LOCATION frente a EXP04.",
            "Border crops no deben considerarse intervención ganadora.",
        ]

    lines += [
        "",
        "IMPORTANTE: no se modificaron dataset, labels ni YAML oficial.",
    ]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 72)
    print("# RESULTADO EXP06 VS EXP04")
    print("=" * 72)
    print(f"GLOBAL: EXP04={overall['EXP04_recall_pct']:.2f}% "
          f"EXP06={overall['EXP06_recall_pct']:.2f}% "
          f"Δ={overall['delta_pp']:+.2f} pp")
    print(f"EDGE:   EXP04={edge['EXP04_recall_pct']:.2f}% "
          f"EXP06={edge['EXP06_recall_pct']:.2f}% "
          f"Δ={edge['delta_pp']:+.2f} pp")
    print()
    print("REPORTS")
    print(f"[OK] {OBJECTS_CSV}")
    print(f"[OK] {FACTORS_CSV}")
    print(f"[OK] {COMBOS_CSV}")
    print(f"[OK] {SUMMARY}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CANCELADO]")
        sys.exit(130)
    except Exception as exc:
        print("\n[ERROR]")
        print(str(exc))
        sys.exit(1)
