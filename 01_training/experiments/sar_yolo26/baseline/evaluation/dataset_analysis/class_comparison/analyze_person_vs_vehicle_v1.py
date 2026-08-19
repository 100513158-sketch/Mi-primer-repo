from pathlib import Path
import csv
import math
from collections import Counter, defaultdict

DATASET_ROOT = Path(r"C:\SARC-Drone\00_datasets\SAR_DATASET_STUDIO\processed\sar\cleaned\VisDrone_SAR_2CLASS_V1")
OUTPUT_ROOT = Path(r"C:\SARC-Drone\01_training\experiments\sar_yolo26\baseline\evaluation\dataset_analysis\class_comparison\person_vs_vehicle_v1")

SPLITS = ["train", "val", "test_dev"]
CLASSES = {0: "person", 1: "vehicle"}
AREA_BINS = [
    ("<16", 0, 16), ("16-32", 16, 32), ("32-64", 32, 64),
    ("64-128", 64, 128), ("128-256", 128, 256),
    ("256-512", 256, 512), ("512-1024", 512, 1024),
    (">=1024", 1024, float("inf")),
]
BORDER_FRAC = 0.02
EPS = 1e-6


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None


def percentile(values, p):
    if not values:
        return 0.0
    a = sorted(values)
    k = (len(a) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(a[lo])
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


def bin_area(area):
    for name, low, high in AREA_BINS:
        if low <= area < high:
            return name
    return "unknown"


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    object_csv = OUTPUT_ROOT / "person_vs_vehicle_objects_v1.csv"
    image_csv = OUTPUT_ROOT / "person_vs_vehicle_images_v1.csv"
    split_csv = OUTPUT_ROOT / "person_vs_vehicle_split_statistics_v1.csv"
    summary = OUTPUT_ROOT / "PERSON_VS_VEHICLE_V1_SUMMARY.txt"

    stats = {}
    for name in CLASSES.values():
        stats[name] = {
            "objects": 0, "images": set(), "areas": [], "widths": [],
            "heights": [], "ratios": [], "partial": 0, "border": 0,
            "bins": Counter(), "split_objects": Counter(),
            "crowding": Counter(),
        }

    split = defaultdict(lambda: defaultdict(lambda: {
        "images": 0, "images_with_class": set(), "objects": 0,
        "areas": [], "partial": 0, "border": 0, "bins": Counter()
    }))

    object_rows = []
    image_rows = []
    total_images = total_objects = corrupt = missing_labels = 0

    print("=" * 72)
    print("# SAR YOLO26 - PERSON VS VEHICLE ANALYSIS V1")
    print("=" * 72)
    print("\nDataset:")
    print(DATASET_ROOT)
    print("\nOutput:")
    print(OUTPUT_ROOT)

    for sp in SPLITS:
        spdir = DATASET_ROOT / sp
        idir = spdir / "images"
        ldir = spdir / "labels"
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(p for p in idir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        if not images:
            print(f"\n[INFO] Split no encontrado o vacío: {sp}")
            continue

        split[sp]["_meta"]["images"] = len(images)
        print(f"\n## Analizando: {sp}")
        print(f"Imágenes encontradas: {len(images)}")

        for i, img in enumerate(images, 1):
            total_images += 1
            size = image_size(img)
            if size is None:
                corrupt += 1
                continue
            iw, ih = size
            rel = img.relative_to(idir)
            lab = (ldir / rel).with_suffix(".txt")
            counts = Counter()

            if not lab.exists():
                missing_labels += 1
                lines = []
            else:
                try:
                    lines = lab.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception:
                    lines = []

            for line_no, line in enumerate(lines, 1):
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    cls = int(float(p[0]))
                    xc, yc, nw, nh = map(float, p[1:5])
                except ValueError:
                    continue
                if cls not in CLASSES:
                    continue

                name = CLASSES[cls]
                bw, bh = max(0, nw * iw), max(0, nh * ih)
                area = bw * bh
                x1, y1 = (xc - nw / 2) * iw, (yc - nh / 2) * ih
                x2, y2 = (xc + nw / 2) * iw, (yc + nh / 2) * ih
                partial = x1 < -EPS or y1 < -EPS or x2 > iw + EPS or y2 > ih + EPS
                border = (
                    x1 <= iw * BORDER_FRAC or y1 <= ih * BORDER_FRAC or
                    x2 >= iw * (1 - BORDER_FRAC) or y2 >= ih * (1 - BORDER_FRAC)
                )
                ratio = bw / bh if bh > 0 else 0
                ab = bin_area(area)

                counts[name] += 1
                s = stats[name]
                s["objects"] += 1
                s["images"].add(str(img))
                s["areas"].append(area)
                s["widths"].append(bw)
                s["heights"].append(bh)
                s["ratios"].append(ratio)
                s["bins"][ab] += 1
                s["split_objects"][sp] += 1
                s["partial"] += int(partial)
                s["border"] += int(border)

                ss = split[sp][name]
                ss["objects"] += 1
                ss["images_with_class"].add(str(img))
                ss["areas"].append(area)
                ss["bins"][ab] += 1
                ss["partial"] += int(partial)
                ss["border"] += int(border)

                object_rows.append({
                    "split": sp, "image": str(img), "label": str(lab),
                    "class_id": cls, "class": name,
                    "image_width": iw, "image_height": ih,
                    "bbox_width_px": round(bw, 4), "bbox_height_px": round(bh, 4),
                    "bbox_area_px2": round(area, 4), "aspect_ratio": round(ratio, 6),
                    "area_bin": ab, "partial_bbox": int(partial),
                    "near_border": int(border),
                })

            for name in CLASSES.values():
                c = counts[name]
                for t in (10, 20, 50, 100):
                    if c >= t:
                        stats[name]["crowding"][t] += 1

            image_rows.append({
                "split": sp, "image": str(img), "image_width": iw, "image_height": ih,
                "person_objects": counts["person"], "vehicle_objects": counts["vehicle"],
                "total_objects": sum(counts.values()),
                "person_ge_10": int(counts["person"] >= 10),
                "person_ge_20": int(counts["person"] >= 20),
                "person_ge_50": int(counts["person"] >= 50),
                "person_ge_100": int(counts["person"] >= 100),
                "vehicle_ge_10": int(counts["vehicle"] >= 10),
                "vehicle_ge_20": int(counts["vehicle"] >= 20),
                "vehicle_ge_50": int(counts["vehicle"] >= 50),
                "vehicle_ge_100": int(counts["vehicle"] >= 100),
            })

            if i % 1000 == 0:
                print(f"Procesadas: {i:,}/{len(images):,}")

    total_objects = sum(stats[x]["objects"] for x in stats)
    write_csv(object_csv, object_rows)
    write_csv(image_csv, image_rows)

    split_rows = []
    for sp in SPLITS:
        for name in CLASSES.values():
            ss = split[sp][name]
            a = ss["areas"]
            lt16 = ss["bins"]["<16"]
            lt32 = lt16 + ss["bins"]["16-32"]
            lt64 = lt32 + ss["bins"]["32-64"]
            split_rows.append({
                "split": sp, "class": name,
                "images": split[sp]["_meta"]["images"] if "_meta" in split[sp] else 0,
                "images_with_class": len(ss["images_with_class"]),
                "objects": ss["objects"],
                "objects_per_image_with_class": round(ss["objects"] / len(ss["images_with_class"]), 4) if ss["images_with_class"] else 0,
                "mean_area_px2": round(sum(a) / len(a), 4) if a else 0,
                "median_area_px2": round(percentile(a, .5), 4),
                "tiny_lt16": lt16, "tiny_lt32": lt32, "tiny_lt64": lt64,
                "partial_bbox": ss["partial"], "near_border": ss["border"],
            })
    write_csv(split_csv, split_rows)

    lines = [
        "=" * 72,
        "SAR YOLO26 - PERSON VS VEHICLE ANALYSIS V1",
        "=" * 72, "",
        f"Dataset: {DATASET_ROOT}", f"Output:  {OUTPUT_ROOT}", "",
        "DATASET",
        f"Images analyzed:    {total_images:,}",
        f"Objects analyzed:   {total_objects:,}",
        f"Corrupt images:      {corrupt:,}",
        f"Missing labels:      {missing_labels:,}", "",
    ]

    for name in ("person", "vehicle"):
        s = stats[name]
        n = s["objects"]
        lt16 = s["bins"]["<16"]
        lt32 = lt16 + s["bins"]["16-32"]
        lt64 = lt32 + s["bins"]["32-64"]
        lines += [
            "=" * 72, name.upper(), "=" * 72,
            f"Objects:              {n:,}",
            f"Images with class:    {len(s['images']):,}",
            f"Objects/image:        {n / len(s['images']) if s['images'] else 0:.2f}", "",
            "AREA",
            f"<16 px²:              {lt16:,} ({pct(lt16,n):.2f} %)",
            f"<32 px²:              {lt32:,} ({pct(lt32,n):.2f} %)",
            f"<64 px²:              {lt64:,} ({pct(lt64,n):.2f} %)",
        ]
        for b, _, _ in AREA_BINS:
            lines.append(f"{b:22}: {s['bins'][b]:,} ({pct(s['bins'][b],n):.2f} %)")
        lines += [
            "", "GEOMETRY",
            f"Area mean px²:        {sum(s['areas'])/len(s['areas']) if s['areas'] else 0:.2f}",
            f"Area median px²:      {percentile(s['areas'],.50):.2f}",
            f"Area P75 px²:         {percentile(s['areas'],.75):.2f}",
            f"Area P90 px²:         {percentile(s['areas'],.90):.2f}",
            f"Area P95 px²:         {percentile(s['areas'],.95):.2f}",
            f"Area P99 px²:         {percentile(s['areas'],.99):.2f}",
            f"Width mean px:        {sum(s['widths'])/len(s['widths']) if s['widths'] else 0:.2f}",
            f"Height mean px:       {sum(s['heights'])/len(s['heights']) if s['heights'] else 0:.2f}",
            f"Aspect ratio median:  {percentile(s['ratios'],.50):.3f}", "",
            "BORDERS / BBOX",
            f"Partial BBox:         {s['partial']:,} ({pct(s['partial'],n):.2f} %)",
            f"Near border:          {s['border']:,} ({pct(s['border'],n):.2f} %)", "",
            "CROWDING",
        ]
        for t in (10,20,50,100):
            lines.append(f"Images >= {t:3}:         {s['crowding'][t]:,}")
        lines.append("")

    p, v = stats["person"], stats["vehicle"]
    lines += [
        "=" * 72, "DIRECT COMPARISON", "=" * 72, "",
        "Metric                    PERSON        VEHICLE",
        f"Objects                 {p['objects']:>10,} {v['objects']:>13,}",
        f"<16 px²                 {p['bins']['<16']:>10,} {v['bins']['<16']:>13,}",
        f"<32 px²                 {p['bins']['<16']+p['bins']['16-32']:>10,} {v['bins']['<16']+v['bins']['16-32']:>13,}",
        f"<64 px²                 {p['bins']['<16']+p['bins']['16-32']+p['bins']['32-64']:>10,} {v['bins']['<16']+v['bins']['16-32']+v['bins']['32-64']:>13,}",
        f"Median area px²         {percentile(p['areas'],.5):>10.2f} {percentile(v['areas'],.5):>13.2f}",
        f"P90 area px²            {percentile(p['areas'],.9):>10.2f} {percentile(v['areas'],.9):>13.2f}",
        f"Partial BBox %          {pct(p['partial'],p['objects']):>10.2f} {pct(v['partial'],v['objects']):>13.2f}",
        f"Near border %           {pct(p['border'],p['objects']):>10.2f} {pct(v['border'],v['objects']):>13.2f}",
        "",
        "INTERPRETATION",
        "This report is descriptive only. It does not measure TP/FP/FN.",
        "The next stage should cross these properties with YOLO26 predictions",
        "to identify which conditions are associated with PERSON misses.",
        "",
        "IMPORTANT: the dataset was NOT modified.",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("# RESULTADO PERSON VS VEHICLE ANALYSIS V1")
    print("=" * 72)
    print(f"\nImágenes:              {total_images:,}")
    print(f"Objetos:               {total_objects:,}\n")
    for name in ("person", "vehicle"):
        s = stats[name]; n = s["objects"]
        lt16=s["bins"]["<16"]; lt32=lt16+s["bins"]["16-32"]; lt64=lt32+s["bins"]["32-64"]
        print(name.upper())
        print(f"Objetos:               {n:,}")
        print(f"<16 px²:               {lt16:,} ({pct(lt16,n):.2f} %)")
        print(f"<32 px²:               {lt32:,} ({pct(lt32,n):.2f} %)")
        print(f"<64 px²:               {lt64:,} ({pct(lt64,n):.2f} %)")
        print(f"Mediana área:          {percentile(s['areas'],.5):.2f} px²")
        print(f"P90 área:              {percentile(s['areas'],.9):.2f} px²")
        print(f"Partial BBox:          {s['partial']:,} ({pct(s['partial'],n):.2f} %)")
        print(f"Near border:           {s['border']:,} ({pct(s['border'],n):.2f} %)\n")
    print(f"[OK] {object_csv}")
    print(f"[OK] {image_csv}")
    print(f"[OK] {split_csv}")
    print(f"[OK] {summary}")
    print("\nIMPORTANTE: el dataset NO ha sido modificado.")
    print("=" * 72)


if __name__ == "__main__":
    main()
