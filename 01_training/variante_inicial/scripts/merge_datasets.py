"""Mezcla datasets crudos en estructura YOLO procesada usando config.yaml."""

import argparse
from pathlib import Path
import random
import shutil

from config_utils import load_config, path_from_config

CFG = load_config()
RAW = path_from_config(CFG, "datasets_raw")
PROC = path_from_config(CFG, "datasets_processed")


def collect_pairs(source_dir: Path):
    images = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.png"))
    pairs = []
    for img in images:
        label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    return pairs


def split_items(items, val_ratio, test_ratio, seed=42):
    random.seed(seed)
    random.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def copy_set(items, split):
    out_img = PROC / split / "images"
    out_lbl = PROC / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for idx, (img, lbl) in enumerate(items):
        stem = f"{split}_{idx:06d}"
        shutil.copy2(img, out_img / f"{stem}{img.suffix.lower()}")
        shutil.copy2(lbl, out_lbl / f"{stem}.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["visdrone", "okutama", "nitc_rescue"])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_items = []
    for source in args.sources:
        source_dir = RAW / source
        if not source_dir.exists():
            continue
        all_items.extend(collect_pairs(source_dir))

    if not all_items:
        print("No se encontraron pares imagen/label para mezclar.")
        return

    train, val, test = split_items(all_items, args.val_ratio, args.test_ratio, args.seed)
    copy_set(train, "train")
    copy_set(val, "val")
    copy_set(test, "test")

    print(f"Merge completado. train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
