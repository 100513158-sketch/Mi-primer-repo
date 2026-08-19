"""Convertidor base de OKUTAMA a YOLO usando config.yaml."""

import argparse
from pathlib import Path
import shutil
import cv2

from config_utils import load_config, path_from_config

CFG = load_config()
RAW_OKUTAMA = path_from_config(CFG, "datasets_raw") / "okutama"
PROCESSED = path_from_config(CFG, "datasets_processed")


def to_yolo_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def parse_annotations(_annotation_file: Path):
    # TODO: Adaptar al formato real de OKUTAMA disponible en tu descarga.
    # Debe retornar una lista de tuplas: (image_name, [(class_id, x1,y1,x2,y2), ...])
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    images_dir = RAW_OKUTAMA / "images"
    ann_file = RAW_OKUTAMA / "annotations.txt"

    out_images = PROCESSED / args.split / "images"
    out_labels = PROCESSED / args.split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    entries = parse_annotations(ann_file)

    for image_name, boxes in entries:
        src = images_dir / image_name
        if not src.exists():
            continue

        image = cv2.imread(str(src))
        if image is None:
            continue

        h, w = image.shape[:2]
        shutil.copy2(src, out_images / image_name)

        yolo_lines = []
        for class_id, x1, y1, x2, y2 in boxes:
            xc, yc, bw, bh = to_yolo_bbox(x1, y1, x2, y2, w, h)
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        (out_labels / f"{Path(image_name).stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    print("Conversion base de OKUTAMA finalizada.")


if __name__ == "__main__":
    main()
