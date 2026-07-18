"""Convierte VisDrone a formato YOLO leyendo rutas desde config.yaml."""

from pathlib import Path
import shutil
import cv2

from config_utils import load_config, path_from_config

CFG = load_config()
VISDRONE_PATH = path_from_config(CFG, "datasets_raw") / "visdrone"
OUTPUT_PATH = path_from_config(CFG, "datasets_processed")

# VisDrone categories (segun la guia del proyecto): 0=pedestrian, 1=people
CLASS_MAPPING = {int(k): int(v) for k, v in CFG["datasets"]["visdrone"]["class_mapping"].items()}


def convert_annotation(input_file: Path, output_file: Path, img_width: int, img_height: int) -> None:
    yolo_annotations = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])

            if category not in CLASS_MAPPING or w <= 0 or h <= 0:
                continue

            x_center = (x + w / 2.0) / img_width
            y_center = (y + h / 2.0) / img_height
            width = w / img_width
            height = h / img_height

            yolo_annotations.append(
                f"{CLASS_MAPPING[category]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(yolo_annotations), encoding="utf-8")


def process_split(split: str) -> None:
    img_dir = VISDRONE_PATH / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = VISDRONE_PATH / f"VisDrone2019-DET-{split}" / "annotations"

    output_img_dir = OUTPUT_PATH / split / "images"
    output_ann_dir = OUTPUT_PATH / split / "labels"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)

    for img_file in img_dir.glob("*.jpg"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        h, w = img.shape[:2]
        shutil.copy2(img_file, output_img_dir / img_file.name)

        ann_file = ann_dir / f"{img_file.stem}.txt"
        output_ann = output_ann_dir / f"{img_file.stem}.txt"
        if ann_file.exists():
            convert_annotation(ann_file, output_ann, w, h)
        else:
            output_ann.write_text("", encoding="utf-8")


def main() -> None:
    print("Convirtiendo VisDrone a formato YOLO...")
    for split in ("train", "val"):
        process_split(split)
    print("Conversion completada.")


if __name__ == "__main__":
    main()
