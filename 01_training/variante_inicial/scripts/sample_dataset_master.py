import os
import random
import cv2
import yaml
from pathlib import Path

# ==========================================================
# Muestreo visual para Dataset_Master
# ==========================================================

DATASET_ROOT = Path(r"C:\SARC-Drone\00_datasets\processed\Dataset_Master")
SPLIT = "train"  # train / val / test

IMAGES_DIR = DATASET_ROOT / SPLIT / "images"
LABELS_DIR = DATASET_ROOT / SPLIT / "labels"
DATA_YAML = DATASET_ROOT / "data.yaml"

# ==========================================================

if DATA_YAML.exists():
    with DATA_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", {})
else:
    class_names = {}

# Para este visor, tratamos todas las clases como "personas".
# Así puedes inspeccionar visualmente el dataset sin depender del nombre exacto de cada clase.
class_id_to_filter = None

# También incluimos clases que suelen representar peatones o personas en otros datasets.
person_like_names = {"person", "people", "human", "persons", "pedestrian", "pedestrians", "walker", "walkers"}

image_files = sorted([p for p in IMAGES_DIR.glob("*.jpg")])
image_files += sorted([p for p in IMAGES_DIR.glob("*.jpeg")])
image_files += sorted([p for p in IMAGES_DIR.glob("*.png")])
image_files += sorted([p for p in IMAGES_DIR.glob("*.bmp")])
image_files = sorted(set(image_files))

if not image_files:
    raise FileNotFoundError(f"No se encontraron imágenes en: {IMAGES_DIR}")

index = 0

while True:
    image_path = image_files[index]
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"No se pudo abrir: {image_path}")
        index = min(index + 1, len(image_files) - 1)
        continue

    h, w = img.shape[:2]

    label_path = LABELS_DIR / f"{image_path.stem}.txt"
    total_objects = 0

    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                if class_id_to_filter is not None and cls_id != class_id_to_filter:
                    continue

                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                color = (0, 255, 0)
                if bw < 0.01 or bh < 0.01:
                    color = (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                if isinstance(class_names, dict):
                    name = class_names.get(cls_id, str(cls_id))
                else:
                    name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

                cv2.putText(img, str(name), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                total_objects += 1
    else:
        cv2.putText(img, "SIN LABEL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    text = f"{index + 1}/{len(image_files)}   Objetos: {total_objects}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Visor Dataset YOLO - Dataset_Master", img)
    key = cv2.waitKey(0)

    if key in (27, ord("q")):
        break
    elif key in (ord("d"), 2555904):
        index = min(index + 1, len(image_files) - 1)
    elif key in (ord("a"), 2424832):
        index = max(index - 1, 0)
    elif key == ord("r"):
        index = random.randint(0, len(image_files) - 1)

cv2.destroyAllWindows()
