from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def ensure_split_dirs(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG file")

    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            index += 1
            continue

        marker = data[index + 1]
        index += 2

        if marker in (0xD8, 0xD9):
            continue

        if index + 2 > len(data):
            break

        segment_length = int.from_bytes(data[index:index + 2], "big")
        if segment_length < 2:
            break

        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            if index + 7 > len(data):
                break
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height

        index += segment_length

    raise ValueError("Could not read JPEG size")


def coco_to_yolo(box: list[float], image_width: float, image_height: float, class_id: int) -> YoloBox:
    x, y, width, height = box
    return YoloBox(
        class_id=class_id,
        x_center=(x + width / 2.0) / image_width,
        y_center=(y + height / 2.0) / image_height,
        width=width / image_width,
        height=height / image_height,
    )


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, image_width: float, image_height: float, class_id: int) -> YoloBox:
    width = x2 - x1
    height = y2 - y1
    return YoloBox(
        class_id=class_id,
        x_center=(x1 + width / 2.0) / image_width,
        y_center=(y1 + height / 2.0) / image_height,
        width=width / image_width,
        height=height / image_height,
    )


def clamp_box(box: YoloBox) -> YoloBox:
    return YoloBox(
        class_id=box.class_id,
        x_center=min(max(box.x_center, 0.0), 1.0),
        y_center=min(max(box.y_center, 0.0), 1.0),
        width=min(max(box.width, 0.0), 1.0),
        height=min(max(box.height, 0.0), 1.0),
    )


def format_box(box: YoloBox) -> str:
    return f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"


def split_from_hash(key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000 / 10000.0
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def archive_entry_name_map(archive: zipfile.ZipFile) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in archive.infolist():
        base = Path(entry.filename).name.lower()
        if base not in mapping:
            mapping[base] = entry.filename
    return mapping


def extract_entry(archive: zipfile.ZipFile, entry_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(entry_name) as source, output_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def find_existing_entry(name_map: dict[str, str], basename: str) -> str:
    key = basename.lower()
    if key not in name_map:
        raise FileNotFoundError(f"Could not find {basename} in archive")
    return name_map[key]


def build_global_zip_index(zip_files: list[Path]) -> dict[str, tuple[Path, str]]:
    index: dict[str, tuple[Path, str]] = {}
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as archive:
            for entry in archive.infolist():
                if not entry.filename.lower().endswith(".jpg"):
                    continue
                base = Path(entry.filename).name.lower()
                if base not in index:
                    index[base] = (zip_path, entry.filename)
    return index


def prepare_nomad(source_root: Path, output_root: Path, train_ratio: float, val_ratio: float) -> None:
    dataset_output = output_root / "NOMAD"
    ensure_split_dirs(dataset_output)
    (dataset_output / "classes.txt").write_text("person\n", encoding="utf-8")

    zip_files = sorted((source_root / "NOMAD").glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No NOMAD zip files found")

    global_image_index = build_global_zip_index(zip_files)

    counts = defaultdict(int)
    for archive_path in zip_files:
        with zipfile.ZipFile(archive_path) as archive:
            json_entries = [entry for entry in archive.infolist() if entry.filename.lower().endswith("annotations.json")]
            if not json_entries:
                continue

            records = json.loads(archive.read(json_entries[0]))
            if not isinstance(records, list):
                continue

            for record in records:
                file_name = record["file_name"]
                image_width = float(record["width"])
                image_height = float(record["height"])
                image_base_name = Path(file_name).name.lower()
                split = split_from_hash(f"{archive_path.stem}/{file_name}", train_ratio, val_ratio)
                output_name = f"{archive_path.stem}__{Path(file_name).name}"
                image_output = dataset_output / split / "images" / output_name
                label_output = dataset_output / split / "labels" / f"{Path(output_name).stem}.txt"

                image_lookup = global_image_index.get(image_base_name)
                if image_lookup is None:
                    continue
                image_zip_path, image_zip_entry = image_lookup
                with zipfile.ZipFile(image_zip_path) as image_archive:
                    extract_entry(image_archive, image_zip_entry, image_output)

                boxes: list[str] = []
                for annotation in record.get("annotations", []):
                    bbox = annotation.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    boxes.append(format_box(clamp_box(coco_to_yolo(bbox, image_width, image_height, 0))))

                write_lines(label_output, boxes)
                counts[split] += 1

    write_lines(dataset_output / "splits.txt", [f"{key}: {value}" for key, value in sorted(counts.items())])


def prepare_okutama(source_root: Path, output_root: Path, val_ratio: float) -> None:
    dataset_output = output_root / "OKUTAMA"
    ensure_split_dirs(dataset_output)
    (dataset_output / "classes.txt").write_text("person\n", encoding="utf-8")

    split_files = {
        "train": source_root / "OKUTAMA" / "TrainSetFrames.zip",
        "test": source_root / "OKUTAMA" / "TestSetFrames.zip",
    }

    counts = defaultdict(int)
    for source_split, zip_path in split_files.items():
        if not zip_path.exists():
            continue

        with zipfile.ZipFile(zip_path) as archive:
            frame_entries = [entry for entry in archive.infolist() if "/extracted-frames-" in entry.filename.lower() and entry.filename.lower().endswith(".jpg")]
            label_entries = [entry for entry in archive.infolist() if entry.filename.lower().startswith("labels/multiactionlabels/") and entry.filename.lower().endswith(".txt")]

            frame_map: dict[str, dict[int, str]] = defaultdict(dict)
            for entry in frame_entries:
                parts = Path(entry.filename).parts
                try:
                    video_id = parts[-2]
                    frame_index = int(Path(entry.filename).stem)
                except (IndexError, ValueError):
                    continue
                frame_map[video_id][frame_index] = entry.filename

            for label_entry in label_entries:
                video_id = Path(label_entry.filename).stem
                split = source_split
                if source_split == "train":
                    split = "val" if split_from_hash(video_id, 1.0 - val_ratio, val_ratio) == "val" else "train"

                source_width, source_height = 3840.0, 2160.0
                label_text = archive.read(label_entry).decode("utf-8", errors="ignore")
                frame_boxes: dict[int, list[YoloBox]] = defaultdict(list)

                for raw_line in label_text.splitlines():
                    if not raw_line.strip():
                        continue
                    try:
                        tokens = shlex.split(raw_line)
                    except ValueError:
                        continue
                    if len(tokens) < 5:
                        continue
                    try:
                        frame_index = int(float(tokens[0]))
                        x1 = float(tokens[1])
                        y1 = float(tokens[2])
                        x2 = float(tokens[3])
                        y2 = float(tokens[4])
                    except ValueError:
                        continue
                    frame_boxes[frame_index].append(xyxy_to_yolo(x1, y1, x2, y2, source_width, source_height, 0))

                for frame_index, boxes in frame_boxes.items():
                    image_entry_name = frame_map.get(video_id, {}).get(frame_index)
                    if image_entry_name is None:
                        continue

                    image_bytes = archive.read(image_entry_name)
                    image_width, image_height = parse_jpeg_size(image_bytes)
                    output_name = f"{video_id}_frame{frame_index:06d}.jpg"
                    image_output = dataset_output / split / "images" / output_name
                    label_output = dataset_output / split / "labels" / f"{Path(output_name).stem}.txt"

                    image_output.parent.mkdir(parents=True, exist_ok=True)
                    image_output.write_bytes(image_bytes)

                    scale_x = image_width / source_width
                    scale_y = image_height / source_height
                    scaled_boxes = [
                        format_box(
                            clamp_box(
                                YoloBox(
                                    class_id=box.class_id,
                                    x_center=box.x_center * scale_x,
                                    y_center=box.y_center * scale_y,
                                    width=box.width * scale_x,
                                    height=box.height * scale_y,
                                )
                            )
                        )
                        for box in boxes
                    ]

                    write_lines(label_output, scaled_boxes)
                    counts[split] += 1

    write_lines(dataset_output / "splits.txt", [f"{key}: {value}" for key, value in sorted(counts.items())])


def prepare_seadronessee(source_root: Path, output_root: Path) -> None:
    dataset_output = output_root / "SeaDronesSee"
    ensure_split_dirs(dataset_output)

    annotation_dir = source_root / "SeaDronesSee" / "annotations"
    image_dir = source_root / "SeaDronesSee" / "images"
    train_json = annotation_dir / "instances_train.json"
    val_json = annotation_dir / "instances_val.json"

    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError("SeaDronesSee COCO annotation files not found")

    classes = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
    (dataset_output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    def convert_split(split: str, json_path: Path, image_subdir: str) -> None:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        images = {int(image["id"]): image for image in data.get("images", [])}
        category_map = {
            int(category["id"]): index
            for index, category in enumerate(category for category in data.get("categories", []) if int(category["id"]) != 0)
        }

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image_id, image in images.items():
            file_name = image["file_name"]
            image_path = image_dir / image_subdir / file_name
            if not image_path.exists():
                image_path = image_dir / file_name

            output_image_path = dataset_output / split / "images" / file_name
            output_label_path = dataset_output / split / "labels" / f"{Path(file_name).stem}.txt"
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.exists():
                shutil.copy2(image_path, output_image_path)

            boxes: list[str] = []
            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                if category_id not in category_map:
                    continue
                bbox = annotation.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                boxes.append(
                    format_box(
                        clamp_box(
                            coco_to_yolo(
                                bbox,
                                float(image["width"]),
                                float(image["height"]),
                                category_map[category_id],
                            )
                        )
                    )
                )

            write_lines(output_label_path, boxes)

    convert_split("train", train_json, "train")
    convert_split("val", val_json, "val")

    test_image_dir = image_dir / "test"
    if test_image_dir.exists():
        for image_path in test_image_dir.glob("*.jpg"):
            output_image_path = dataset_output / "test" / "images" / image_path.name
            output_label_path = dataset_output / "test" / "labels" / f"{image_path.stem}.txt"
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image_path)
            write_lines(output_label_path, [])

    (dataset_output / "dataset.yaml").write_text(
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: swimmer\n"
        "  1: boat\n"
        "  2: jetski\n"
        "  3: life_saving_appliances\n"
        "  4: buoy\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare rescue datasets into a YOLO-style structure.")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "prepared")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nomad", "okutama", "seadronessee"],
        choices=["nomad", "okutama", "seadronessee", "all"],
        help="Datasets to prepare.",
    )
    parser.add_argument("--nomad-train-ratio", type=float, default=0.8)
    parser.add_argument("--nomad-val-ratio", type=float, default=0.1)
    parser.add_argument("--okutama-val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"nomad", "okutama", "seadronessee"}

    args.output_root.mkdir(parents=True, exist_ok=True)

    if "nomad" in selected:
        prepare_nomad(args.source_root, args.output_root, args.nomad_train_ratio, args.nomad_val_ratio)
    if "okutama" in selected:
        prepare_okutama(args.source_root, args.output_root, args.okutama_val_ratio)
    if "seadronessee" in selected:
        prepare_seadronessee(args.source_root, args.output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())