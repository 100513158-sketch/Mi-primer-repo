"""
convert_all_datasets.py — Convierte todos los datasets raw a formato YOLO en processed/.

Lee rutas desde config.yaml. Limpia processed/ y regenera desde cero.

Formatos de anotaciones raw por dataset:
  VisDrone    : CSV  x,y,w,h,score,cat,trunc,occl  -> YOLO (requiere dims. de imagen)
  NOMAD       : YOLO ya normalizado  (annotations/ -> labels/)
  OKUTAMA     : YOLO ya normalizado  (annotations/ -> labels/)
  SeaDronesSee: YOLO multiclase      (annotations/ -> labels/)
  NITC        : YOLO, class_id=1     (annotations/ -> labels/, remap 1->0)
  RESDataset  : YOLO ya normalizado  (annotations/ -> labels/)
  C2A         : COCO JSON            -> YOLO

Uso:
  python convert_all_datasets.py                     # convierte todo (limpia primero)
  python convert_all_datasets.py --no-clean          # no limpiar processed/
  python convert_all_datasets.py --dataset VisDrone  # solo un dataset
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

from config_utils import load_config, path_from_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ─── Configuración de datasets ────────────────────────────────────────────────
# converter: estrategia de conversión
# class_remap: {class_id_src: class_id_dst} o None para copiar sin cambios

DATASETS: dict[str, dict] = {
    "VisDrone":     {"converter": "visdrone"},
    "NOMAD":        {"converter": "yolo_copy", "class_remap": None},
    "OKUTAMA":      {"converter": "yolo_copy", "class_remap": None},
    "SeaDronesSee": {"converter": "yolo_copy", "class_remap": None},
    "NITC":         {"converter": "yolo_copy", "class_remap": {1: 0}},  # 1=person → 0
    "RESDataset":   {"converter": "yolo_copy", "class_remap": None, "normalize_class_ids": True},  # 0.0 → 0
    "C2A":          {"converter": "coco_json"},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_split_dirs(ds_output: Path, split: str) -> tuple[Path, Path]:
    """Crea y retorna (images_dir, labels_dir) para el split dado."""
    img_dir = ds_output / split / "images"
    lbl_dir = ds_output / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def _apply_class_remap(content: str, class_remap: dict[int, int]) -> str:
    """Reescribe un archivo label YOLO aplicando el remapeo de clases."""
    lines = []
    for line in content.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))  # Parsear como float primero (0.0 → 0)
        except (ValueError, IndexError):
            continue
        new_cid = class_remap.get(cid, cid)
        lines.append(f"{new_cid} " + " ".join(parts[1:]))
    return "\n".join(lines) + ("\n" if lines else "")


def _normalize_class_ids(content: str) -> str:
    """Normaliza class_id a enteros (limpia 0.0 → 0, etc)."""
    lines = []
    for line in content.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))  # Parsear como float, castear a int
        except (ValueError, IndexError):
            continue
        lines.append(f"{cid} " + " ".join(parts[1:]))
    return "\n".join(lines) + ("\n" if lines else "")


def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


# ─── Conversores ──────────────────────────────────────────────────────────────

def copy_yolo_splits(
    raw_ds: Path,
    ds_output: Path,
    class_remap: dict[int, int] | None = None,
    normalize_class_ids: bool = False,
) -> dict[str, int]:
    """
    Copia datasets que ya están en formato YOLO (images/ + annotations/).
    Si class_remap no es None, reescribe los labels con el nuevo class_id.
    Si normalize_class_ids es True, convierte class_id flotantes a enteros (0.0 → 0).
    Retorna {split: n_images}.
    """
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        img_src = raw_ds / split / "images"
        ann_src = raw_ds / split / "annotations"
        if not img_src.exists():
            continue

        img_dst, lbl_dst = _make_split_dirs(ds_output, split)
        n = 0
        for img_path in img_src.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            shutil.copy2(img_path, img_dst / img_path.name)

            ann_file = (ann_src / (img_path.stem + ".txt")) if ann_src.exists() else None
            lbl_out  = lbl_dst / (img_path.stem + ".txt")

            if ann_file and ann_file.exists():
                text = ann_file.read_text(encoding="utf-8", errors="replace")
                if normalize_class_ids:
                    text = _normalize_class_ids(text)
                if class_remap:
                    text = _apply_class_remap(text, class_remap)
                lbl_out.write_text(text, encoding="utf-8")
            else:
                lbl_out.write_text("", encoding="utf-8")  # imagen sin objetos

            n += 1
        counts[split] = n
    return counts


def convert_visdrone(
    raw_ds: Path,
    ds_output: Path,
    class_mapping: dict[int, int],
) -> dict[str, int]:
    """
    Convierte VisDrone CSV → YOLO.
    Formato CSV: bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
    La dimensión de cada imagen se lee desde el archivo con PIL.
    """
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        img_src = raw_ds / split / "images"
        ann_src = raw_ds / split / "annotations"
        if not img_src.exists() or not ann_src.exists():
            continue

        img_dst, lbl_dst = _make_split_dirs(ds_output, split)
        n = 0
        for img_path in img_src.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            # Leer dimensiones de imagen (solo cabecera, no decodifica pixeles)
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = float(im.size[0]), float(im.size[1])
            except Exception:
                continue

            shutil.copy2(img_path, img_dst / img_path.name)

            ann_file = ann_src / (img_path.stem + ".txt")
            lbl_out  = lbl_dst / (img_path.stem + ".txt")
            yolo_lines: list[str] = []

            if ann_file.exists():
                for row in ann_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = row.strip().split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        x   = float(parts[0])
                        y   = float(parts[1])
                        w   = float(parts[2])
                        h   = float(parts[3])
                        cat = int(parts[5])
                    except ValueError:
                        continue
                    if cat not in class_mapping or w <= 0 or h <= 0:
                        continue
                    cx = _clamp01((x + w / 2.0) / img_w)
                    cy = _clamp01((y + h / 2.0) / img_h)
                    nw = _clamp01(w / img_w)
                    nh = _clamp01(h / img_h)
                    yolo_lines.append(
                        f"{class_mapping[cat]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                    )

            lbl_out.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )
            n += 1
        counts[split] = n
    return counts


def convert_coco_json(raw_ds: Path, ds_output: Path) -> dict[str, int]:
    """
    Convierte C2A COCO JSON → YOLO.
    Formato JSON: {images: [{id, width, height, file_name}], annotations: [{image_id, bbox: [x,y,w,h]}]}
    Todos los objetos se mapean a clase 0 (humano/persona).
    """
    split_json_map = {
        "train": "train_annotations.json",
        "val":   "val_annotations.json",
        "test":  "test_annotations.json",
    }
    counts: dict[str, int] = {}

    for split, json_name in split_json_map.items():
        json_path = raw_ds / split / json_name
        img_src   = raw_ds / split / "images"
        if not json_path.exists() or not img_src.exists():
            continue

        img_dst, lbl_dst = _make_split_dirs(ds_output, split)
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # Índice image_id → metadata
        images: dict[int, dict] = {int(img["id"]): img for img in data.get("images", [])}

        # Índice image_id → lista de anotaciones
        ann_by_image: dict[int, list] = {}
        for ann in data.get("annotations", []):
            iid = int(ann["image_id"])
            ann_by_image.setdefault(iid, []).append(ann)

        n = 0
        for iid, img_info in images.items():
            file_name = img_info["file_name"]
            img_w     = float(img_info.get("width", 0))
            img_h     = float(img_info.get("height", 0))
            if img_w <= 0 or img_h <= 0:
                continue

            # Localizar imagen (puede estar en subdirectorio)
            img_path = img_src / file_name
            if not img_path.exists():
                candidates = list(img_src.rglob(Path(file_name).name))
                if not candidates:
                    continue
                img_path = candidates[0]

            shutil.copy2(img_path, img_dst / img_path.name)

            yolo_lines: list[str] = []
            for ann in ann_by_image.get(iid, []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                if w <= 0 or h <= 0:
                    continue
                cx = _clamp01((x + w / 2.0) / img_w)
                cy = _clamp01((y + h / 2.0) / img_h)
                nw = _clamp01(w / img_w)
                nh = _clamp01(h / img_h)
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            lbl_out = lbl_dst / (Path(file_name).stem + ".txt")
            lbl_out.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )
            n += 1
        counts[split] = n
    return counts


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convierte todos los datasets raw a formato YOLO en processed/."
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="No limpiar el directorio processed/ antes de convertir.",
    )
    parser.add_argument(
        "--dataset", metavar="DATASET",
        help=f"Convertir solo este dataset. Opciones: {', '.join(DATASETS)}",
    )
    args = parser.parse_args()

    cfg             = load_config()
    raw_root        = path_from_config(cfg, "datasets_raw")
    processed_root  = path_from_config(cfg, "datasets_processed")
    visdrone_mapping = {
        int(k): int(v)
        for k, v in cfg.get("datasets", {})
                       .get("visdrone", {})
                       .get("class_mapping", {})
                       .items()
    }

    # Selección de datasets
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"ERROR: dataset '{args.dataset}' no conocido. "
                  f"Opciones: {', '.join(DATASETS)}")
            return 1
        to_process = {args.dataset: DATASETS[args.dataset]}
    else:
        to_process = DATASETS

    # Limpiar processed/
    if not args.no_clean:
        if processed_root.exists():
            print(f"Limpiando {processed_root} ...")
            for child in processed_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        processed_root.mkdir(parents=True, exist_ok=True)
        print(f"Directorio listo: {processed_root}\n")

    print("=== Conversión de datasets raw → YOLO processed ===\n")
    total_ok = total_fail = 0

    for ds_name, ds_cfg in to_process.items():
        raw_ds    = raw_root / ds_name
        ds_output = processed_root / ds_name
        converter = ds_cfg["converter"]

        if not raw_ds.exists():
            print(f"  [SKIP]  {ds_name:<16} no encontrado en {raw_root}")
            total_fail += 1
            continue

        try:
            if converter == "visdrone":
                counts = convert_visdrone(raw_ds, ds_output, visdrone_mapping)
            elif converter == "coco_json":
                counts = convert_coco_json(raw_ds, ds_output)
            elif converter == "yolo_copy":
                counts = copy_yolo_splits(
                    raw_ds,
                    ds_output,
                    ds_cfg.get("class_remap"),
                    ds_cfg.get("normalize_class_ids", False),
                )
            else:
                print(f"  [FAIL]  {ds_name:<16} conversor desconocido: '{converter}'")
                total_fail += 1
                continue

            splits_str = "  ".join(
                f"{s}:{n}" for s, n in sorted(counts.items()) if n > 0
            )
            remap_note = ""
            if ds_cfg.get("class_remap"):
                remap_note = f"  [remap {ds_cfg['class_remap']}]"
            print(f"  [OK]    {ds_name:<16} {splits_str}{remap_note}")
            total_ok += 1

        except Exception as exc:
            print(f"  [FAIL]  {ds_name:<16} {exc}")
            import traceback
            traceback.print_exc()
            total_fail += 1

    print()
    if total_fail == 0:
        print(f"Conversión completada. {total_ok}/{total_ok + total_fail} datasets procesados.")
        print("Siguiente paso: python integrity_check.py")
    else:
        print(f"Atención: {total_ok} OK, {total_fail} fallaron.")

    return 0 if not total_fail else 2


if __name__ == "__main__":
    sys.exit(main())
