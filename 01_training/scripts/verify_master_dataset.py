"""
verify_master_dataset.py — Validación específica para Dataset_Master.

Comprueba:
  - Estructura YOLO por split
  - Integridad básica usando integrity_check.py
  - Estadísticas útiles para SAR (objetos pequeños y labels vacíos)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config_utils import load_config, path_from_config
from integrity_check import check_dataset

SPLITS = ("train", "val", "test")


def _scan_stats(ds_path: Path) -> dict:
    total_images = 0
    total_labels = 0
    empty_labels = 0
    total_boxes = 0
    small_boxes = 0

    for split in SPLITS:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            total_images += 1
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue

            total_labels += 1
            if lbl.stat().st_size == 0:
                empty_labels += 1
                continue

            raw = lbl.read_text(encoding="utf-8", errors="replace")
            for line in raw.splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    continue
                area = bw * bh
                total_boxes += 1
                if area <= 0.01:
                    small_boxes += 1

    return {
        "images": total_images,
        "labels": total_labels,
        "empty_labels": empty_labels,
        "boxes": total_boxes,
        "small_boxes": small_boxes,
        "small_box_ratio": (small_boxes / total_boxes) if total_boxes else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica Dataset_Master antes de entrenar.")
    parser.add_argument(
        "--dataset-name",
        default="Dataset_Master",
        help="Nombre de la carpeta en processed/ (default: Dataset_Master)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla si hay imágenes sin label.",
    )
    args = parser.parse_args()

    cfg = load_config()
    processed_root = path_from_config(cfg, "datasets_processed")
    ds_path = processed_root / args.dataset_name

    if not ds_path.exists():
        print(f"[FAIL] Dataset no encontrado: {ds_path}")
        return 1

    passed, detail = check_dataset(ds_path, strict=args.strict)
    stats = _scan_stats(ds_path)

    print("\n=== Verificación Dataset_Master ===")
    print(f"Dataset  : {ds_path}")
    print(f"Estado   : {'OK' if passed else 'FAIL'}")
    print(f"Imágenes : {stats['images']}")
    print(f"Labels   : {stats['labels']} (vacíos: {stats['empty_labels']})")
    print(f"Boxes    : {stats['boxes']}")
    print(f"Small<=1%: {stats['small_boxes']} ({stats['small_box_ratio']:.2%})")

    if detail["warnings"]:
        print("\nWarnings:")
        for w in detail["warnings"][:10]:
            print(f"  - {w}")

    if detail["errors"]:
        print("\nErrores:")
        for e in detail["errors"][:10]:
            print(f"  - {e}")

    report = {
        "dataset": str(ds_path),
        "passed": passed,
        "stats": stats,
        "warnings": detail["warnings"],
        "errors": detail["errors"],
    }
    out = ds_path / "reports" / "verify_master_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte: {out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
