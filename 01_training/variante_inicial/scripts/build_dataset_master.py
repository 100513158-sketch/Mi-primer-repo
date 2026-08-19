"""
build_dataset_master.py — Construye un Dataset_Master unificado para detección SAR.

Origen:
  00_datasets/processed/<DATASET>/{train,val,test}/{images,labels}

Destino:
  00_datasets/processed/Dataset_Master/{train,val,test}/{images,labels}
  00_datasets/processed/Dataset_Master/master.yaml
  00_datasets/processed/Dataset_Master/reports/*

Características:
  - No modifica raw/ ni datasets fuente.
  - Renombra archivos con prefijos por dataset.
  - Unifica clases a una sola clase YOLO: person (id 0).
  - Modos de selección: FULL, BALANCED, SMALL_PERSON_PRIORITY, PERSON_PRIORITY.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from config_utils import load_config, path_from_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
DEFAULT_DATASETS = (
    "VisDrone",
    "SeaDronesSee",
    "NOMAD",
    "OKUTAMA",
    "RESDataset",
    "NITC",
    "C2A",
)
DEFAULT_PREFIX = {
    "VisDrone": "VIS",
    "SeaDronesSee": "SEA",
    "NOMAD": "NOM",
    "OKUTAMA": "OKU",
    "RESDataset": "RES",
    "NITC": "NIT",
    "C2A": "C2A",
}
DEFAULT_PERSON_CLASS_IDS = {
    "VisDrone": [0],
    "SeaDronesSee": [0],
    "NOMAD": [0],
    "OKUTAMA": [0],
    "RESDataset": [0],
    "NITC": [0],
    "C2A": [0],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Record:
    dataset: str
    split: str
    image_path: Path
    label_path: Path
    image_ext: str
    person_lines: list[str]
    max_person_area: float


class MasterBuilder:
    def __init__(
        self,
        cfg: dict,
        mode: str,
        seed: int,
        force: bool,
        balance_multiplier: float,
        priority_keep_ratio: float,
        image_storage: str,
    ):
        self.cfg = cfg
        self.mode = mode
        self.seed = seed
        self.force = force
        self.balance_multiplier = balance_multiplier
        self.priority_keep_ratio = priority_keep_ratio
        self.image_storage = image_storage

        self.processed_root = path_from_config(cfg, "datasets_processed")

        ds_cfg = cfg.get("dataset_master", {})
        self.dataset_names = ds_cfg.get("datasets", list(DEFAULT_DATASETS))
        self.prefix_map = {**DEFAULT_PREFIX, **ds_cfg.get("prefixes", {})}

        merged_person = {k: list(v) for k, v in DEFAULT_PERSON_CLASS_IDS.items()}
        merged_person.update({k: list(v) for k, v in ds_cfg.get("person_class_ids", {}).items()})
        self.person_class_ids = merged_person

        self.master_name = ds_cfg.get("master_name", "Dataset_Master")
        self.master_dir = self.processed_root / self.master_name
        self.report_dir = self.master_dir / "reports"

        self.random = random.Random(seed)

    def _materialize_image(self, src: Path, dst: Path) -> None:
        if dst.exists():
            dst.unlink()

        # hardlink evita duplicar bytes en disco cuando origen y destino están en el mismo volumen
        if self.image_storage == "hardlink":
            os.link(src, dst)
            return

        if self.image_storage == "symlink":
            os.symlink(src, dst)
            return

        shutil.copy2(src, dst)

    def _prepare_dirs(self) -> None:
        if self.master_dir.exists() and self.force:
            shutil.rmtree(self.master_dir)
        self.master_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            (self.master_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.master_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _parse_person_lines(self, dataset: str, label_path: Path) -> tuple[list[str], float]:
        allowed_ids = {int(x) for x in self.person_class_ids.get(dataset, [0])}
        person_lines: list[str] = []
        max_area = 0.0

        try:
            raw = label_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], 0.0

        for line in raw.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue

            if cls_id not in allowed_ids:
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                continue

            area = bw * bh
            if area > max_area:
                max_area = area

            # Unificación a clase person=0
            person_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        return person_lines, max_area

    def _collect_records(self) -> list[Record]:
        records: list[Record] = []
        for dataset in self.dataset_names:
            ds_root = self.processed_root / dataset
            if not ds_root.exists():
                logger.warning("Dataset no encontrado en processed/: %s", dataset)
                continue

            for split in SPLITS:
                img_dir = ds_root / split / "images"
                lbl_dir = ds_root / split / "labels"
                if not img_dir.exists() or not lbl_dir.exists():
                    continue

                for image_path in img_dir.iterdir():
                    if image_path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    label_path = lbl_dir / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        continue

                    person_lines, max_area = self._parse_person_lines(dataset, label_path)
                    if not person_lines:
                        continue

                    records.append(
                        Record(
                            dataset=dataset,
                            split=split,
                            image_path=image_path,
                            label_path=label_path,
                            image_ext=image_path.suffix.lower(),
                            person_lines=person_lines,
                            max_person_area=max_area,
                        )
                    )

        return records

    def _group_counts(self, records: list[Record]) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for r in records:
            k = (r.dataset, r.split)
            counts[k] = counts.get(k, 0) + 1
        return counts

    def _select_records(self, records: list[Record]) -> list[Record]:
        if self.mode == "FULL":
            return records

        by_ds_split: dict[tuple[str, str], list[Record]] = {}
        for r in records:
            by_ds_split.setdefault((r.dataset, r.split), []).append(r)

        selected: list[Record] = []

        if self.mode == "BALANCED":
            for split in SPLITS:
                groups = [
                    (k, v) for k, v in by_ds_split.items()
                    if k[1] == split and len(v) > 0
                ]
                if not groups:
                    continue
                min_count = min(len(v) for _, v in groups)
                quota = max(1, int(min_count * self.balance_multiplier))
                for _, recs in groups:
                    picked = recs[:]
                    self.random.shuffle(picked)
                    selected.extend(picked[: min(quota, len(picked))])
            return selected

        if self.mode in {"SMALL_PERSON_PRIORITY", "PERSON_PRIORITY"}:
            reverse = self.mode == "PERSON_PRIORITY"
            keep_ratio = max(0.05, min(1.0, self.priority_keep_ratio))
            for recs in by_ds_split.values():
                ordered = sorted(recs, key=lambda x: x.max_person_area, reverse=reverse)
                keep_n = max(1, int(len(ordered) * keep_ratio))
                selected.extend(ordered[:keep_n])
            return selected

        raise ValueError(f"Modo no soportado: {self.mode}")

    def _write_master(self, records: list[Record]) -> None:
        counters: dict[tuple[str, str], int] = {}

        for rec in records:
            key = (rec.dataset, rec.split)
            counters[key] = counters.get(key, 0) + 1

            prefix = self.prefix_map.get(rec.dataset, rec.dataset[:3].upper())
            new_stem = f"{prefix}_{rec.split}_{counters[key]:07d}"

            out_img = self.master_dir / rec.split / "images" / f"{new_stem}{rec.image_ext}"
            out_lbl = self.master_dir / rec.split / "labels" / f"{new_stem}.txt"

            self._materialize_image(rec.image_path, out_img)
            out_lbl.write_text("\n".join(rec.person_lines) + "\n", encoding="utf-8")

    def _write_master_yaml(self) -> None:
        content = (
            f"path: {str(self.master_dir).replace('\\', '/')}\n"
            "train: train/images\n"
            "val:   val/images\n"
            "test:  test/images\n"
            "nc: 1\n"
            "names:\n"
            "  0: person\n"
        )
        (self.master_dir / "master.yaml").write_text(content, encoding="utf-8")

    def _write_reports(self, all_records: list[Record], selected: list[Record]) -> None:
        before_counts = self._group_counts(all_records)
        after_counts = self._group_counts(selected)

        csv_path = self.report_dir / "dataset_master_report.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "split", "before", "after"])
            for dataset in self.dataset_names:
                for split in SPLITS:
                    writer.writerow([
                        dataset,
                        split,
                        before_counts.get((dataset, split), 0),
                        after_counts.get((dataset, split), 0),
                    ])

        small_thr = 0.01
        payload = {
            "mode": self.mode,
            "seed": self.seed,
            "balance_multiplier": self.balance_multiplier,
            "priority_keep_ratio": self.priority_keep_ratio,
            "datasets": self.dataset_names,
            "records_before": len(all_records),
            "records_after": len(selected),
            "small_person_threshold_area": small_thr,
            "small_person_ratio_after": (
                sum(1 for r in selected if r.max_person_area <= small_thr) / len(selected)
                if selected else 0.0
            ),
            "by_dataset_split": [
                {
                    "dataset": ds,
                    "split": sp,
                    "before": before_counts.get((ds, sp), 0),
                    "after": after_counts.get((ds, sp), 0),
                }
                for ds in self.dataset_names
                for sp in SPLITS
            ],
        }
        (self.report_dir / "dataset_master_report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def build(self) -> int:
        self._prepare_dirs()

        all_records = self._collect_records()
        if not all_records:
            logger.error("No se encontraron muestras válidas con clase person para construir Dataset_Master.")
            return 1

        selected = self._select_records(all_records)
        if not selected:
            logger.error("La estrategia de selección dejó 0 muestras. Ajusta los parámetros del modo.")
            return 1

        self._write_master(selected)
        self._write_master_yaml()
        self._write_reports(all_records, selected)

        logger.info("Dataset_Master generado en: %s", self.master_dir)
        logger.info("Modo: %s | Muestras antes: %d | después: %d", self.mode, len(all_records), len(selected))
        logger.info("Reporte: %s", self.report_dir)
        return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye Dataset_Master para entrenamiento único SAR.")
    parser.add_argument(
        "--mode",
        choices=["FULL", "BALANCED", "SMALL_PERSON_PRIORITY", "PERSON_PRIORITY"],
        default="FULL",
        help="Modo de selección de muestras.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Si existe Dataset_Master, lo elimina y reconstruye.",
    )
    parser.add_argument(
        "--balance-multiplier",
        type=float,
        default=1.0,
        help="Solo BALANCED: cuota = min_count * multiplier.",
    )
    parser.add_argument(
        "--priority-keep-ratio",
        type=float,
        default=0.7,
        help="Solo PRIORITY: fracción de muestras que se conserva por dataset/split.",
    )
    parser.add_argument(
        "--image-storage",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="Cómo materializar imágenes en Dataset_Master. hardlink minimiza uso de disco.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config()
    builder = MasterBuilder(
        cfg=cfg,
        mode=args.mode,
        seed=args.seed,
        force=args.force,
        balance_multiplier=args.balance_multiplier,
        priority_keep_ratio=args.priority_keep_ratio,
        image_storage=args.image_storage,
    )
    return builder.build()


if __name__ == "__main__":
    raise SystemExit(main())
