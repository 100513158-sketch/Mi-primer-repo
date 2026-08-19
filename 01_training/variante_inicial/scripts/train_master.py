"""
train_master.py — Entrena un único modelo sobre Dataset_Master.

Flujo:
  1) Carga config.yaml
  2) Usa 00_datasets/processed/Dataset_Master/master.yaml
  3) Entrena YOLO26 con hiperparámetros de training.detection
  4) Guarda best_Dataset_Master.pt y metrics CSV en 02_models/weights
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from config_utils import load_config, path_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _setup_gpu(cfg: dict) -> tuple[bool, int, int]:
    mem_cfg = cfg.get("training", {}).get("memory", {})
    max_batch_gpu = int(mem_cfg.get("max_batch_gpu", 32))
    workers_gpu = int(mem_cfg.get("workers_gpu", 4))
    workers_cpu = int(mem_cfg.get("workers_cpu", 2))
    target_gb = float(mem_cfg.get("max_gpu_memory_gb", 0))

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB)", name, vram)

        if target_gb > 0:
            fraction = max(0.05, min(0.98, target_gb / vram))
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                logger.info("Limite VRAM por proceso aplicado: %.2f GB", target_gb)
            except Exception as exc:
                logger.warning("No se pudo aplicar límite VRAM: %s", exc)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True, max_batch_gpu, workers_gpu

    logger.warning("No se detectó GPU: usando CPU")
    return False, 8, workers_cpu


def _load_model(model_path: str) -> YOLO:
    try:
        return YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar el modelo '{model_path}'. Este flujo requiere YOLO26."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena un modelo único sobre Dataset_Master.")
    parser.add_argument("--dataset-name", default="Dataset_Master", help="Carpeta en processed/")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch")
    parser.add_argument("--imgsz", type=int, default=None, help="Override imgsz")
    parser.add_argument("--freeze", type=int, default=0, help="Capas a congelar")
    parser.add_argument("--name", default="yolo26m_master", help="Nombre de run")
    args = parser.parse_args()

    cfg = load_config()
    det_cfg = cfg["training"]["detection"]

    processed_root = path_from_config(cfg, "datasets_processed")
    output_root = path_from_config(cfg, "training_output") / "detection"
    weights_dir = path_from_config(cfg, "weights_dir")
    weights_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = processed_root / args.dataset_name
    data_yaml = dataset_root / "master.yaml"
    if not data_yaml.exists():
        logger.error("No existe master.yaml: %s", data_yaml)
        logger.error("Ejecuta primero build_dataset_master.py")
        return 1

    base_model = cfg.get("training_pipeline", {}).get("base_model", det_cfg["model"])
    model = _load_model(base_model)

    has_gpu, max_batch, workers = _setup_gpu(cfg)

    epochs = int(args.epochs if args.epochs is not None else det_cfg["epochs"])
    batch = int(args.batch if args.batch is not None else det_cfg["batch_size"])
    imgsz = int(args.imgsz if args.imgsz is not None else det_cfg["imgsz"])
    batch = min(batch, max_batch)

    logger.info("Entrenamiento Master")
    logger.info("Dataset: %s", data_yaml)
    logger.info("Modelo : %s", base_model)
    logger.info("Epochs : %d | Batch: %d | ImgSz: %d", epochs, batch, imgsz)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "optimizer": det_cfg.get("optimizer", "auto"),
        "lr0": det_cfg["lr0"],
        "lrf": det_cfg["lrf"],
        "momentum": det_cfg["momentum"],
        "weight_decay": det_cfg["weight_decay"],
        "warmup_epochs": det_cfg["warmup_epochs"],
        "mosaic": det_cfg["mosaic"],
        "mixup": det_cfg["mixup"],
        "copy_paste": det_cfg["copy_paste"],
        "scale": det_cfg["scale"],
        "fliplr": det_cfg["fliplr"],
        "flipud": det_cfg["flipud"],
        "hsv_h": det_cfg["hsv_h"],
        "hsv_s": det_cfg["hsv_s"],
        "hsv_v": det_cfg["hsv_v"],
        "patience": det_cfg["patience"],
        "project": str(output_root),
        "name": args.name,
        "exist_ok": True,
        "device": 0 if has_gpu else "cpu",
        "amp": True,
        "save": True,
        "save_period": 10,
    }
    if args.freeze > 0:
        train_kwargs["freeze"] = int(args.freeze)

    results = model.train(**train_kwargs)

    best_src = output_root / args.name / "weights" / "best.pt"
    if not best_src.exists():
        logger.error("No se encontró best.pt tras entrenamiento: %s", best_src)
        return 1

    best_dst = weights_dir / "best_Dataset_Master.pt"
    shutil.copy2(best_src, best_dst)
    logger.info("Pesos finales: %s", best_dst)

    results_csv = output_root / args.name / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, weights_dir / "results_Dataset_Master.csv")
        logger.info("Métricas CSV: %s", weights_dir / "results_Dataset_Master.csv")

    try:
        metrics = results.results_dict
        logger.info(
            "mAP50=%.4f | mAP50-95=%.4f",
            metrics.get("metrics/mAP50(B)", 0.0),
            metrics.get("metrics/mAP50-95(B)", 0.0),
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
