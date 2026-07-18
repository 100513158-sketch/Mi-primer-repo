"""Entrenamiento YOLO con configuracion centralizada en config.yaml."""

from pathlib import Path
import logging
import torch
from ultralytics import YOLO

from config_utils import load_config, path_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CFG = load_config()
DATASET_CONFIG = path_from_config(CFG, "dataset_config")
OUTPUT_DIR = path_from_config(CFG, "training_output")
WEIGHTS_DIR = path_from_config(CFG, "weights_dir")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_yolo_model(model_name: str) -> YOLO:
    try:
        return YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar el modelo '{model_name}'. "
            "Este proyecto requiere YOLO26 y no usa fallback a YOLO11."
        ) from exc


def check_gpu():
    mem_cfg = CFG.get("training", {}).get("memory", {})
    max_batch_gpu = int(mem_cfg.get("max_batch_gpu", 32))
    workers_gpu = int(mem_cfg.get("workers_gpu", 4))
    workers_cpu = int(mem_cfg.get("workers_cpu", 2))
    target_gb = float(mem_cfg.get("max_gpu_memory_gb", 0))

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU detectada: %s - %.1f GB", gpu_name, gpu_memory)

        if target_gb > 0:
            fraction = max(0.05, min(0.98, target_gb / gpu_memory))
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                logger.info(
                    "Limite VRAM por proceso aplicado: %.2f GB (%.1f%% de %.2f GB)",
                    target_gb,
                    fraction * 100.0,
                    gpu_memory,
                )
            except Exception as exc:
                logger.warning("No se pudo aplicar limite VRAM: %s", exc)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True, max_batch_gpu, workers_gpu

    logger.warning("No se detecto GPU, se usara CPU")
    return False, 8, workers_cpu


def train_detection():
    logger.info("INICIANDO ENTRENAMIENTO DE DETECCION")
    has_gpu, batch_size, workers = check_gpu()
    det_cfg = CFG["training"]["detection"]

    model_name = det_cfg["model"]
    model = _load_yolo_model(model_name)

    effective_batch = min(int(det_cfg["batch_size"]), int(batch_size))

    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=det_cfg["epochs"],
        batch=effective_batch,
        imgsz=det_cfg["imgsz"],
        workers=workers,
        optimizer=det_cfg.get("optimizer", "auto"),
        lr0=det_cfg["lr0"],
        lrf=det_cfg["lrf"],
        momentum=det_cfg["momentum"],
        weight_decay=det_cfg["weight_decay"],
        warmup_epochs=det_cfg["warmup_epochs"],
        warmup_momentum=0.8,
        mosaic=det_cfg["mosaic"],
        mixup=det_cfg["mixup"],
        copy_paste=det_cfg["copy_paste"],
        degrees=0.0,
        translate=0.1,
        scale=det_cfg["scale"],
        shear=0.0,
        perspective=0.0005,
        fliplr=det_cfg["fliplr"],
        flipud=det_cfg["flipud"],
        hsv_h=det_cfg["hsv_h"],
        hsv_s=det_cfg["hsv_s"],
        hsv_v=det_cfg["hsv_v"],
        dropout=0.0,
        label_smoothing=0.0,
        project=str(OUTPUT_DIR / "detection"),
        name="yolo26m_detection",
        exist_ok=True,
        device=0 if has_gpu else "cpu",
        amp=True,
        patience=det_cfg["patience"],
        save=True,
        save_period=10,
    )

    best = OUTPUT_DIR / "detection/yolo26m_detection/weights/best.pt"
    if best.exists():
        import shutil
        shutil.copy2(best, WEIGHTS_DIR / "best_detection.pt")
        logger.info("Modelo guardado en %s", WEIGHTS_DIR / "best_detection.pt")

    return results


def train_pose():
    logger.info("INICIANDO ENTRENAMIENTO DE POSE")
    has_gpu, batch_size, workers = check_gpu()
    pose_cfg = CFG["training"]["pose"]

    model_name = pose_cfg["model"]
    model = _load_yolo_model(model_name)

    effective_batch = min(int(pose_cfg["batch_size"]), int(batch_size))

    results = model.train(
        data=str(DATASET_CONFIG),
        task="pose",
        epochs=pose_cfg["epochs"],
        batch=max(1, effective_batch),
        imgsz=pose_cfg["imgsz"],
        workers=workers,
        optimizer=pose_cfg.get("optimizer", "auto"),
        lr0=pose_cfg["lr0"],
        lrf=pose_cfg["lrf"],
        mosaic=pose_cfg["mosaic"],
        mixup=pose_cfg["mixup"],
        copy_paste=pose_cfg["copy_paste"],
        scale=pose_cfg["scale"],
        fliplr=pose_cfg["fliplr"],
        project=str(OUTPUT_DIR / "pose"),
        name="yolo26m_pose",
        exist_ok=True,
        device=0 if has_gpu else "cpu",
        amp=True,
        patience=pose_cfg["patience"],
    )

    best = OUTPUT_DIR / "pose/yolo26m_pose/weights/best.pt"
    if best.exists():
        import shutil
        shutil.copy2(best, WEIGHTS_DIR / "best_pose.pt")
        logger.info("Modelo guardado en %s", WEIGHTS_DIR / "best_pose.pt")

    return results


def transfer_learning_from_visdrone():
    logger.info("INICIANDO TRANSFER LEARNING")
    has_gpu, batch_size, workers = check_gpu()
    tl_cfg = CFG["training"]["transfer_learning"]

    base_weights = tl_cfg["source_weights"]
    model = _load_yolo_model(base_weights)

    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=tl_cfg["epochs"],
        batch=batch_size,
        imgsz=CFG["training"]["detection"]["imgsz"],
        workers=workers,
        optimizer=CFG["training"]["detection"].get("optimizer", "auto"),
        lr0=tl_cfg["lr0"],
        lrf=0.001,
        project=str(OUTPUT_DIR / "transfer"),
        name="visdrone_pretrained",
        exist_ok=True,
        device=0 if has_gpu else "cpu",
        amp=True,
        freeze=tl_cfg["freeze_layers"],
    )

    return results


def main():
    logger.info("Iniciando pipeline de entrenamiento completo")
    if CFG["training"]["transfer_learning"].get("enabled", False):
        transfer_learning_from_visdrone()
    train_detection()
    train_pose()
    logger.info("Entrenamiento completado")


if __name__ == "__main__":
    main()
