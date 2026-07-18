"""
train_pipeline.py — Pipeline de entrenamiento curricular secuencial SARC-Drone.

Entrena cada dataset de forma independiente, en orden de menor a mayor
especificidad SARC, con transferencia de pesos entre etapas.

Flujo automático:
  1. Verifica integridad de todos los datasets del pipeline
  2. Genera data.yaml individuales si faltan
  3. Entrena etapa por etapa; si una falla, continúa con el último
     checkpoint válido disponible

Uso:
  python train_pipeline.py                        # pipeline completo
  python train_pipeline.py --start-from NITC      # reanudar desde etapa
  python train_pipeline.py --only SeaDronesSee    # solo una etapa
  python train_pipeline.py --dry-run              # simula sin entrenar

Salidas por etapa:
  01_training/runs/detection/<DATASET>/
  02_models/weights/best_<DATASET>.pt
  02_models/weights/results_<DATASET>.csv
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from config_utils import load_config, path_from_config
from integrity_check import check_all_datasets
from generate_dataset_yamls import ensure_dataset_yamls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DIV = "=" * 56


# ─── GPU ─────────────────────────────────────────────────────────────────────

def _setup_gpu(cfg: dict) -> tuple[bool, int, int]:
    """Detecta GPU y ajusta backends. Retorna (has_gpu, max_batch, workers)."""
    mem_cfg = cfg.get("training", {}).get("memory", {})
    max_batch_gpu = int(mem_cfg.get("max_batch_gpu", 32))
    workers_gpu = int(mem_cfg.get("workers_gpu", 4))
    workers_cpu = int(mem_cfg.get("workers_cpu", 2))
    target_gb = float(mem_cfg.get("max_gpu_memory_gb", 0))

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB VRAM)", name, vram)

        if target_gb > 0:
            fraction = max(0.05, min(0.98, target_gb / vram))
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                logger.info(
                    "Limite VRAM por proceso aplicado: %.2f GB (%.1f%% de %.2f GB)",
                    target_gb,
                    fraction * 100.0,
                    vram,
                )
            except Exception as exc:
                logger.warning("No se pudo aplicar limite VRAM: %s", exc)

        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        return True, max_batch_gpu, workers_gpu
    logger.warning("No se detectó GPU — usando CPU")
    return False, 8, workers_cpu


# ─── Etapa de entrenamiento ───────────────────────────────────────────────────

def _run_stage(
    dataset_name: str,
    dataset_yaml: Path,
    base_weights: str,
    stage_cfg: dict,
    det_cfg: dict,
    output_dir: Path,
    weights_dir: Path,
    has_gpu: bool,
    max_batch: int,
    workers: int,
    dry_run: bool,
) -> tuple[bool, Path | None]:
    """
    Entrena una etapa del pipeline.

    Retorna (success, new_weights_path).
    Si falla, new_weights_path es None y el llamador usará el último checkpoint válido.
    """
    run_name    = f"yolo26m_{dataset_name}"
    project_dir = output_dir / "detection"
    best_src    = project_dir / run_name / "weights" / "best.pt"
    dest        = weights_dir / f"best_{dataset_name}.pt"

    effective_batch = min(stage_cfg.get("batch_size", 16), max_batch)
    freeze          = stage_cfg.get("freeze", 0)
    epochs          = stage_cfg.get("epochs", 100)
    lr0             = stage_cfg.get("lr0", det_cfg["lr0"])
    lrf             = stage_cfg.get("lrf", det_cfg["lrf"])

    logger.info(_DIV)
    logger.info("Dataset : %s", dataset_name)
    logger.info("Pesos   : %s", Path(base_weights).name)
    logger.info("Data    : %s", dataset_yaml)
    logger.info("Epochs  : %d  |  Batch: %d  |  Freeze: %d capas", epochs, effective_batch, freeze)
    logger.info("lr0     : %s  |  lrf: %s", lr0, lrf)
    logger.info(_DIV)

    if dry_run:
        logger.info("[DRY-RUN] Etapa omitida.")
        p = Path(base_weights)
        return True, p if p.exists() else None

    if not dataset_yaml.exists():
        logger.error("data.yaml no encontrado: %s", dataset_yaml)
        return False, None

    try:
        model = YOLO(base_weights)

        train_kwargs: dict = {
            "data"         : str(dataset_yaml),
            "epochs"       : epochs,
            "batch"        : effective_batch,
            "imgsz"        : stage_cfg.get("imgsz", det_cfg["imgsz"]),
            "workers"      : workers,
            "optimizer"    : det_cfg.get("optimizer", "auto"),
            "lr0"          : lr0,
            "lrf"          : lrf,
            "momentum"     : det_cfg["momentum"],
            "weight_decay" : det_cfg["weight_decay"],
            "warmup_epochs": det_cfg["warmup_epochs"],
            "mosaic"       : det_cfg["mosaic"],
            "mixup"        : det_cfg["mixup"],
            "copy_paste"   : det_cfg["copy_paste"],
            "scale"        : det_cfg["scale"],
            "fliplr"       : det_cfg["fliplr"],
            "flipud"       : det_cfg["flipud"],
            "hsv_h"        : det_cfg["hsv_h"],
            "hsv_s"        : det_cfg["hsv_s"],
            "hsv_v"        : det_cfg["hsv_v"],
            "patience"     : det_cfg["patience"],
            "project"      : str(project_dir),
            "name"         : run_name,
            "exist_ok"     : True,
            "device"       : 0 if has_gpu else "cpu",
            "amp"          : True,
            "save"         : True,
            "save_period"  : 10,
        }
        if freeze > 0:
            train_kwargs["freeze"] = freeze

        results = model.train(**train_kwargs)

        if not best_src.exists():
            logger.error("best.pt no generado tras entrenamiento: %s", best_src)
            return False, None

        shutil.copy2(best_src, dest)
        logger.info("Pesos guardados: %s", dest)

        # Copiar results.csv al directorio de pesos
        results_csv = best_src.parent.parent / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, weights_dir / f"results_{dataset_name}.csv")

        # Métricas finales
        try:
            m       = results.results_dict
            map50   = m.get("metrics/mAP50(B)", 0.0)
            map5095 = m.get("metrics/mAP50-95(B)", 0.0)
            logger.info("Métricas finales — mAP50: %.3f  mAP50-95: %.3f", map50, map5095)
        except Exception:
            pass

        return True, dest

    except Exception as exc:
        logger.error("Error en etapa '%s': %s", dataset_name, exc, exc_info=True)
        return False, None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento curricular SARC-Drone."
    )
    parser.add_argument(
        "--start-from", metavar="DATASET",
        help="Reanudar pipeline desde este dataset usando el checkpoint previo si existe.",
    )
    parser.add_argument(
        "--only", metavar="DATASET",
        help="Ejecutar únicamente la etapa de este dataset.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula el pipeline completo sin entrenar realmente.",
    )
    args = parser.parse_args()

    cfg          = load_config()
    pipeline_cfg = cfg.get("training_pipeline", {})
    full_order: list[str] = pipeline_cfg.get("order", [])

    if not full_order:
        logger.error("'training_pipeline.order' no encontrado en config.yaml")
        return 1

    processed_root = path_from_config(cfg, "datasets_processed")
    output_dir     = path_from_config(cfg, "training_output")
    weights_dir    = path_from_config(cfg, "weights_dir")
    weights_dir.mkdir(parents=True, exist_ok=True)

    det_cfg = cfg["training"]["detection"]

    # ── Modelo base (YOLO26 obligatorio) ──────────────────────────────────────
    base_model = pipeline_cfg.get("base_model", "yolo26m.pt")

    # ── Determinar etapas a ejecutar ───────────────────────────────────────────
    if args.only:
        if args.only not in full_order:
            logger.error("'%s' no está en training_pipeline.order de config.yaml", args.only)
            return 1
        stages = [args.only]
    elif args.start_from:
        if args.start_from not in full_order:
            logger.error("'%s' no está en training_pipeline.order de config.yaml", args.start_from)
            return 1
        stages = full_order[full_order.index(args.start_from):]
    else:
        stages = list(full_order)

    # ── Paso 1: Verificación de integridad ─────────────────────────────────────
    if not check_all_datasets(processed_root, dataset_names=stages):
        logger.error("Pipeline abortado — corrige los errores de integridad antes de continuar.")
        return 1

    # ── Paso 2: Generar data.yaml faltantes ────────────────────────────────────
    logger.info("Verificando data.yaml individuales...")
    ensure_dataset_yamls(cfg, dataset_names=stages, overwrite=False)

    # ── GPU ────────────────────────────────────────────────────────────────────
    has_gpu, max_batch, workers = _setup_gpu(cfg)

    # ── Validación temprana del modelo base (YOLO26 obligatorio) ──────────────
    try:
        YOLO(base_model)
    except Exception as exc:
        logger.error(
            "No se pudo cargar el modelo base '%s'. Este pipeline requiere YOLO26 y no usa fallback a YOLO11.",
            base_model,
        )
        logger.error("Detalle: %s", exc)
        return 1

    # ── Pesos de inicio del pipeline ───────────────────────────────────────────
    # Si se reanuda con --start-from, buscar el checkpoint previo más reciente
    current_weights = base_model
    if args.start_from and not args.only:
        start_idx = full_order.index(args.start_from)
        for ds in full_order[:start_idx]:
            candidate = weights_dir / f"best_{ds}.pt"
            if candidate.exists():
                current_weights = str(candidate)
        if current_weights != base_model:
            logger.info("Reanudando con checkpoint previo: %s", Path(current_weights).name)

    # ── Resumen de inicio ──────────────────────────────────────────────────────
    logger.info("\n%s", _DIV)
    logger.info("SARC-Drone — Pipeline curricular de entrenamiento")
    logger.info("Etapas (%d): %s", len(stages), " -> ".join(stages))
    logger.info("Modelo base : %s", base_model)
    logger.info("Dry-run     : %s", args.dry_run)
    logger.info("%s\n", _DIV)

    passed        = 0
    failed_stages: list[str] = []
    t0_global     = time.time()

    for stage_idx, dataset_name in enumerate(stages, 1):
        logger.info("\n[%d/%d] Iniciando etapa: %s", stage_idx, len(stages), dataset_name)
        t0 = time.time()

        dataset_yaml = processed_root / dataset_name / "data.yaml"
        stage_cfg    = pipeline_cfg.get("per_dataset", {}).get(dataset_name, {})

        success, new_weights = _run_stage(
            dataset_name  = dataset_name,
            dataset_yaml  = dataset_yaml,
            base_weights  = current_weights,
            stage_cfg     = stage_cfg,
            det_cfg       = det_cfg,
            output_dir    = output_dir,
            weights_dir   = weights_dir,
            has_gpu       = has_gpu,
            max_batch     = max_batch,
            workers       = workers,
            dry_run       = args.dry_run,
        )

        elapsed_min = (time.time() - t0) / 60.0

        if success and new_weights and new_weights.exists():
            current_weights = str(new_weights)
            passed += 1
            logger.info(
                "[%d/%d] OK — %s completado en %.1f min | Pesos: %s",
                stage_idx, len(stages), dataset_name, elapsed_min, new_weights.name,
            )
        else:
            failed_stages.append(dataset_name)
            logger.warning(
                "[%d/%d] FAIL — %s falló (%.1f min) | Continuando con: %s",
                stage_idx, len(stages), dataset_name, elapsed_min, Path(current_weights).name,
            )

    total_min = (time.time() - t0_global) / 60.0

    logger.info("\n%s", _DIV)
    logger.info("Pipeline finalizado en %.1f min", total_min)
    logger.info("Etapas OK   : %d / %d", passed, len(stages))
    if failed_stages:
        logger.warning("Etapas FAIL : %s", ", ".join(failed_stages))
    logger.info("Mejor modelo: %s", Path(current_weights).name)
    logger.info(_DIV)

    # Código de salida: 0 = todo OK, 2 = completado con fallos parciales
    return 0 if not failed_stages else 2


if __name__ == "__main__":
    sys.exit(main())
