"""
export_model.py - Exporta modelos a TFLite, ONNX y OpenVINO.
"""

from pathlib import Path
import logging
from ultralytics import YOLO

from config_utils import load_config, path_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CFG = load_config()
WEIGHTS_DIR = path_from_config(CFG, "weights_dir")
EXPORT_DIR = path_from_config(CFG, "export_dir")
DATASET_CONFIG = path_from_config(CFG, "dataset_config")


def _find_new_file(pattern: str):
    for f in Path(".").glob(pattern):
        return f
    return None


def _resolve_detection_weights() -> Path | None:
    candidates = [
        WEIGHTS_DIR / "best_C2A.pt",
        WEIGHTS_DIR / "best_detection.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    stage_candidates = sorted(WEIGHTS_DIR.glob("best_*.pt"))
    for candidate in reversed(stage_candidates):
        if candidate.name != "best_pose.pt":
            return candidate
    return None


def export_to_tflite(model_path: Path):
    logger.info("Exportando %s a TFLite", model_path)
    model = YOLO(str(model_path))
    tflite_cfg = CFG["export"]["tflite"]

    args = {
        "format": "tflite",
        "imgsz": tflite_cfg["imgsz"],
        "batch": 1,
        "nms": tflite_cfg["nms"],
    }
    if tflite_cfg["int8"]:
        args["int8"] = True
        args["data"] = str(DATASET_CONFIG)

    model.export(**args)

    tflite_file = _find_new_file("*.tflite")
    if tflite_file:
        dest = EXPORT_DIR / "tflite" / model_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"{model_path.stem}.tflite"
        import shutil
        shutil.copy2(tflite_file, out)
        logger.info("TFLite exportado a %s", out)
        return out
    return None


def export_to_onnx(model_path: Path):
    logger.info("Exportando %s a ONNX", model_path)
    model = YOLO(str(model_path))
    onnx_cfg = CFG["export"]["onnx"]
    model.export(
        format="onnx",
        imgsz=onnx_cfg["imgsz"],
        batch=1,
        dynamic=onnx_cfg["dynamic"],
        simplify=True,
        opset=onnx_cfg["opset"],
    )

    onnx_file = _find_new_file("*.onnx")
    if onnx_file:
        dest = EXPORT_DIR / "onnx"
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"{model_path.stem}.onnx"
        import shutil
        shutil.copy2(onnx_file, out)
        logger.info("ONNX exportado a %s", out)
        return out
    return None


def export_to_openvino(model_path: Path):
    logger.info("Exportando %s a OpenVINO", model_path)
    model = YOLO(str(model_path))
    ov_cfg = CFG["export"]["openvino"]
    model.export(format="openvino", imgsz=CFG["export"]["tflite"]["imgsz"], half=ov_cfg["half"], int8=False)

    openvino_dir = None
    for d in Path(".").glob("*_openvino_model"):
        openvino_dir = d
        break

    if openvino_dir:
        dest = EXPORT_DIR / "openvino" / model_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(openvino_dir, dest, dirs_exist_ok=True)
        logger.info("OpenVINO exportado a %s", dest)
        return dest
    return None


def optimize_for_mobile(model_path: Path):
    export_to_tflite(model_path)
    export_to_onnx(model_path)


def main():
    detection = _resolve_detection_weights()
    pose = WEIGHTS_DIR / "best_pose.pt"

    if detection and detection.exists():
        optimize_for_mobile(detection)
        export_to_openvino(detection)
    else:
        logger.warning("No se encontraron pesos de deteccion para exportar en %s", WEIGHTS_DIR)

    if pose.exists():
        optimize_for_mobile(pose)
        export_to_openvino(pose)


if __name__ == "__main__":
    main()
