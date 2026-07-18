"""
extreme_optimization.py - Optimizacion para despliegue movil.
"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from config_utils import load_config, path_from_config

CFG = load_config()
WEIGHTS_DIR = path_from_config(CFG, "weights_dir")
EXPORT_DIR = path_from_config(CFG, "export_dir")
DATASET_CONFIG = path_from_config(CFG, "dataset_config")


def resolve_detection_weights() -> Path | None:
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


def quantize_to_int8_onnx(onnx_model_path: Path):
    quantized_path = onnx_model_path.with_suffix("").with_suffix(".quantized.onnx")
    quantize_dynamic(
        model_input=str(onnx_model_path),
        model_output=str(quantized_path),
        per_channel=True,
        reduce_range=True,
        weight_type=QuantType.QInt8,
    )
    return quantized_path


def create_optimized_tflite(model_path: Path, output_path: Path, int8=True):
    model = YOLO(str(model_path))
    tflite_cfg = CFG["export"]["tflite"]
    args = {
        "format": "tflite",
        "imgsz": tflite_cfg["imgsz"],
        "batch": 1,
        "nms": tflite_cfg["nms"],
    }
    if int8:
        args["int8"] = True
        args["data"] = str(DATASET_CONFIG)

    model.export(**args)

    import shutil
    for f in Path(".").glob("*.tflite"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(output_path))
        return output_path
    return None


def benchmark_onnx(model_path: Path, input_shape=(1, 3, 640, 640)):
    import time

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)

    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    times = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - start)

    avg_ms = float(np.mean(times) * 1000)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    print(f"ONNX Runtime - Avg: {avg_ms:.2f}ms, FPS: {fps:.1f}")


def main():
    model_path = resolve_detection_weights()
    if not model_path:
        print(f"Modelo no encontrado en {WEIGHTS_DIR}")
        return

    optimized_tflite = EXPORT_DIR / "tflite/production_model.tflite"
    create_optimized_tflite(model_path, optimized_tflite, int8=True)
    print("Optimizacion completada")


if __name__ == "__main__":
    main()
