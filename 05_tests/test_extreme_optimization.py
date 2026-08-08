import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_training" / "scripts"))

import extreme_optimization as eo


def test_resolve_calibration_data_uses_dataset_yaml_for_staged_checkpoint():
    model_path = Path("C:/SARC-Drone/02_models/weights/best_C2A.pt")
    calibration_data = eo.resolve_calibration_data(model_path)

    assert calibration_data.exists()
    assert calibration_data.parent.name == "C2A"
    assert calibration_data.name == "data.yaml"
