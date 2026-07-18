"""
Utilidades para cargar config.yaml central del proyecto.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def project_root() -> Path:
    # 01_training/scripts/config_utils.py -> C:/SARC-Drone
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    return project_root() / "config.yaml"


def load_config() -> Dict[str, Any]:
    cfg_file = config_path()
    if not cfg_file.exists():
        raise FileNotFoundError(f"No se encontro config.yaml en {cfg_file}")
    with cfg_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_from_config(cfg: Dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])
