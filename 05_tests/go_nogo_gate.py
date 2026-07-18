"""
Evaluador Go/No-Go para SARC-Drone.

Evalua el estado del pipeline curricular y los artefactos de despliegue usando
umbrales definidos en config.yaml (seccion go_nogo.profiles).

Uso:
  python 05_tests/go_nogo_gate.py
  python 05_tests/go_nogo_gate.py --profile release_edge_android
  python 05_tests/go_nogo_gate.py --json

Codigos de salida:
  0 -> GO
  2 -> NO-GO
  3 -> PENDING (pipeline en progreso o artefactos aun no generados)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "01_training" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config_utils import load_config, path_from_config  # type: ignore


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _collect_metrics(rows: list[dict[str, str]], mean_last_n: int) -> dict[str, float]:
    if not rows:
        return {
            "final_precision": 0.0,
            "final_recall": 0.0,
            "final_map50": 0.0,
            "final_map50_95": 0.0,
            "peak_map50": 0.0,
            "mean_map50_last_n": 0.0,
        }

    last = rows[-1]
    map50_values = [_to_float(r.get("metrics/mAP50(B)"), 0.0) for r in rows]

    n = max(1, min(mean_last_n, len(rows)))
    mean_last = sum(map50_values[-n:]) / n

    return {
        "final_precision": _to_float(last.get("metrics/precision(B)"), 0.0),
        "final_recall": _to_float(last.get("metrics/recall(B)"), 0.0),
        "final_map50": _to_float(last.get("metrics/mAP50(B)"), 0.0),
        "final_map50_95": _to_float(last.get("metrics/mAP50-95(B)"), 0.0),
        "peak_map50": max(map50_values) if map50_values else 0.0,
        "mean_map50_last_n": mean_last,
    }


def evaluate(profile: str) -> tuple[str, dict[str, Any]]:
    cfg = load_config()
    weights_dir = path_from_config(cfg, "weights_dir")
    export_dir = path_from_config(cfg, "export_dir")

    go_nogo_cfg = cfg.get("go_nogo", {})
    profiles = go_nogo_cfg.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Perfil Go/No-Go no encontrado: {profile}")

    p = profiles[profile]
    required_stages: list[str] = list(p.get("required_stage_results", []))
    final_stage = str(p.get("final_stage", "C2A"))
    required_weight = str(p.get("required_final_weight", f"best_{final_stage}.pt"))
    mean_last_n = int(p.get("mean_map50_last_n", 5))

    required_csvs = [weights_dir / f"results_{stage}.csv" for stage in required_stages]
    missing_csvs = [str(path.name) for path in required_csvs if not path.exists()]

    final_csv = weights_dir / f"results_{final_stage}.csv"
    final_weight = weights_dir / required_weight

    pending_reasons: list[str] = []
    if missing_csvs:
        pending_reasons.append(
            "Faltan resultados de etapas: " + ", ".join(missing_csvs)
        )
    if not final_csv.exists():
        pending_reasons.append(f"No existe {final_csv.name}")
    if not final_weight.exists():
        pending_reasons.append(f"No existe {final_weight.name}")

    exports_required = bool(p.get("require_exports", False))
    missing_exports: list[str] = []
    required_export_paths: list[str] = list(p.get("required_export_paths", []))
    if exports_required:
        for rel in required_export_paths:
            export_path = export_dir / rel
            if not export_path.exists():
                missing_exports.append(rel)
        if missing_exports:
            pending_reasons.append(
                "Faltan exportaciones requeridas: " + ", ".join(missing_exports)
            )

    if pending_reasons:
        return "PENDING", {
            "profile": profile,
            "status": "PENDING",
            "reasons": pending_reasons,
            "final_stage": final_stage,
        }

    rows = _read_csv_rows(final_csv)
    metrics = _collect_metrics(rows, mean_last_n)

    thresholds = {
        "min_final_precision": _to_float(p.get("min_final_precision"), 0.0),
        "min_final_recall": _to_float(p.get("min_final_recall"), 0.0),
        "min_final_map50": _to_float(p.get("min_final_map50"), 0.0),
        "min_final_map50_95": _to_float(p.get("min_final_map50_95"), 0.0),
        "max_map50_drop_vs_peak": _to_float(p.get("max_map50_drop_vs_peak"), 1.0),
        "min_mean_map50_last_n": _to_float(p.get("min_mean_map50_last_n"), 0.0),
    }

    map50_drop = max(0.0, metrics["peak_map50"] - metrics["final_map50"])

    failures: list[str] = []
    if metrics["final_precision"] < thresholds["min_final_precision"]:
        failures.append(
            f"precision final {metrics['final_precision']:.4f} < {thresholds['min_final_precision']:.4f}"
        )
    if metrics["final_recall"] < thresholds["min_final_recall"]:
        failures.append(
            f"recall final {metrics['final_recall']:.4f} < {thresholds['min_final_recall']:.4f}"
        )
    if metrics["final_map50"] < thresholds["min_final_map50"]:
        failures.append(
            f"mAP50 final {metrics['final_map50']:.4f} < {thresholds['min_final_map50']:.4f}"
        )
    if metrics["final_map50_95"] < thresholds["min_final_map50_95"]:
        failures.append(
            f"mAP50-95 final {metrics['final_map50_95']:.4f} < {thresholds['min_final_map50_95']:.4f}"
        )
    if map50_drop > thresholds["max_map50_drop_vs_peak"]:
        failures.append(
            f"caida mAP50 ({map50_drop:.4f}) > max permitido ({thresholds['max_map50_drop_vs_peak']:.4f})"
        )
    if metrics["mean_map50_last_n"] < thresholds["min_mean_map50_last_n"]:
        failures.append(
            f"media mAP50 ultimas {mean_last_n} epocas {metrics['mean_map50_last_n']:.4f} < {thresholds['min_mean_map50_last_n']:.4f}"
        )

    status = "GO" if not failures else "NO-GO"
    report = {
        "profile": profile,
        "status": status,
        "final_stage": final_stage,
        "final_weight": str(final_weight),
        "metrics": metrics,
        "thresholds": thresholds,
        "map50_drop_vs_peak": map50_drop,
        "failures": failures,
        "exports_checked": exports_required,
        "missing_exports": missing_exports,
    }
    return status, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluador Go/No-Go SARC-Drone")
    parser.add_argument(
        "--profile",
        default="model_candidate",
        help="Perfil de umbrales en config.yaml -> go_nogo.profiles",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime reporte completo en JSON",
    )
    args = parser.parse_args()

    try:
        status, report = evaluate(args.profile)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=" * 64)
        print("SARC-Drone Go/No-Go")
        print("=" * 64)
        print(f"Perfil      : {report.get('profile')}")
        print(f"Estado      : {report.get('status')}")
        print(f"Etapa final : {report.get('final_stage')}")

        if status == "PENDING":
            for reason in report.get("reasons", []):
                print(f" - {reason}")
            return 3

        m = report.get("metrics", {})
        print(f"Precision   : {m.get('final_precision', 0.0):.4f}")
        print(f"Recall      : {m.get('final_recall', 0.0):.4f}")
        print(f"mAP50       : {m.get('final_map50', 0.0):.4f}")
        print(f"mAP50-95    : {m.get('final_map50_95', 0.0):.4f}")
        print(f"Drop mAP50  : {report.get('map50_drop_vs_peak', 0.0):.4f}")

        failures = report.get("failures", [])
        if failures:
            print("Fallos:")
            for f in failures:
                print(f" - {f}")

    if status == "GO":
        return 0
    if status == "PENDING":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
