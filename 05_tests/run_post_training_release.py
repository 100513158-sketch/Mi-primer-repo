"""
Orquestador post-entrenamiento para SARC-Drone.

Secuencia automatizada:
1) Gate model_candidate
2) Exportacion de modelos
3) Optimizacion extrema (opcional)
4) Gate release_edge_android
5) Build Android (opcional)

Uso:
  python 05_tests/run_post_training_release.py
  python 05_tests/run_post_training_release.py --skip-build
  python 05_tests/run_post_training_release.py --no-extreme
  python 05_tests/run_post_training_release.py --release-profile release_edge_android

Codigos de salida:
  0  -> Flujo completo OK
  10 -> Gate model_candidate en PENDING
  11 -> Gate model_candidate en NO-GO
  20 -> Fallo en exportacion u optimizacion
  30 -> Gate release en PENDING
  31 -> Gate release en NO-GO
  40 -> Fallo en build Android
  99 -> Error interno
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "05_tests" / "reports"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("\n[RUN]", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def parse_gate_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("Salida vacia del gate")
    return json.loads(text)


def save_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"post_training_release_{ts}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Orquestador post-entrenamiento SARC-Drone")
    parser.add_argument("--python-exe", default=sys.executable, help="Ejecutable de Python")
    parser.add_argument("--candidate-profile", default="model_candidate")
    parser.add_argument("--release-profile", default="release_edge_android")
    parser.add_argument("--skip-build", action="store_true", help="Omite build Android")
    parser.add_argument("--no-extreme", action="store_true", help="Omite extreme_optimization.py")
    parser.add_argument(
        "--build-mode",
        choices=["debug", "release"],
        default="release",
        help="Modo de build Android",
    )
    args = parser.parse_args()

    py = args.python_exe
    gate_script = PROJECT_ROOT / "05_tests" / "go_nogo_gate.py"
    export_script = PROJECT_ROOT / "01_training" / "scripts" / "export_model.py"
    extreme_script = PROJECT_ROOT / "01_training" / "scripts" / "extreme_optimization.py"
    build_script = PROJECT_ROOT / "03_android_app" / "build_all.ps1"

    report: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_profile": args.candidate_profile,
        "release_profile": args.release_profile,
        "steps": [],
    }

    try:
        # 1) Gate model_candidate
        step = {"name": "gate_candidate"}
        cp = run_cmd([py, str(gate_script), "--profile", args.candidate_profile, "--json"], cwd=PROJECT_ROOT)
        step["returncode"] = cp.returncode
        step["stdout"] = cp.stdout.strip()
        step["stderr"] = cp.stderr.strip()
        report["steps"].append(step)

        gate_candidate = parse_gate_json(cp.stdout)
        report["gate_candidate"] = gate_candidate

        if gate_candidate.get("status") == "PENDING":
            report["status"] = "PENDING"
            out = save_report(report)
            print(f"\nResultado: PENDING (candidate). Reporte: {out}")
            return 10
        if gate_candidate.get("status") != "GO":
            report["status"] = "NO-GO"
            out = save_report(report)
            print(f"\nResultado: NO-GO (candidate). Reporte: {out}")
            return 11

        # 2) Exportacion
        step = {"name": "export_model"}
        cp = run_cmd([py, str(export_script)], cwd=PROJECT_ROOT / "01_training" / "scripts")
        step["returncode"] = cp.returncode
        step["stdout_tail"] = "\n".join(cp.stdout.splitlines()[-30:])
        step["stderr_tail"] = "\n".join(cp.stderr.splitlines()[-30:])
        report["steps"].append(step)

        if cp.returncode != 0:
            report["status"] = "FAILED_EXPORT"
            out = save_report(report)
            print(f"\nResultado: fallo export_model.py. Reporte: {out}")
            return 20

        # 3) Optimizacion extrema opcional
        if not args.no_extreme:
            step = {"name": "extreme_optimization"}
            cp = run_cmd([py, str(extreme_script)], cwd=PROJECT_ROOT / "01_training" / "scripts")
            step["returncode"] = cp.returncode
            step["stdout_tail"] = "\n".join(cp.stdout.splitlines()[-30:])
            step["stderr_tail"] = "\n".join(cp.stderr.splitlines()[-30:])
            report["steps"].append(step)

            if cp.returncode != 0:
                tflite_artifact = PROJECT_ROOT / "02_models" / "exported" / "tflite" / "best_C2A" / "best_C2A.tflite"
                onnx_artifact = PROJECT_ROOT / "02_models" / "weights" / "best_C2A.onnx"
                if tflite_artifact.exists() or onnx_artifact.exists():
                    report["status"] = "WARN_EXTREME_OPT"
                    report["warning"] = "extreme_optimization.py fallo, pero los artefactos base ya existen"
                    report["artifacts"] = {
                        "tflite": str(tflite_artifact),
                        "onnx": str(onnx_artifact),
                    }
                    print(f"\nAdvertencia: extreme_optimization.py fallo, pero se encontraron artefactos exportados. Continuando.")
                else:
                    report["status"] = "FAILED_EXTREME_OPT"
                    out = save_report(report)
                    print(f"\nResultado: fallo extreme_optimization.py. Reporte: {out}")
                    return 20

        # 4) Gate release
        step = {"name": "gate_release"}
        cp = run_cmd([py, str(gate_script), "--profile", args.release_profile, "--json"], cwd=PROJECT_ROOT)
        step["returncode"] = cp.returncode
        step["stdout"] = cp.stdout.strip()
        step["stderr"] = cp.stderr.strip()
        report["steps"].append(step)

        gate_release = parse_gate_json(cp.stdout)
        report["gate_release"] = gate_release

        if gate_release.get("status") == "PENDING":
            report["status"] = "PENDING"
            out = save_report(report)
            print(f"\nResultado: PENDING (release). Reporte: {out}")
            return 30
        if gate_release.get("status") != "GO":
            report["status"] = "NO-GO"
            out = save_report(report)
            print(f"\nResultado: NO-GO (release). Reporte: {out}")
            return 31

        # 5) Build Android opcional
        if not args.skip_build:
            step = {"name": "android_build"}
            build_cmd = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(build_script),
            ]
            if args.build_mode == "release":
                build_cmd.append("-Release")

            cp = run_cmd(build_cmd, cwd=PROJECT_ROOT)
            step["returncode"] = cp.returncode
            step["stdout_tail"] = "\n".join(cp.stdout.splitlines()[-40:])
            step["stderr_tail"] = "\n".join(cp.stderr.splitlines()[-40:])
            report["steps"].append(step)

            if cp.returncode != 0:
                report["status"] = "FAILED_BUILD"
                out = save_report(report)
                print(f"\nResultado: fallo build Android. Reporte: {out}")
                return 40

        report["status"] = "GO"
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        out = save_report(report)
        print(f"\nResultado final: GO. Reporte: {out}")
        return 0

    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = str(exc)
        out = save_report(report)
        print(f"\nError interno: {exc}. Reporte: {out}")
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
