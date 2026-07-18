"""
generate_dataset_yamls.py — Genera un data.yaml individual por dataset procesado.

Los nombres de clase se obtienen de training_pipeline.class_names en config.yaml.
Si un dataset no está definido en config, las clases se infieren desde las etiquetas
y se nombran class_0, class_1, ...

Uso independiente:
  python generate_dataset_yamls.py
  python generate_dataset_yamls.py --dataset SeaDronesSee
  python generate_dataset_yamls.py --overwrite

También importable desde train_pipeline.py:
  from generate_dataset_yamls import ensure_dataset_yamls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config_utils import load_config, path_from_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _infer_classes(ds_path: Path) -> dict[int, str]:
    """Escanea los labels y retorna {class_id: 'class_<id>'} para todos los IDs hallados."""
    found: set[int] = set()
    for split in ("train", "val", "test"):
        lbl_dir = ds_path / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            try:
                for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            found.add(int(parts[0]))
                        except ValueError:
                            pass
            except OSError:
                pass
    return {cid: f"class_{cid}" for cid in sorted(found)}


def _count_images(ds_path: Path) -> dict[str, int]:
    """Devuelve {split: imagen_count} solo para splits que tienen imágenes."""
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        img_dir = ds_path / split / "images"
        if img_dir.exists():
            n = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if n > 0:
                counts[split] = n
    return counts


def _write_yaml(yaml_path: Path, ds_path: Path, class_names: dict[int, str], counts: dict[str, int]) -> None:
    """Escribe el data.yaml con formato limpio y compatible con Ultralytics."""
    # Usar barras diagonales (compatible Windows y Linux)
    posix_path = str(ds_path).replace("\\", "/")

    lines = [
        f"path: {posix_path}",
        "train: train/images",
        "val:   val/images",
    ]
    if "test" in counts:
        lines.append("test:  test/images")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for cid, cname in sorted(class_names.items()):
        lines.append(f"  {cid}: {cname}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── API pública (importable) ─────────────────────────────────────────────────

def ensure_dataset_yamls(
    cfg: dict,
    dataset_names: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """
    Genera data.yaml para los datasets que no lo tienen (o todos si overwrite=True).

    Parámetros:
      cfg           : config cargado desde config.yaml
      dataset_names : lista de datasets a procesar; None = todos los subdirectorios
      overwrite     : si True, sobreescribe los data.yaml existentes

    Llamable desde train_pipeline.py.
    """
    processed_root = path_from_config(cfg, "datasets_processed")
    config_names: dict = cfg.get("training_pipeline", {}).get("class_names", {})

    if dataset_names:
        datasets = [processed_root / name for name in dataset_names if (processed_root / name).exists()]
    else:
        datasets = sorted([d for d in processed_root.iterdir() if d.is_dir()])

    for ds_path in datasets:
        yaml_path = ds_path / "data.yaml"

        if yaml_path.exists() and not overwrite:
            continue  # ya existe y no se pide sobreescribir

        ds_name = ds_path.name

        # Obtener nombres de clase: desde config primero, sino inferir
        raw = config_names.get(ds_name)
        if raw:
            class_names = {int(k): str(v) for k, v in raw.items()}
        else:
            class_names = _infer_classes(ds_path)
            if not class_names:
                print(f"  [WARN]  {ds_name}: sin clases detectadas, se omite")
                continue
            print(f"  [INFO]  {ds_name}: clases inferidas automáticamente desde labels")

        counts = _count_images(ds_path)
        _write_yaml(yaml_path, ds_path, class_names, counts)

        nc         = len(class_names)
        names_str  = ", ".join(f"{k}:{v}" for k, v in sorted(class_names.items()))
        splits_str = "  ".join(f"{s}:{counts.get(s, 0)}" for s in ("train", "val", "test") if s in counts)
        action     = "actualizado" if overwrite else "generado"
        print(f"  [OK]    {ds_name}/data.yaml  nc={nc}  ({names_str})  [{splits_str}]  [{action}]")


# ─── Punto de entrada standalone ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera un data.yaml individual por dataset procesado."
    )
    parser.add_argument("--dataset", help="Generar solo para este dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir data.yaml existentes.")
    args = parser.parse_args()

    cfg = load_config()
    processed_root = path_from_config(cfg, "datasets_processed")

    if not processed_root.exists():
        print(f"ERROR: No existe {processed_root}")
        return 1

    print(f"\n=== Generando data.yaml por dataset ===")
    print(f"Directorio: {processed_root}\n")

    names = [args.dataset] if args.dataset else None
    ensure_dataset_yamls(cfg, dataset_names=names, overwrite=args.overwrite)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
