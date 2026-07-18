"""
integrity_check.py — Verifica la integridad de los datasets procesados para YOLO.

Por cada dataset valida:
  - Existencia de train/val con images/ y labels/
  - Correspondencia imagen <-> label  (archivo .txt vacío = imagen sin objetos, válido)
  - Formato YOLO en cada línea: <int> <float> <float> <float> <float> con valores en [0,1]
  - Imágenes sin contenido (0 bytes)

Uso independiente:
  python integrity_check.py
  python integrity_check.py --dataset VisDrone
  python integrity_check.py --strict

También importable desde train_pipeline.py:
  from integrity_check import check_all_datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from config_utils import load_config, path_from_config

# Habilitar secuencias ANSI en la consola de Windows
if sys.platform == "win32":
    os.system("")

_G = "\033[92m"   # verde
_R = "\033[91m"   # rojo
_Y = "\033[93m"   # amarillo
_B = "\033[1m"    # negrita
_X = "\033[0m"    # reset

IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp"}
REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)
MAX_ERR_SHOWN   = 5


# ─── Validación de línea ──────────────────────────────────────────────────────

def _validate_line(line: str) -> str | None:
    """Retorna None si la línea es válida, o una descripción del error."""
    parts = line.strip().split()
    if not parts:
        return None  # línea vacía: permitida
    if len(parts) != 5:
        return f"se esperan 5 campos, hay {len(parts)}: '{line.strip()}'"
    try:
        cid = int(parts[0])
        if cid < 0:
            return f"class_id negativo: {cid}"
        for i, tok in enumerate(parts[1:], 1):
            v = float(tok)
            if not (0.0 <= v <= 1.0):
                return f"campo[{i}]={v:.6f} fuera de [0,1]"
    except ValueError as exc:
        return f"valor no numérico: {exc}"
    return None


# ─── Validación de dataset ────────────────────────────────────────────────────

def check_dataset(ds_path: Path, strict: bool = False) -> tuple[bool, dict]:
    """
    Valida un dataset. Retorna (passed, stats).

    stats contiene:
      splits   : {split_name: image_count}
      classes  : set de class_ids encontrados
      errors   : lista de strings con errores
      warnings : lista de strings con avisos
    """
    errors:   list[str] = []
    warnings: list[str] = []
    split_counts: dict[str, int] = {}
    class_ids: set[int] = set()

    for split in list(REQUIRED_SPLITS) + list(OPTIONAL_SPLITS):
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"
        required = split in REQUIRED_SPLITS

        if not img_dir.exists() or not lbl_dir.exists():
            if required:
                errors.append(f"Directorio requerido ausente: {split}/images o {split}/labels")
            continue

        images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        split_counts[split] = len(images)

        for img in images:
            # Imagen vacía
            if img.stat().st_size == 0:
                errors.append(f"{split}/images/{img.name}: 0 bytes")
                continue

            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                msg = f"{split}: sin label para '{img.name}'"
                (errors if strict else warnings).append(msg)
                continue

            if lbl.stat().st_size == 0:
                continue  # imagen sin objetos, válido

            try:
                text = lbl.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append(f"{split}/labels/{lbl.name}: error de lectura: {exc}")
                continue

            for lineno, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                err = _validate_line(line)
                if err:
                    errors.append(f"{split}/labels/{lbl.name}:{lineno}: {err}")
                else:
                    class_ids.add(int(line.strip().split()[0]))

    # Split requerido sin imágenes
    for split in REQUIRED_SPLITS:
        if split_counts.get(split, 0) == 0 and not any(split in e for e in errors):
            errors.append(f"Split '{split}' no tiene imágenes")

    return len(errors) == 0, {
        "splits":   split_counts,
        "classes":  class_ids,
        "errors":   errors,
        "warnings": warnings,
    }


# ─── Salida de consola ────────────────────────────────────────────────────────

def _print_result(name: str, passed: bool, stats: dict) -> None:
    splits_str  = "  ".join(
        f"{s}:{stats['splits'].get(s, 0)}"
        for s in ("train", "val", "test")
        if s in stats["splits"] or s in REQUIRED_SPLITS
    )
    classes_str = str(sorted(stats["classes"])) if stats["classes"] else "[]"

    if passed:
        print(f"{_G}[OK]  {_X}{name:<24}{splits_str}  clases:{classes_str}")
        for w in stats["warnings"][:2]:
            print(f"      {_Y}→ {w}{_X}")
    else:
        print(f"{_R}[FAIL]{_X}{name:<24}{splits_str}  clases:{classes_str}")
        for e in stats["errors"][:MAX_ERR_SHOWN]:
            print(f"      {_R}→ {e}{_X}")
        extra = len(stats["errors"]) - MAX_ERR_SHOWN
        if extra > 0:
            print(f"      {_R}→ ... y {extra} error(es) más{_X}")


# ─── API pública (importable) ─────────────────────────────────────────────────

def check_all_datasets(
    processed_root: Path,
    dataset_names: list[str] | None = None,
    strict: bool = False,
) -> bool:
    """
    Verifica la integridad de los datasets indicados (o todos los subdirectorios).

    Parámetros:
      processed_root : Path al directorio 00_datasets/processed
      dataset_names  : lista de nombres a verificar; None = todos los subdirectorios
      strict         : si True, falla cuando una imagen no tiene label

    Retorna True si todos los datasets pasan la verificación.
    Llamable desde train_pipeline.py.
    """
    if dataset_names:
        candidates = [processed_root / name for name in dataset_names]
    else:
        candidates = sorted([d for d in processed_root.iterdir() if d.is_dir()])

    if not candidates:
        print(f"{_Y}No se encontraron datasets en {processed_root}{_X}")
        return False

    print(f"\n{_B}=== Verificación de integridad YOLO ==={_X}")
    print(f"Directorio : {processed_root}")
    print(f"Datasets   : {len(candidates)}\n")

    total_ok = total_fail = 0
    for ds_path in candidates:
        if not ds_path.exists():
            print(f"{_R}[FAIL]{_X}{ds_path.name:<24}no encontrado en processed/")
            total_fail += 1
            continue
        passed, stats = check_dataset(ds_path, strict=strict)
        _print_result(ds_path.name, passed, stats)
        if passed:
            total_ok += 1
        else:
            total_fail += 1

    print()
    if total_fail == 0:
        print(f"{_G}{_B}Integridad OK — {total_ok}/{total_ok + total_fail} datasets válidos."
              f" Listo para entrenar.{_X}\n")
        return True
    else:
        print(f"{_R}{_B}ATENCIÓN — {total_fail} dataset(s) con errores."
              f" Corrige antes de entrenar.{_X}\n")
        return False


# ─── Punto de entrada standalone ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica la integridad de los datasets procesados para YOLO."
    )
    parser.add_argument("--dataset", help="Verificar solo este dataset (nombre de carpeta).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla si hay imágenes sin su archivo .txt de labels.",
    )
    args = parser.parse_args()

    cfg = load_config()
    processed_root = path_from_config(cfg, "datasets_processed")

    if not processed_root.exists():
        print(f"{_R}No existe el directorio de procesados: {processed_root}{_X}")
        return 1

    names = [args.dataset] if args.dataset else None
    return 0 if check_all_datasets(processed_root, dataset_names=names, strict=args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
