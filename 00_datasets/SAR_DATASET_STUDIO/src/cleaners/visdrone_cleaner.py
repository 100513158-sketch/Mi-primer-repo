from __future__ import annotations

import json
import shutil
from pathlib import Path


class VisDroneCleaner:
    """
    Limpieza conservadora del dataset VisDrone.

    IMPORTANTE:
        - Nunca modifica raw/.
        - Las imágenes se copian a processed/cleaned/.
        - Las anotaciones mantienen inicialmente el formato VisDrone.
        - Las decisiones se registran en manifest.jsonl.

    Estados posibles:

        KEEP
        IGNORE
        QUARANTINE
    """

    # =========================================================
    # CLASES OFICIALES VISDRONE
    # =========================================================

    CLASSES = {
        0: "ignored_region",
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor",
        11: "others",
    }

    # Clases reales que pueden entrar al entrenamiento.
    VALID_CLASSES = set(range(1, 12))

    # Clase 0 NO es una clase de entrenamiento.
    IGNORED_CLASS = 0

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(self):

        # -----------------------------------------------------
        # SAR_DATASET_STUDIO
        #
        # __file__:
        #
        # SAR_DATASET_STUDIO/
        #   src/
        #     cleaners/
        #       visdrone_cleaner.py
        #
        # parents[2] = SAR_DATASET_STUDIO
        # -----------------------------------------------------

        self.studio_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        self.raw_root = (
            self.studio_root
            / "raw"
            / "VisDrone"
            / "original"
        )

        self.cleaned_root = (
            self.studio_root
            / "processed"
            / "cleaned"
            / "VisDrone"
        )

        self.report_root = (
            self.studio_root
            / "reports"
            / "quality"
            / "VisDrone"
        )

        self.quarantine_root = (
            self.cleaned_root
            / "quarantine"
        )

        self.report_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cleaned_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.quarantine_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # SPLITS
    # =========================================================

    def get_splits(self):

        return {
            "train": (
                self.raw_root
                / "train"
                / "VisDrone2019-DET-train"
            ),

            "val": (
                self.raw_root
                / "val"
                / "VisDrone2019-DET-val"
            ),

            "test_dev": (
                self.raw_root
                / "test_dev"
            ),
        }

    # =========================================================
    # LEER ANNOTATION
    # =========================================================

    def parse_annotation(
        self,
        line: str,
        line_number: int,
    ):

        clean_line = line.strip().rstrip(",")

        if not clean_line:
            return None

        parts = [
            p.strip()
            for p in clean_line.split(",")
        ]

        # VisDrone debe tener 8 campos.
        if len(parts) != 8:

            return {
                "status": "QUARANTINE",
                "reason": "invalid_format",
                "line": line_number,
                "raw": line,
            }

        try:

            x = int(parts[0])
            y = int(parts[1])
            width = int(parts[2])
            height = int(parts[3])
            score = int(parts[4])
            class_id = int(parts[5])
            truncation = int(parts[6])
            occlusion = int(parts[7])

        except ValueError:

            return {
                "status": "QUARANTINE",
                "reason": "non_numeric_value",
                "line": line_number,
                "raw": line,
            }

        # -----------------------------------------------------
        # REGIÓN IGNORADA
        # -----------------------------------------------------

        if class_id == self.IGNORED_CLASS:

            return {
                "status": "IGNORE",
                "reason": "ignored_region",
                "line": line_number,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "score": score,
                "class_id": class_id,
                "class_name": self.CLASSES[class_id],
                "truncation": truncation,
                "occlusion": occlusion,
                "raw": line,
            }

        # -----------------------------------------------------
        # CLASE DESCONOCIDA
        # -----------------------------------------------------

        if class_id not in self.VALID_CLASSES:

            return {
                "status": "QUARANTINE",
                "reason": "invalid_class",
                "line": line_number,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "score": score,
                "class_id": class_id,
                "class_name": f"class_{class_id}",
                "truncation": truncation,
                "occlusion": occlusion,
                "raw": line,
            }

        # -----------------------------------------------------
        # GEOMETRÍA INVALIDA
        # -----------------------------------------------------

        if width <= 0:

            return {
                "status": "QUARANTINE",
                "reason": "width<=0",
                "line": line_number,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "score": score,
                "class_id": class_id,
                "class_name": self.CLASSES[class_id],
                "truncation": truncation,
                "occlusion": occlusion,
                "raw": line,
            }

        if height <= 0:

            return {
                "status": "QUARANTINE",
                "reason": "height<=0",
                "line": line_number,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "score": score,
                "class_id": class_id,
                "class_name": self.CLASSES[class_id],
                "truncation": truncation,
                "occlusion": occlusion,
                "raw": line,
            }

        # -----------------------------------------------------
        # BOX VÁLIDA
        # -----------------------------------------------------

        return {
            "status": "KEEP",
            "reason": "valid",
            "line": line_number,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "score": score,
            "class_id": class_id,
            "class_name": self.CLASSES[class_id],
            "truncation": truncation,
            "occlusion": occlusion,
            "raw": line,
        }

    # =========================================================
    # COMPROBAR BOUNDING BOX CONTRA IMAGEN
    # =========================================================

    def validate_coordinates(
        self,
        annotation: dict,
        image_width: int,
        image_height: int,
    ):

        x = annotation["x"]
        y = annotation["y"]
        width = annotation["width"]
        height = annotation["height"]

        x2 = x + width
        y2 = y + height

        # Coordenadas negativas.

        if x < 0:
            return "x<0"

        if y < 0:
            return "y<0"

        # Punto inicial fuera.

        if x >= image_width:
            return "x>=image_width"

        if y >= image_height:
            return "y>=image_height"

        # Bounding box completamente fuera.

        if x2 <= 0:
            return "box_outside_left"

        if y2 <= 0:
            return "box_outside_top"

        # Parte derecha fuera.

        if x2 > image_width:
            return "x+width>image_width"

        # Parte inferior fuera.

        if y2 > image_height:
            return "y+height>image_height"

        return None

    # =========================================================
    # LEER TODA UNA ANNOTATION
    # =========================================================

    def process_annotation_file(
        self,
        annotation_file: Path,
        image_file: Path,
        image_width: int,
        image_height: int,
    ):

        kept = []
        ignored = []
        quarantined = []

        lines = annotation_file.read_text(
            encoding="utf-8"
        ).splitlines()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            parsed = self.parse_annotation(
                line,
                line_number,
            )

            if parsed is None:
                continue

            status = parsed["status"]

            # -------------------------------------------------
            # IGNORE
            # -------------------------------------------------

            if status == "IGNORE":

                ignored.append(parsed)

                continue

            # -------------------------------------------------
            # QUARANTINE
            # -------------------------------------------------

            if status == "QUARANTINE":

                quarantined.append(parsed)

                continue

            # -------------------------------------------------
            # VALIDAR COORDENADAS
            # -------------------------------------------------

            coordinate_error = (
                self.validate_coordinates(
                    parsed,
                    image_width,
                    image_height,
                )
            )

            if coordinate_error:

                parsed["status"] = "QUARANTINE"

                parsed["reason"] = (
                    coordinate_error
                )

                quarantined.append(parsed)

                continue

            # -------------------------------------------------
            # KEEP
            # -------------------------------------------------

            kept.append(parsed)

        return (
            kept,
            ignored,
            quarantined,
        )

    # =========================================================
    # ESCRIBIR ANNOTATION LIMPIA
    # =========================================================

    def write_clean_annotation(
        self,
        output_file: Path,
        kept_annotations: list[dict],
    ):

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        lines = []

        for annotation in kept_annotations:

            line = (
                f"{annotation['x']},"
                f"{annotation['y']},"
                f"{annotation['width']},"
                f"{annotation['height']},"
                f"{annotation['score']},"
                f"{annotation['class_id']},"
                f"{annotation['truncation']},"
                f"{annotation['occlusion']}"
            )

            lines.append(line)

        output_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    # =========================================================
    # REGISTRAR QUARANTINE
    # =========================================================

    def write_quarantine(
        self,
        split: str,
        annotation_file: Path,
        image_file: Path,
        quarantined: list[dict],
    ):

        if not quarantined:
            return

        quarantine_split = (
            self.quarantine_root
            / split
        )

        quarantine_split.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_file = (
            quarantine_split
            / f"{annotation_file.stem}.json"
        )

        data = {
            "dataset": "VisDrone2019-DET",
            "split": split,
            "image": str(image_file),
            "annotation": str(annotation_file),
            "objects": quarantined,
        }

        report_file.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # =========================================================
    # COPIAR IMAGEN
    # =========================================================

    def copy_image(
        self,
        source: Path,
        destination: Path,
    ):

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not destination.exists():

            shutil.copy2(
                source,
                destination,
            )

    # =========================================================
    # MANIFEST
    # =========================================================

    def write_manifest(
        self,
        split: str,
        records: list[dict],
    ):

        manifest_dir = (
            self.studio_root
            / "registry"
            / "manifests"
        )

        manifest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_file = (
            manifest_dir
            / f"visdrone_{split}_cleaning.jsonl"
        )

        with manifest_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            for record in records:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return manifest_file

    # =========================================================
    # PROCESAR SPLIT
    # =========================================================

    def process_split(
        self,
        split: str,
        source_root: Path,
    ):

        print()
        print("=" * 70)

        print(
            f"LIMPIANDO SPLIT: {split}"
        )

        print(
            f"Origen: {source_root}"
        )

        # -----------------------------------------------------
        # Directorios originales
        # -----------------------------------------------------

        source_images = (
            source_root
            / "images"
        )

        source_annotations = (
            source_root
            / "annotations"
        )

        if not source_images.exists():

            print(
                "[ERROR] No existe:"
            )

            print(
                source_images
            )

            return None

        if not source_annotations.exists():

            print(
                "[ERROR] No existe:"
            )

            print(
                source_annotations
            )

            return None

        # -----------------------------------------------------
        # Destino
        # -----------------------------------------------------

        destination_root = (
            self.cleaned_root
            / split
        )

        destination_images = (
            destination_root
            / "images"
        )

        destination_annotations = (
            destination_root
            / "annotations"
        )

        destination_images.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_annotations.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Buscar imágenes
        # -----------------------------------------------------

        image_files = sorted(
            source_images.rglob("*.jpg")
        )

        print(
            f"Imágenes encontradas: "
            f"{len(image_files)}"
        )

        records = []

        statistics = {
            "images": 0,
            "annotations": 0,
            "kept": 0,
            "ignored": 0,
            "quarantine": 0,
            "empty_clean_annotations": 0,
        }

        # -----------------------------------------------------
        # Procesar
        # -----------------------------------------------------

        for index, image_file in enumerate(
            image_files,
            start=1,
        ):

            statistics["images"] += 1

            # ---------------------------------------------
            # Buscar annotation correspondiente
            # ---------------------------------------------

            relative_image = (
                image_file.relative_to(
                    source_images
                )
            )

            annotation_file = (
                source_annotations
                / relative_image.with_suffix(
                    ".txt"
                )
            )

            # ---------------------------------------------
            # Imagen sin annotation
            # ---------------------------------------------

            if not annotation_file.exists():

                record = {
                    "dataset": "VisDrone2019-DET",
                    "split": split,
                    "image": str(
                        image_file.relative_to(
                            self.raw_root
                        )
                    ),
                    "annotation": None,
                    "status": "QUARANTINE",
                    "reason": "annotation_missing",
                }

                records.append(record)

                # No eliminamos la imagen del RAW.
                continue

            # ---------------------------------------------
            # Abrir imagen
            # ---------------------------------------------

            try:

                from PIL import Image

                with Image.open(
                    image_file
                ) as img:

                    image_width = img.width
                    image_height = img.height

            except Exception as exc:

                record = {
                    "dataset": "VisDrone2019-DET",
                    "split": split,
                    "image": str(image_file),
                    "annotation": str(annotation_file),
                    "status": "QUARANTINE",
                    "reason": "image_read_error",
                    "error": str(exc),
                }

                records.append(record)

                continue

            # ---------------------------------------------
            # Procesar annotation
            # ---------------------------------------------

            (
                kept,
                ignored,
                quarantined,
            ) = self.process_annotation_file(
                annotation_file,
                image_file,
                image_width,
                image_height,
            )

            statistics["annotations"] += (
                len(kept)
                + len(ignored)
                + len(quarantined)
            )

            statistics["kept"] += len(kept)

            statistics["ignored"] += len(
                ignored
            )

            statistics["quarantine"] += len(
                quarantined
            )

            if not kept:

                statistics[
                    "empty_clean_annotations"
                ] += 1

            # ---------------------------------------------
            # Destinos
            # ---------------------------------------------

            destination_image = (
                destination_images
                / relative_image
            )

            destination_annotation = (
                destination_annotations
                / relative_image.with_suffix(
                    ".txt"
                )
            )

            # ---------------------------------------------
            # Copiar imagen
            # ---------------------------------------------

            self.copy_image(
                image_file,
                destination_image,
            )

            # ---------------------------------------------
            # Escribir annotation limpia
            # ---------------------------------------------

            self.write_clean_annotation(
                destination_annotation,
                kept,
            )

            # ---------------------------------------------
            # Quarantine
            # ---------------------------------------------

            self.write_quarantine(
                split,
                annotation_file,
                image_file,
                quarantined,
            )

            # ---------------------------------------------
            # Registro
            # ---------------------------------------------

            record = {
                "dataset": "VisDrone2019-DET",
                "split": split,
                "image": str(
                    image_file.relative_to(
                        self.raw_root
                    )
                ),
                "annotation": str(
                    annotation_file.relative_to(
                        self.raw_root
                    )
                ),
                "image_width": image_width,
                "image_height": image_height,
                "objects": {
                    "keep": len(kept),
                    "ignore": len(ignored),
                    "quarantine": len(
                        quarantined
                    ),
                },
                "quarantine_reasons": [
                    item["reason"]
                    for item in quarantined
                ],
                "status": (
                    "KEEP"
                    if kept
                    else (
                        "IGNORE"
                        if ignored
                        and not quarantined
                        else "QUARANTINE"
                    )
                ),
            }

            records.append(record)

            # ---------------------------------------------
            # Progreso
            # ---------------------------------------------

            if (
                index % 500 == 0
                or index == len(image_files)
            ):

                print(
                    f"Procesadas: "
                    f"{index}/{len(image_files)}"
                )

        # -----------------------------------------------------
        # Manifest
        # -----------------------------------------------------

        manifest = self.write_manifest(
            split,
            records,
        )

        # -----------------------------------------------------
        # Informe
        # -----------------------------------------------------

        report = {
            "dataset": "VisDrone2019-DET",
            "split": split,
            "source": str(source_root),
            "destination": str(
                destination_root
            ),
            "statistics": statistics,
            "manifest": str(manifest),
        }

        report_file = (
            self.report_root
            / f"{split}_cleaning_report.json"
        )

        report_file.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("RESULTADO")
        print("-" * 70)

        print(
            f"Imágenes:       "
            f"{statistics['images']}"
        )

        print(
            f"Annotations:    "
            f"{statistics['annotations']}"
        )

        print(
            f"KEEP:           "
            f"{statistics['kept']}"
        )

        print(
            f"IGNORE:         "
            f"{statistics['ignored']}"
        )

        print(
            f"QUARANTINE:     "
            f"{statistics['quarantine']}"
        )

        print(
            f"Sin objetos KEEP:"
            f" {statistics['empty_clean_annotations']}"
        )

        print()
        print(
            f"Informe: {report_file}"
        )

        print(
            f"Manifest: {manifest}"
        )

        return report

    # =========================================================
    # EJECUTAR TODO
    # =========================================================

    def run(self):

        print()
        print("=" * 70)
        print("VISDRONE CLEANER")
        print("=" * 70)

        print()
        print(
            f"RAW:     {self.raw_root}"
        )

        print(
            f"CLEANED: {self.cleaned_root}"
        )

        print()
        print(
            "IMPORTANTE:"
        )

        print(
            "El directorio RAW no será modificado."
        )

        all_reports = {}

        for split, source_root in (
            self.get_splits().items()
        ):

            if not source_root.exists():

                print()
                print(
                    f"[WARNING] No existe split: "
                    f"{split}"
                )

                continue

            result = self.process_split(
                split,
                source_root,
            )

            all_reports[split] = result

        # -----------------------------------------------------
        # Informe global
        # -----------------------------------------------------

        global_report = (
            self.report_root
            / "global_cleaning_report.json"
        )

        global_report.write_text(
            json.dumps(
                all_reports,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 70)
        print("LIMPIEZA TERMINADA")
        print("=" * 70)

        print()
        print(
            f"Informe global:"
        )

        print(
            global_report
        )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    cleaner = VisDroneCleaner()

    cleaner.run()