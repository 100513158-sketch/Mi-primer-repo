from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import gdown

from .base_downloader import BaseDownloader


class VisDroneDownloader(BaseDownloader):
    """
    Downloader de VisDrone2019-DET.

    Descarga:
        - train
        - val
        - test-dev

    Conserva los ZIP originales y los archivos extraídos.
    """

    def __init__(
        self,
        dataset_id: str,
        config: dict[str, Any],
        destination: Path,
    ):
        super().__init__(
            dataset_id=dataset_id,
            config=config,
            destination=destination,
        )

        self.archives_dir = self.destination / "archives"
        self.original_dir = self.destination / "original"

        self.archives_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.original_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # DESCARGA
    # =========================================================

    def download(self) -> None:

        downloads = self.config.get(
            "downloads",
            {}
        )

        for split, information in downloads.items():

            if not information.get(
                "enabled",
                False
            ):
                continue

            filename = information["filename"]
            file_id = information["google_drive_id"]

            archive_path = (
                self.archives_dir / filename
            )

            print()
            print("-" * 70)
            print(f"Split:    {split}")
            print(f"Archivo:  {filename}")
            print(f"Google ID: {file_id}")
            print("-" * 70)

            # -------------------------------------------------
            # Si ya existe, no volvemos a descargarlo
            # -------------------------------------------------

            if archive_path.exists():

                print(
                    "[OK] El archivo ya existe."
                )

                print(
                    f"     {archive_path}"
                )

                continue

            print(
                "[INFO] Descargando desde Google Drive..."
            )

            url = (
                f"https://drive.google.com/uc?id={file_id}"
            )

            temporary_path = (
                archive_path.with_suffix(
                    archive_path.suffix + ".download"
                )
            )

            if temporary_path.exists():

                temporary_path.unlink()

            # -------------------------------------------------
            # IMPORTANTE:
            #
            # No utilizamos fuzzy porque algunas versiones
            # de gdown no soportan ese argumento.
            # -------------------------------------------------

            result = gdown.download(
                url=url,
                output=str(
                    temporary_path
                ),
                quiet=False,
            )

            if result is None:

                raise RuntimeError(
                    f"Falló la descarga de {filename}"
                )

            if not temporary_path.exists():

                raise RuntimeError(
                    "gdown terminó pero no se encontró "
                    f"el archivo: {temporary_path}"
                )

            # -------------------------------------------------
            # Renombrado atómico
            # -------------------------------------------------

            temporary_path.replace(
                archive_path
            )

            print(
                "[OK] Descarga completada."
            )

            print(
                f"     {archive_path}"
            )

    # =========================================================
    # SHA-256
    # =========================================================

    @staticmethod
    def calculate_sha256(
        file_path: Path
    ) -> str:

        sha256 = hashlib.sha256()

        with file_path.open(
            "rb"
        ) as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                sha256.update(
                    chunk
                )

        return sha256.hexdigest()

    # =========================================================
    # VERIFICACIÓN
    # =========================================================

    def verify(self) -> bool:

        downloads = self.config.get(
            "downloads",
            {}
        )

        all_valid = True

        for split, information in downloads.items():

            if not information.get(
                "enabled",
                False
            ):
                continue

            filename = information["filename"]

            archive_path = (
                self.archives_dir / filename
            )

            # -------------------------------------------------
            # Existe
            # -------------------------------------------------

            if not archive_path.exists():

                print(
                    f"[ERROR] Falta: {filename}"
                )

                all_valid = False

                continue

            # -------------------------------------------------
            # ZIP válido
            # -------------------------------------------------

            try:

                with zipfile.ZipFile(
                    archive_path,
                    "r"
                ) as zip_file:

                    bad_file = (
                        zip_file.testzip()
                    )

                    if bad_file is not None:

                        print(
                            f"[ERROR] ZIP corrupto: "
                            f"{filename}"
                        )

                        print(
                            f"        Archivo: "
                            f"{bad_file}"
                        )

                        all_valid = False

                        continue

            except zipfile.BadZipFile:

                print(
                    f"[ERROR] No es un ZIP válido: "
                    f"{filename}"
                )

                all_valid = False

                continue

            # -------------------------------------------------
            # SHA-256
            # -------------------------------------------------

            sha256 = (
                self.calculate_sha256(
                    archive_path
                )
            )

            print(
                f"[OK] {filename}"
            )

            print(
                f"     SHA-256: {sha256}"
            )

        return all_valid

    # =========================================================
    # EXTRACCIÓN
    # =========================================================

    def extract(self) -> None:

        downloads = self.config.get(
            "downloads",
            {}
        )

        for split, information in downloads.items():

            if not information.get(
                "enabled",
                False
            ):
                continue

            filename = information["filename"]

            archive_path = (
                self.archives_dir / filename
            )

            if not archive_path.exists():

                raise FileNotFoundError(
                    archive_path
                )

            split_directory = (
                self.original_dir / split
            )

            # -------------------------------------------------
            # Ya existe contenido
            # -------------------------------------------------

            if (
                split_directory.exists()
                and any(
                    split_directory.iterdir()
                )
            ):

                print(
                    f"[OK] {split} ya está extraído."
                )

                continue

            split_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            print(
                f"[INFO] Extrayendo {filename}..."
            )

            with zipfile.ZipFile(
                archive_path,
                "r"
            ) as zip_file:

                self._safe_extract(
                    zip_file,
                    split_directory
                )

            print(
                f"[OK] {split} extraído."
            )

    # =========================================================
    # EXTRACCIÓN SEGURA
    # =========================================================

    @staticmethod
    def _safe_extract(
        zip_file: zipfile.ZipFile,
        destination: Path,
    ) -> None:

        destination = (
            destination.resolve()
        )

        for member in zip_file.infolist():

            member_path = (
                destination
                / member.filename
            ).resolve()

            if not str(
                member_path
            ).startswith(
                str(destination)
            ):

                raise RuntimeError(
                    "ZIP potencialmente inseguro: "
                    f"{member.filename}"
                )

        zip_file.extractall(
            destination
        )

    # =========================================================
    # VALIDACIÓN DE ESTRUCTURA
    # =========================================================

    def validate_structure(self) -> bool:

        valid = True

        expected_splits = [
            "train",
            "val",
            "test_dev",
        ]

        for split in expected_splits:

            split_root = (
                self.original_dir
                / split
            )

            # -------------------------------------------------
            # Existe
            # -------------------------------------------------

            if not split_root.exists():

                print(
                    f"[ERROR] No existe: "
                    f"{split_root}"
                )

                valid = False

                continue

            # -------------------------------------------------
            # Tiene contenido
            # -------------------------------------------------

            entries = list(
                split_root.iterdir()
            )

            if not entries:

                print(
                    f"[ERROR] {split} está vacío."
                )

                valid = False

                continue

            print(
                f"[OK] Split {split}"
            )

            print(
                f"     Elementos: "
                f"{len(entries)}"
            )

        return valid

    # =========================================================
    # PIPELINE
    # =========================================================

    def run(self) -> bool:

        print()
        print("=" * 70)
        print(
            "SAR DATASET STUDIO"
        )
        print(
            "VisDrone2019-DET Downloader"
        )
        print("=" * 70)

        # -----------------------------------------------------
        # 1
        # -----------------------------------------------------

        print()
        print(
            "[1/4] Descargando..."
        )

        self.download()

        # -----------------------------------------------------
        # 2
        # -----------------------------------------------------

        print()
        print(
            "[2/4] Verificando archivos..."
        )

        if not self.verify():

            print()
            print(
                "[ERROR] La verificación falló."
            )

            return False

        # -----------------------------------------------------
        # 3
        # -----------------------------------------------------

        print()
        print(
            "[3/4] Extrayendo..."
        )

        self.extract()

        # -----------------------------------------------------
        # 4
        # -----------------------------------------------------

        print()
        print(
            "[4/4] Validando estructura..."
        )

        if not self.validate_structure():

            print()
            print(
                "[ERROR] La validación de estructura falló."
            )

            return False

        # -----------------------------------------------------
        # OK
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "VisDrone descargado correctamente."
        )
        print("=" * 70)

        return True