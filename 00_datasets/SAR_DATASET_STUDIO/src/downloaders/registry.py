from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetRegistry:
    """
    Registro local de datasets instalados.

    Guarda información sobre:
        - dataset
        - fuente
        - ubicación
        - estado
        - fecha de registro
    """

    def __init__(
        self,
        registry_path: Path,
    ):

        self.registry_path = Path(
            registry_path
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data = self._load()

    # =========================================================
    # LOAD
    # =========================================================

    def _load(self) -> dict[str, Any]:

        if not self.registry_path.exists():

            return {
                "datasets": {}
            }

        try:

            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError,
        ):

            print(
                "[WARNING] Registry vacío o inválido."
            )

            data = {}

        # -----------------------------------------------------
        # Garantizar estructura
        # -----------------------------------------------------

        if not isinstance(data, dict):

            data = {}

        if "datasets" not in data:

            data["datasets"] = {}

        if not isinstance(
            data["datasets"],
            dict,
        ):

            data["datasets"] = {}

        return data

    # =========================================================
    # SAVE
    # =========================================================

    def _save(self) -> None:

        temporary_path = (
            self.registry_path.with_suffix(
                ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        temporary_path.replace(
            self.registry_path
        )

    # =========================================================
    # REGISTER
    # =========================================================

    def register(
        self,
        dataset_id: str,
        information: dict[str, Any],
    ) -> None:

        if not dataset_id:
            raise ValueError(
                "dataset_id no puede estar vacío."
            )

        if not isinstance(
            information,
            dict,
        ):
            raise TypeError(
                "information debe ser un diccionario."
            )

        self.data["datasets"][
            dataset_id
        ] = information

        self._save()

        print()
        print(
            f"[OK] Dataset registrado: "
            f"{dataset_id}"
        )

        print(
            f"[OK] Registry: "
            f"{self.registry_path}"
        )

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        dataset_id: str,
    ) -> dict[str, Any] | None:

        return self.data[
            "datasets"
        ].get(
            dataset_id
        )

    # =========================================================
    # EXISTS
    # =========================================================

    def exists(
        self,
        dataset_id: str,
    ) -> bool:

        return (
            dataset_id
            in self.data["datasets"]
        )

    # =========================================================
    # LIST
    # =========================================================

    def list(
        self,
    ) -> list[str]:

        return list(
            self.data["datasets"].keys()
        )

    # =========================================================
    # REMOVE
    # =========================================================

    def remove(
        self,
        dataset_id: str,
    ) -> bool:

        if not self.exists(
            dataset_id
        ):

            return False

        del self.data[
            "datasets"
        ][dataset_id]

        self._save()

        return True