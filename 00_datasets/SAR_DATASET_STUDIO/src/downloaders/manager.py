from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .registry import DatasetRegistry
from .visdrone_downloader import VisDroneDownloader


class DatasetManager:
    """
    Punto central para gestionar los datasets
    de SAR Dataset Studio.
    """

    def __init__(self, studio_root: Path):

        self.studio_root = Path(studio_root)

        self.config_path = (
            self.studio_root
            / "configs"
            / "datasets.yaml"
        )

        self.registry_path = (
            self.studio_root
            / "registry"
            / "datasets.json"
        )

        self.registry = DatasetRegistry(
            self.registry_path
        )

        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No existe el catálogo: "
                f"{self.config_path}"
            )

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return yaml.safe_load(file) or {}

    def get_dataset_config(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:

        datasets = self.config.get(
            "datasets",
            {},
        )

        if dataset_id not in datasets:
            raise ValueError(
                f"Dataset no registrado: "
                f"{dataset_id}"
            )

        return datasets[dataset_id]

    def list_datasets(self) -> list[str]:

        return list(
            self.config.get(
                "datasets",
                {},
            ).keys()
        )

    def show_dataset(
        self,
        dataset_id: str,
    ) -> None:

        config = self.get_dataset_config(
            dataset_id
        )

        print()
        print("=" * 60)
        print(
            f"DATASET: {dataset_id}"
        )
        print("=" * 60)

        print(
            f"Nombre: "
            f"{config.get('name', 'N/A')}"
        )

        print(
            f"Tarea: "
            f"{config.get('task', 'N/A')}"
        )

        source = config.get(
            "source",
            {},
        )

        print(
            f"Organización: "
            f"{source.get('organization', 'N/A')}"
        )

        print(
            f"Repositorio: "
            f"{source.get('repository', 'N/A')}"
        )

        print(
            f"Fuente oficial: "
            f"{source.get('official', False)}"
        )

        print()
        print("Splits:")

        for split, information in config.get(
            "splits",
            {},
        ).items():

            print(
                f"  {split}: "
                f"{information.get('enabled', False)}"
            )

        print()
        print("Clases:")

        for class_id, class_name in config.get(
            "classes",
            {},
        ).items():

            print(
                f"  {class_id}: {class_name}"
            )

        print()

    def install(
        self,
        dataset_id: str,
    ) -> bool:

        config = self.get_dataset_config(
            dataset_id
        )

        if not config.get(
            "enabled",
            False,
        ):
            raise ValueError(
                f"Dataset deshabilitado: "
                f"{dataset_id}"
            )

        if dataset_id != "visdrone_det":
            raise ValueError(
                f"No existe todavía un downloader "
                f"para: {dataset_id}"
            )

        destination = (
            self.studio_root
            / config["destination"]["raw"]
        )

        downloader = VisDroneDownloader(
            dataset_id=dataset_id,
            config=config,
            destination=destination,
        )

        success = downloader.run()

        if success:

            self.registry.register(
                dataset_id,
                {
                    "name": config["name"],
                    "task": config["task"],
                    "source": config["source"],
                    "destination": str(
                        destination
                    ),
                    "status": (
                        "downloaded_and_validated"
                    ),
                },
            )

            print()
            print(
                "[OK] Dataset registrado "
                "en el Registry."
            )

        return success


def main():

    parser = argparse.ArgumentParser(
        description=(
            "SAR Dataset Studio - "
            "Dataset Manager"
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "list",
            "show",
            "install",
        ],
    )

    parser.add_argument(
        "dataset",
        nargs="?",
    )

    args = parser.parse_args()

    # manager.py está en:
    #
    # SAR_DATASET_STUDIO/
    #   src/
    #     downloaders/
    #
    # parents[0] -> downloaders
    # parents[1] -> src
    # parents[2] -> SAR_DATASET_STUDIO

    studio_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    manager = DatasetManager(
        studio_root
    )

    if args.command == "list":

        print()
        print(
            "Datasets registrados:"
        )
        print()

        for dataset in (
            manager.list_datasets()
        ):
            print(
                f"  - {dataset}"
            )

        print()

    elif args.command == "show":

        if not args.dataset:
            parser.error(
                "Debes indicar el dataset."
            )

        manager.show_dataset(
            args.dataset
        )

    elif args.command == "install":

        if not args.dataset:
            parser.error(
                "Debes indicar el dataset."
            )

        success = manager.install(
            args.dataset
        )

        if not success:
            raise SystemExit(
                "La instalación del dataset "
                "falló."
            )


if __name__ == "__main__":
    main()