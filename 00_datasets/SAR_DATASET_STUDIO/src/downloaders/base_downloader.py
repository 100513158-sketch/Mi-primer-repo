from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDownloader(ABC):
    """
    Clase base para todos los descargadores de datasets.
    """

    def __init__(
        self,
        dataset_id: str,
        config: dict[str, Any],
        destination: Path
    ):

        self.dataset_id = dataset_id

        self.config = config

        self.destination = Path(destination)

        self.destination.mkdir(
            parents=True,
            exist_ok=True
        )

    @abstractmethod
    def download(self) -> None:
        """
        Descarga el dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def verify(self) -> bool:
        """
        Verifica la integridad del dataset descargado.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_structure(self) -> bool:
        """
        Comprueba que la estructura del dataset sea correcta.
        """
        raise NotImplementedError