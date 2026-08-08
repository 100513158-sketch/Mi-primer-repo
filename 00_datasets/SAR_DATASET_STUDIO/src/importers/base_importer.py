from abc import ABC, abstractmethod

from src.core.dataset import Dataset


class BaseImporter(ABC):

    @abstractmethod
    def load(self) -> Dataset:
        """
        Carga un dataset y devuelve el modelo interno.
        """
        pass