from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .annotation import Annotation


@dataclass(slots=True)
class ImageRecord:
    """
    Representa una imagen del dataset.

    Una imagen contiene cero o más anotaciones y toda la información
    necesaria para validar su integridad.
    """

    image_path: Path

    width: int
    height: int

    annotations: list[Annotation] = field(default_factory=list)

    dataset_name: str = ""

    sha256: str = ""
    perceptual_hash: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    uuid: str = field(default_factory=lambda: str(uuid4()))

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def extension(self) -> str:
        return self.image_path.suffix.lower()

    @property
    def object_count(self) -> int:
        return len(self.annotations)

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)

    def validate(self) -> list[str]:
        """
        Valida la imagen y todas sus anotaciones.
        """

        errors = []

        if self.width <= 0:
            errors.append("Anchura inválida")

        if self.height <= 0:
            errors.append("Altura inválida")

        if not self.image_path.exists():
            errors.append("La imagen no existe")

        for annotation in self.annotations:
            errors.extend(annotation.validate())

        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def class_distribution(self) -> dict[int, int]:
        """
        Devuelve el número de objetos por clase.
        """

        result: dict[int, int] = {}

        for ann in self.annotations:
            class_id = ann.class_id
            result[class_id] = result.get(class_id, 0) + 1

        return result