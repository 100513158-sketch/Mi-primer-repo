from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .image_record import ImageRecord


@dataclass(slots=True)
class Dataset:
    """
    Representa un dataset completo.

    Toda la aplicación trabajará sobre esta clase,
    independientemente del formato original
    (YOLO, COCO, VisDrone, DOTA...).
    """

    name: str

    root_path: Path

    images: list[ImageRecord] = field(default_factory=list)

    class_map: dict[int, str] = field(default_factory=dict)

    metadata: dict = field(default_factory=dict)

    uuid: str = field(default_factory=lambda: str(uuid4()))

    # -------------------------------------------------------
    # Gestión de imágenes
    # -------------------------------------------------------

    def add_image(self, image: ImageRecord) -> None:
        self.images.append(image)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def annotation_count(self) -> int:
        return sum(len(img.annotations) for img in self.images)

    @property
    def class_count(self) -> int:
        return len(self.class_map)

    # -------------------------------------------------------
    # Estadísticas
    # -------------------------------------------------------

    def class_distribution(self) -> dict[int, int]:

        result = {}

        for image in self.images:

            for ann in image.annotations:

                cid = ann.class_id

                result[cid] = result.get(cid, 0) + 1

        return result

    def average_objects_per_image(self) -> float:

        if not self.images:
            return 0.0

        return self.annotation_count / self.image_count

    # -------------------------------------------------------
    # Validación
    # -------------------------------------------------------

    def validate(self) -> list[str]:

        errors = []

        if self.image_count == 0:
            errors.append("Dataset vacío")

        for image in self.images:

            errors.extend(image.validate())

        return errors

    def is_valid(self) -> bool:

        return len(self.validate()) == 0

    # -------------------------------------------------------
    # Resumen
    # -------------------------------------------------------

    def summary(self) -> str:

        return (
            f"\n"
            f"Dataset: {self.name}\n"
            f"UUID: {self.uuid}\n"
            f"Imágenes: {self.image_count}\n"
            f"Anotaciones: {self.annotation_count}\n"
            f"Clases: {self.class_count}\n"
            f"Objetos/imagen: {self.average_objects_per_image():.2f}\n"
        )