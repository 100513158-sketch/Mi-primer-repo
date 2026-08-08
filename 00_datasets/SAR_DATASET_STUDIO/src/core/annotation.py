from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .bounding_box import BoundingBox
from .enums import AnnotationStatus


@dataclass(slots=True)
class Annotation:
    """
    Representa una anotación de un objeto dentro de una imagen.

    Es independiente del formato del dataset (YOLO, COCO, VisDrone...).
    """

    bounding_box: BoundingBox

    status: AnnotationStatus = AnnotationStatus.UNKNOWN

    occluded: bool = False
    truncated: bool = False
    ignored: bool = False

    source_dataset: str = ""
    source_image: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    uuid: str = field(default_factory=lambda: str(uuid4()))

    @property
    def class_id(self) -> int:
        return self.bounding_box.class_id

    @property
    def class_name(self) -> str:
        return self.bounding_box.class_name

    def validate(self) -> list[str]:
        """
        Valida la anotación completa.
        """

        errors = []

        errors.extend(self.bounding_box.validate())

        return errors

    def is_valid(self) -> bool:
        """
        Devuelve True si la anotación no contiene errores.
        """
        return len(self.validate()) == 0