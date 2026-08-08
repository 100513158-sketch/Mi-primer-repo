from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """
    Bounding Box normalizado.

    Todas las coordenadas están normalizadas entre 0 y 1.
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    class_name: str = ""
    confidence: Optional[float] = None
    dataset_name: str = ""

    uuid: str = field(default_factory=lambda: str(uuid4()))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def left(self) -> float:
        return self.x_center - self.width / 2

    @property
    def right(self) -> float:
        return self.x_center + self.width / 2

    @property
    def top(self) -> float:
        return self.y_center - self.height / 2

    @property
    def bottom(self) -> float:
        return self.y_center + self.height / 2

    def validate(self) -> list[str]:
        errors = []

        if self.class_id < 0:
            errors.append("class_id negativo")

        if not 0 <= self.x_center <= 1:
            errors.append("x_center fuera del rango")

        if not 0 <= self.y_center <= 1:
            errors.append("y_center fuera del rango")

        if self.width <= 0:
            errors.append("width <= 0")

        if self.height <= 0:
            errors.append("height <= 0")

        if self.width > 1:
            errors.append("width > 1")

        if self.height > 1:
            errors.append("height > 1")

        if self.left < 0:
            errors.append("bbox sale por la izquierda")

        if self.right > 1:
            errors.append("bbox sale por la derecha")

        if self.top < 0:
            errors.append("bbox sale por arriba")

        if self.bottom > 1:
            errors.append("bbox sale por abajo")

        return errors