from enum import Enum


class AnnotationStatus(str, Enum):
    """Estados posibles de una anotación."""

    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"
