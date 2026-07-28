"""Runtime-selectable sign recognizers for ViSTAR."""

from .factory import get_recognizer, reset_recognizer

__all__ = ["get_recognizer", "reset_recognizer"]
