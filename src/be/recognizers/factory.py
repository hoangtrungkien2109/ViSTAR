"""Recognizer construction and process-local singleton management."""

from __future__ import annotations

from threading import Lock

from .config import RecognizerSettings
from .runtime import BaselineRecognizer, FELFSLRRecognizer, SignRecognizer


_recognizer: SignRecognizer | None = None
_lock = Lock()


def get_recognizer(
    settings: RecognizerSettings | None = None,
) -> SignRecognizer:
    global _recognizer
    if _recognizer is not None and settings is None:
        return _recognizer
    with _lock:
        if _recognizer is not None and settings is None:
            return _recognizer
        selected = settings or RecognizerSettings.from_env()
        if selected.mode == "baseline":
            instance: SignRecognizer = BaselineRecognizer(selected)
        else:
            instance = FELFSLRRecognizer(selected)
        if settings is None:
            _recognizer = instance
        return instance


def reset_recognizer() -> None:
    global _recognizer
    with _lock:
        _recognizer = None
