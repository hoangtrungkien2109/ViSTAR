"""Compatibility facade for ViSTAR's runtime-selectable recognizer."""

from __future__ import annotations

import numpy as np

from src.be.recognizers import get_recognizer


recognizer = get_recognizer()
actions = recognizer.actions
actions_tts = recognizer.actions_tts
num_actions = len(actions)
device = recognizer.device
threshold = recognizer.settings.threshold
sequence = []
sentence = []
predictions = []
EMA_option = True
ema_predictions = None


def prepare_recognizer_frame(
    pose_landmarks,
    left_hand_landmarks,
    right_hand_landmarks,
    legacy_features,
) -> np.ndarray:
    return recognizer.prepare_frame(
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
        legacy_features=legacy_features,
    )


def predict_recognizer_sequence(frames) -> np.ndarray:
    return recognizer.predict_proba(np.asarray(frames, dtype=np.float32))


def recognizer_status() -> dict[str, object]:
    return recognizer.status()
