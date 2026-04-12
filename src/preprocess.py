from __future__ import annotations

import numpy as np
import librosa


TARGET_LENGTH = 64000
LABEL_MAP = {"bonafide": 0, "spoof": 1}


def fix_length(waveform: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """
    Trim or pad a 1D waveform to a fixed length.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")

    if len(waveform) > target_length:
        return waveform[:target_length]

    if len(waveform) < target_length:
        pad_amount = target_length - len(waveform)
        return np.pad(waveform, (0, pad_amount), mode="constant")

    return waveform


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize waveform to roughly [-1, 1] while avoiding divide-by-zero.
    """
    max_val = np.max(np.abs(waveform))
    if max_val == 0:
        return waveform.astype(np.float32)
    return (waveform / max_val).astype(np.float32)


def encode_label(label: str) -> int:
    """
    Convert string label to integer.
    """
    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_MAP[label]


def preprocess_waveform(waveform: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """
    Full preprocessing pipeline for raw waveform input.
    """
    waveform = fix_length(waveform, target_length=target_length)
    waveform = normalize_waveform(waveform)
    return waveform

def waveform_to_mel_spectrogram(waveform, sr=16000):
    spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=128
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)

    return spec_db.astype(np.float32)