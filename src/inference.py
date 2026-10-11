"""Stable local inference interface. Never downloads or contacts a remote service."""

import os
from pathlib import Path

import numpy as np

from src.audio import AudioDecodeError, decode_audio
from src.model import MODEL_PATH, load_model, segment_scores, window_features


def predict_audio(audio_bytes: bytes, filename: str | None = None) -> dict:
    """Run local segment inference using a locally trained, trusted model artifact."""
    result = {"status": "unavailable", "label": "inconclusive", "synthetic_score": None,
              "threshold": None, "segments": [], "model_version": "baseline-v1",
              "message": "Train the local model using python -m scripts.train_model."}
    try:
        recording = decode_audio(audio_bytes, filename)
    except AudioDecodeError as exc:
        return dict(result, status="invalid", message=str(exc))
    path = Path(os.environ.get("ECHOPROOF_MODEL_PATH", str(MODEL_PATH)))
    if not path.is_file():
        return result
    if recording.is_silent or recording.duration_s < 0.25:
        return dict(result, status="invalid", message="Upload at least 0.25 seconds of audible speech; silence cannot be screened.")
    try:
        artifact = load_model(path)
        result["model_version"] = artifact["model_version"]
        features, weights, windows = window_features(recording)
        scores = segment_scores(artifact["pipeline"], features)
        score = float(np.average(scores, weights=weights))
        threshold = float(artifact["threshold"])
    except Exception:
        return dict(result, message="Local model could not run. Check its feature settings and retrain with the backend dependencies.")
    label = "likely_synthetic" if score >= threshold else "likely_human"
    # No calibration is claimed; a narrow decision boundary band is a UI abstention rule.
    if abs(score - threshold) <= 0.05:
        label = "inconclusive"
    return dict(result, status="ok", label=label, synthetic_score=score, threshold=threshold,
                segments=[{"start_s": float(window.start_s), "end_s": float(window.end_s),
                           "synthetic_score": float(value)} for window, value in zip(windows, scores)],
                message="Research-grade estimate; not forensic proof. Uncalibrated score; recording artifacts and unseen generators may cause errors.")
