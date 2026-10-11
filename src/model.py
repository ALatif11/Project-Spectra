"""Shared local feature policy, cache, artifact validation, and clip scoring."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.audio import SAMPLE_RATE, WINDOW_SECONDS, decode_audio, split_windows
from src.features import FEATURE_NAMES, N_MFCC, extract_features
from src.spectrogram import N_FFT, HOP_LENGTH, N_MELS

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline.joblib"
FEATURE_SETTINGS = {
    "sample_rate": SAMPLE_RATE, "window_seconds": WINDOW_SECONDS,
    "padding": "zero", "n_fft": N_FFT, "hop_length": HOP_LENGTH,
    "n_mels": N_MELS, "n_mfcc": N_MFCC, "feature_names": list(FEATURE_NAMES),
    "aggregation": "duration_weighted_mean_segment_score", "feature_version": 1,
}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_features(recording) -> tuple[np.ndarray, np.ndarray, list]:
    windows = split_windows(recording.samples)
    features = np.stack([extract_features(window.samples) for window in windows])
    weights = np.array([window.valid_samples for window in windows], dtype=float)
    return features, weights, windows


def cached_features(root: Path, row: dict, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    path = root / row["path"]
    if not path.is_file():
        raise ValueError(f"Missing clip: {path}. Rerun explicit download/preparation.")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"Clip changed since preparation: {path}")
    policy = json.dumps(FEATURE_SETTINGS, sort_keys=True).encode()
    key = hashlib.sha256(row["sha256"].encode() + policy).hexdigest()
    target = cache / f"{key}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as data:
            features, weights = data["features"], data["weights"]
    else:
        recording = decode_audio(payload)
        if recording.is_silent:
            raise ValueError(f"Silent dataset clip: {path}")
        features, weights, _ = window_features(recording)
        cache.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, features=features, weights=weights)
        temporary.replace(target)
    if (features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES)
            or weights.shape != (len(features),) or not len(features)
            or not np.isfinite(features).all() or not np.isfinite(weights).all()
            or not (weights > 0).all()):
        raise ValueError(f"Invalid feature cache: {target}. Remove that cache file and retry.")
    return features, weights


def load_model(path: Path = MODEL_PATH) -> dict:
    # Only load artifacts created locally by this workflow; joblib uses pickle.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("feature_settings") != FEATURE_SETTINGS:
        raise ValueError("Model feature settings are incompatible; retrain locally.")
    pipeline = artifact.get("pipeline")
    if (not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["scaler", "classifier"]
            or not isinstance(pipeline.named_steps["scaler"], StandardScaler)
            or not isinstance(pipeline.named_steps["classifier"], LogisticRegression)
            or not np.array_equal(pipeline.classes_, [0, 1])
            or pipeline.n_features_in_ != len(FEATURE_NAMES)):
        raise ValueError("Invalid fitted classifier.")
    threshold = artifact.get("threshold")
    if not isinstance(threshold, (float, int)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid decision threshold.")
    if artifact.get("model_version") != "baseline-v1":
        raise ValueError("Unsupported model version.")
    return artifact


def segment_scores(pipeline, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(pipeline.predict_proba(features)[:, 1], dtype=float)
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Classifier returned invalid scores.")
    return scores


def score_rows(pipeline, root: Path, rows: list[dict], cache: Path) -> np.ndarray:
    scores = []
    for row in rows:
        features, weights = cached_features(root, row, cache)
        scores.append(float(np.average(segment_scores(pipeline, features), weights=weights)))
    return np.array(scores)
