"""Fit on train windows; choose clip decision threshold on validation only."""

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.download_dataset import manifest
from src.dataset import load_splits
from src.model import FEATURE_SETTINGS, cached_features, file_digest, score_rows

LIMITATIONS = (
    "Small English research dataset; only 14 genuine YouTube sources. Source-group holdout "
    "does not establish unseen-vendor generalization. Shared text/speakers across synthetic "
    "source IDs cannot be recovered from filenames. Recording artifacts versus clean TTS "
    "may dominate; domain shift is unmeasured. Scores are uncalibrated, not forensic proof."
)


def metrics(labels, scores, threshold: float) -> dict:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    return {"n_clips": len(labels), "threshold": float(threshold),
            "accuracy": float(np.mean(predictions == labels)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "f1_synthetic": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "confusion_matrix_real_fake": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def select_threshold(labels, scores) -> float:
    unique = np.unique(scores)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], (unique[:-1] + unique[1:]) / 2)))
    # Maximize validation balanced accuracy, then choose nearest to 0.5.
    return float(max(candidates, key=lambda threshold: (
        balanced_accuracy_score(labels, scores >= threshold), -abs(threshold - 0.5), -threshold)))


def train(root: Path, splits: Path, model_path: Path, run_dir: Path, cache: Path) -> dict:
    if model_path.exists() or (run_dir / "test_started.json").exists() or (run_dir / "test_metrics.json").exists():
        raise ValueError("Preserve existing model/final evaluation; choose a new model/run path before test access.")
    rows = load_splits(splits, optimization_only=True)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    seed = int(rows[0]["seed"])
    arrays, labels, weights = [], [], []
    for row in train_rows:
        features, duration = cached_features(root, row, cache)
        arrays.append(features)
        labels.extend([int(row["label"])] * len(features))
        # Each clip has equal total weight, shared by its actual window durations.
        weights.extend(duration / duration.sum())
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
        C=1.0, max_iter=1000, random_state=seed, solver="lbfgs"))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(np.concatenate(arrays), np.array(labels),
                     scaler__sample_weight=np.array(weights), classifier__sample_weight=np.array(weights))
    scores = score_rows(pipeline, root, validation, cache)
    validation_labels = np.array([int(row["label"]) for row in validation])
    threshold = select_threshold(validation_labels, scores)
    report = {"partition": "validation", "seed": seed, "source": manifest()["repo_id"],
              "revision": manifest()["revision"], "splits_sha256": file_digest(splits),
              "metrics": metrics(validation_labels, scores, threshold),
              "threshold_selection": "maximum validation clip balanced accuracy; ties nearest 0.5",
              "groups": sorted({row["group"] for row in validation}),
              "limitations": LIMITATIONS,
              "clips": [dict(path=row["path"], label=int(row["label"]), group=row["group"],
                             synthetic_score=float(score)) for row, score in zip(validation, scores)]}
    artifact = {"pipeline": pipeline, "threshold": threshold, "model_version": "baseline-v1",
                "feature_settings": FEATURE_SETTINGS, "seed": seed, "validation": report,
                "sklearn_version": sklearn.__version__, "splits_sha256": file_digest(splits),
                "train_groups": sorted({row["group"] for row in train_rows}),
                "validation_groups": report["groups"]}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.joblib")
    joblib.dump(artifact, temporary)
    temporary.replace(model_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
