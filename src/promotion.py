"""Offline reproduction and checkpoint export from existing experiment events.

No provider client is constructed and no experiment history is appended here.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import joblib
import librosa
import numpy as np
import sklearn
import soundfile

from agents.experiment_runner import fit_candidate_model, fixed_context
from agents.proposals import validate_proposal
from scripts.download_dataset import manifest
from src.dataset import LABELS
from src.model import FEATURE_SETTINGS, file_digest, load_model, score_rows
from src.training import LIMITATIONS, metrics

# Discrete classification metrics/counts must match exactly. Continuous score
# summaries allow only floating-point roundoff, far below a one-clip change.
NUMERICAL_TOLERANCE = 1e-9


def verify_metrics(actual: dict, expected: dict) -> None:
    if set(actual) != set(expected):
        raise ValueError("Validation metric keys differ from the saved experiment.")
    for key, value in expected.items():
        if key in ("threshold", "roc_auc"):
            matches = bool(np.isclose(actual[key], value, rtol=0, atol=NUMERICAL_TOLERANCE))
        else:
            matches = actual[key] == value
        if not matches:
            raise ValueError(f"Validation mismatch in {key}: {actual[key]} versus {value}; "
                             "inspect feature policy, library versions, seed and input hashes before export.")


def winning_event(history: Path, run_id: str) -> dict:
    events = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    events = [event for event in events if event["run_id"] == run_id]
    finished = [event for event in events if event["event"] == "run_finished"]
    if len(finished) != 1 or finished[0]["dry_run"] or finished[0]["mode"] != "cloud":
        raise ValueError("Select a completed, real provider run from the saved history.")
    final = finished[0]
    candidates = [event for event in events if event["event"] == "experiment"
                  and event["status"] == "ok" and event["metrics"] is not None]
    if not candidates:
        raise ValueError("Run contains no successful measured experiment.")
    winner = max(candidates, key=lambda event: event["metrics"]["balanced_accuracy"])
    if (winner["proposal"] != final["best_proposal"]
            or winner["metrics"]["balanced_accuracy"] != final["best_balanced_accuracy"]
            or winner["decision"] != "accepted"):
        raise ValueError("Winner disagrees with the completed run; inspect saved events.")
    validate_proposal(winner["proposal"], winner["role"])
    # This deployment format intentionally supports the current full-feature LR
    # winner only; other experiment families require an explicit format change.
    if winner["proposal"]["feature_set"] != "all" or winner["proposal"]["classifier"] != "logistic_regression":
        raise ValueError("Deployment format requires all features and logistic regression.")
    if any(event["splits_sha256"] != winner["splits_sha256"]
           or event["baseline_sha256"] != winner["baseline_sha256"] for event in events):
        raise ValueError("Run input identities changed.")
    return winner


def reproduce(*, root: Path, splits: Path, baseline_path: Path, history: Path,
              run_id: str, model_path: Path, run_dir: Path, cache: Path) -> dict:
    if model_path.resolve() == baseline_path.resolve():
        raise ValueError("Winning checkpoint must have a separate path from the baseline.")
    event = winning_event(history, run_id)
    rows, baseline = fixed_context(root, splits, baseline_path, event["splits_sha256"], event["baseline_sha256"])
    if event["seed"] != baseline["seed"]:
        raise ValueError("Experiment seed differs from the preserved baseline.")
    if model_path.exists():
        artifact = load_model(model_path)
        if (artifact.get("provenance", {}).get("run_id") != run_id
                or artifact.get("training_configuration") != event["proposal"]
                or artifact.get("splits_sha256") != event["splits_sha256"]
                or artifact.get("provenance", {}).get("baseline_sha256") != event["baseline_sha256"]):
            raise ValueError("Existing checkpoint has different provenance; preserve it.")
        verify_metrics(artifact["validation"]["metrics"], event["metrics"])
        return artifact["validation"]
    if any((run_dir / name).exists() for name in ("test_started.json", "test_metrics.json")):
        raise ValueError("Test access already started; refusing to refit a model in this run.")
    history_hash = file_digest(history)
    pipeline, measured = fit_candidate_model(root, splits, baseline_path, cache,
        event["splits_sha256"], event["baseline_sha256"], event["proposal"])
    verify_metrics(measured["metrics"], event["metrics"])
    validation = [row for row in rows if row["split"] == "validation"]
    # Verify the exported pipeline's ordinary inference layout too.
    scores = score_rows(pipeline, root, validation, cache)
    labels = np.array([int(row["label"]) for row in validation])
    threshold = event["metrics"]["threshold"]  # Freeze the original validation decision rule.
    verify_metrics(metrics(labels, scores, threshold), event["metrics"])
    report = {"partition": "validation", **measured, "source": manifest()["repo_id"],
              "revision": manifest()["revision"], "groups": baseline["validation_groups"],
              "reproduced_at": datetime.now(timezone.utc).isoformat(),
              "expected_metrics": event["metrics"], "numerical_tolerance": NUMERICAL_TOLERANCE,
              "discrete_metrics_match": "exact", "limitations": LIMITATIONS,
              "clips": [dict(path=row["path"], label=int(row["label"]), group=row["group"],
                             synthetic_score=float(score)) for row, score in zip(validation, scores)]}
    artifact = {"pipeline": pipeline, "threshold": threshold, "model_version": "optimized-v1",
                "feature_settings": FEATURE_SETTINGS, "label_mapping": LABELS,
                "training_configuration": event["proposal"], "seed": baseline["seed"],
                "validation": report, "splits_sha256": event["splits_sha256"],
                "train_groups": baseline["train_groups"], "validation_groups": baseline["validation_groups"],
                "sklearn_version": sklearn.__version__,
                "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                     "librosa": librosa.__version__, "soundfile": soundfile.__version__},
                "inference_metadata": {"score": "uncalibrated synthetic class output",
                    "abstention_band": 0.05, "training_weighting": "equal clip weight, duration weighted windows",
                    "threshold_selection": "validation balanced accuracy; ties nearest 0.5"},
                "provenance": {"run_id": run_id, "round": event["round"], "role": event["role"],
                    "event_timestamp": event["timestamp"], "history_sha256_at_export": history_hash,
                    "baseline_sha256": event["baseline_sha256"], "source": manifest()["repo_id"],
                    "revision": manifest()["revision"]}}
    fixed_context(root, splits, baseline_path, event["splits_sha256"], event["baseline_sha256"])
    if file_digest(history) != history_hash:
        raise ValueError("Experiment history changed during reproduction; retry offline export after inspection.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.joblib")
    try:
        joblib.dump(artifact, temporary)
        load_model(temporary)
        # Same-directory hard link publishes a complete file atomically, and
        # fails if the destination exists; never replace another checkpoint.
        os.link(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata = {key: value for key, value in artifact.items() if key not in ("pipeline", "validation")}
    metadata["model_sha256"] = file_digest(model_path)
    (run_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return report
