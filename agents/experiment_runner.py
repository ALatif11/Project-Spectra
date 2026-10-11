"""Deterministic candidate fits on frozen train/validation rows only."""

from dataclasses import dataclass
import math
import multiprocessing
import os
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.configuration import bounded_env_int, load_environment
from agents.proposals import ProposalError, feature_indices, validate_proposal
from src.dataset import load_splits
from src.inference import predict_audio
from src.model import cached_features, file_digest, load_model, segment_scores
from src.training import metrics, select_threshold


@dataclass(frozen=True)
class Limits:
    rounds: int = 3
    agents_per_round: int = 3
    llm_calls: int = 20
    experiments: int = 12
    train_timeout_s: float = 120.0
    run_timeout_s: float = 600.0

    @classmethod
    def from_env(cls):
        load_environment()
        return cls(rounds=bounded_env_int("MAX_AGENT_ROUNDS", 3),
                   llm_calls=bounded_env_int("MAX_LLM_CALLS", 20),
                   experiments=bounded_env_int("MAX_MODEL_EXPERIMENTS", 12))

    def __post_init__(self):
        for name, maximum in (("rounds", 3), ("agents_per_round", 3), ("llm_calls", 20), ("experiments", 12)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}; hard caps cannot be raised.")
        for name in ("train_timeout_s", "run_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= 3600 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite, positive, and at most 3600 seconds.")


def fixed_context(root: Path, splits: Path, model_path: Path, splits_hash: str, model_hash: str):
    if file_digest(splits) != splits_hash or file_digest(model_path) != model_hash:
        raise ValueError("Frozen splits or baseline artifact changed; refusing optimization.")
    rows = load_splits(splits, optimization_only=True)
    artifact = load_model(model_path)
    if artifact["splits_sha256"] != splits_hash:
        raise ValueError("Baseline was trained on different partitions; preserve its original split file.")
    for partition, key in (("train", "train_groups"), ("validation", "validation_groups")):
        if sorted({row["group"] for row in rows if row["split"] == partition}) != artifact[key]:
            raise ValueError("Baseline source groups differ from fixed partitions.")
    if int(rows[0]["seed"]) != artifact["seed"]:
        raise ValueError("Baseline seed differs from fixed splits.")
    return rows, artifact


def baseline_check(root, splits, model_path, cache, splits_hash, model_hash):
    rows, artifact = fixed_context(root, splits, model_path, splits_hash, model_hash)
    validation = [row for row in rows if row["split"] == "validation"]
    previous_path = os.environ.get("ECHOPROOF_MODEL_PATH")
    try:
        os.environ["ECHOPROOF_MODEL_PATH"] = str(model_path.resolve())
        result = predict_audio((root / validation[0]["path"]).read_bytes())
    finally:
        if previous_path is None:
            os.environ.pop("ECHOPROOF_MODEL_PATH", None)
        else:
            os.environ["ECHOPROOF_MODEL_PATH"] = previous_path
    if result["status"] != "ok":
        raise ValueError("Baseline inference did not return ok; repair the local model before experiments.")
    # Recompute actual validation metrics, without fitting or accessing final test rows.
    scores = []
    for row in validation:
        values, duration = cached_features(root, row, cache)
        scores.append(float(np.average(segment_scores(artifact["pipeline"], values), weights=duration)))
    labels = np.array([int(row["label"]) for row in validation])
    measured = metrics(labels, np.array(scores), artifact["threshold"])
    if measured != artifact["validation"]["metrics"]:
        raise ValueError("Baseline validation results no longer reproduce; repair the baseline before optimizing.")
    fixed_context(root, splits, model_path, splits_hash, model_hash)
    return {"metrics": measured, "seed": artifact["seed"],
            "n_train": sum(row["split"] == "train" for row in rows), "n_validation": len(validation)}


def fit_candidate(root, splits, model_path, cache, splits_hash, model_hash, proposal):
    _, report = fit_candidate_model(root, splits, model_path, cache, splits_hash, model_hash, proposal)
    return report


def fit_candidate_model(root, splits, model_path, cache, splits_hash, model_hash, proposal):
    """Same deterministic experiment fit, retaining the fitted pipeline for export."""
    proposal = validate_proposal(proposal, proposal["role"])
    rows, artifact = fixed_context(root, splits, model_path, splits_hash, model_hash)
    seed = artifact["seed"]
    columns = feature_indices(proposal["feature_set"])
    arrays, labels, weights = [], [], []
    validation = []
    for row in rows:
        values, duration = cached_features(root, row, cache)
        if row["split"] == "train":
            arrays.append(values[:, columns])
            labels.extend([int(row["label"])] * len(values))
            weights.extend(duration / duration.sum())
        else:
            validation.append((row, values[:, columns], duration))
    params = proposal["parameters"]
    if proposal["classifier"] == "logistic_regression":
        classifier = LogisticRegression(**params, random_state=seed, solver="lbfgs", max_iter=1000)
    else:
        classifier = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    # No fitted transform ever sees validation data. Equal total weight per clip.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(np.concatenate(arrays), np.asarray(labels), scaler__sample_weight=np.asarray(weights),
                     classifier__sample_weight=np.asarray(weights))
    scores = np.array([np.average(segment_scores(pipeline, values), weights=duration)
                       for _, values, duration in validation])
    validation_labels = np.array([int(row["label"]) for row, _, _ in validation])
    threshold = select_threshold(validation_labels, scores)
    measured = metrics(validation_labels, scores, threshold)
    # These subsets were present during optimization: no unseen-vendor claims.
    diagnostics = {}
    for vendor in sorted({row["vendor"] for row, _, _ in validation}):
        mask = np.array([row["vendor"] == vendor for row, _, _ in validation])
        predicted = scores[mask] >= threshold
        diagnostics[vendor] = {"n_clips": int(mask.sum()),
            "class_recall": float(np.mean(predicted == validation_labels[mask])),
            "label": int(validation_labels[mask][0])}
    fixed_context(root, splits, model_path, splits_hash, model_hash)
    return pipeline, {"metrics": measured, "validation_diagnostics": diagnostics, "seed": seed,
                      "feature_count": len(columns), "n_train": sum(row["split"] == "train" for row in rows)}


def _child_entry(connection, action, args):
    try:
        connection.send({"status": "ok", "result": action(*args), "reason": None})
    except ProposalError:
        connection.send({"status": "rejected", "result": None, "reason": "proposal_outside_allowlist"})
    except Exception as exc:
        # Never echo provider bodies, credentials, arbitrary messages, or paths.
        reason = "operation_failed"
        from agents.luminal_client import LuminalError
        if isinstance(exc, LuminalError):
            reason = str(exc)  # Client errors are explicitly redacted/actionable.
        elif isinstance(exc, ConvergenceWarning):
            reason = "classifier_did_not_converge"
        elif isinstance(exc, ValueError):
            reason = "data_or_frozen_artifact_validation_failed"
        connection.send({"status": "failed", "result": None, "reason": reason})
    finally:
        connection.close()


def bounded_call(action, args: tuple, timeout_s: float) -> dict:
    """A killable process bounds feature extraction, fitting, and cloud calls alike."""
    if timeout_s <= 0:
        return {"status": "timeout", "result": None, "reason": "run_timeout", "runtime_s": 0.0}
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child_entry, args=(send, action, args), daemon=True)
    started = time.monotonic()
    try:
        process.start()
        send.close()
        if receive.poll(max(0.0, timeout_s - (time.monotonic() - started))):
            try:
                outcome = receive.recv()
            except EOFError:
                outcome = {"status": "failed", "result": None, "reason": "worker_exited_without_result"}
        else:
            outcome = {"status": "timeout", "result": None, "reason": "operation_timeout"}
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
        receive.close()
        send.close()
    return dict(outcome, runtime_s=time.monotonic() - started)
