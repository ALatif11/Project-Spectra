"""Cache-only InTheWild adapter for the existing bounded three-role runner.

Never reads audio shards, sealed manifests, or diagnostic recordings. Exports
research artifacts from the actual experiment fit, without promotion/refitting.
"""

import hashlib
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from agents.proposals import ProposalError, feature_indices, validate_proposal
from src.inthewild import LABELS, LIMITATIONS, REPO, REVISION, optimization_rows, read_features, summary
from src.model import FEATURE_SETTINGS, file_digest, load_model, segment_scores
from src.training import select_threshold


def validate_logistic(proposal):
    proposal = validate_proposal(proposal, proposal["role"])
    if proposal["classifier"] != "logistic_regression":
        raise ProposalError("This approved run permits logistic regression only.")
    return proposal


def fixed_context(root, splits, model_path, cache, splits_hash, model_hash):
    experiment = splits.parent
    if splits.name != "optimization.json" or cache.resolve() != (experiment / "features").resolve():
        raise ValueError("Use the fixed InTheWild optimization manifest and its existing feature cache.")
    if file_digest(splits) != splits_hash or file_digest(model_path) != model_hash:
        raise ValueError("Frozen optimization inputs changed.")
    artifact = load_model(model_path)
    if artifact["model_version"] != "inthewild-v1" or artifact["splits_sha256"] != splits_hash:
        raise ValueError("Use the InTheWild candidate trained on these fixed partitions.")
    rows = optimization_rows(experiment)
    if len({r["path"] for r in rows}) != len(rows) or len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate optimization clips.")
    if {r["seed"] for r in rows} != {artifact["seed"]}:
        raise ValueError("Optimization seed differs from candidate.")
    for split, key in (("train", "train_groups"), ("validation", "validation_groups")):
        if sorted({r["speaker"] for r in rows if r["split"] == split}) != artifact[key]:
            raise ValueError("Optimization speakers differ from candidate.")
    classifier = artifact["pipeline"].named_steps["classifier"]
    if classifier.solver != "lbfgs" or classifier.max_iter != 1000 or classifier.random_state != artifact["seed"]:
        raise ValueError("Candidate classifier settings differ from the bounded runner.")
    return rows, artifact


def cache_snapshot(splits):
    """Hash only allowed train/validation caches; never glob a sealed partition."""
    policy = json.dumps(FEATURE_SETTINGS, sort_keys=True)
    digest = hashlib.sha256()
    for row in optimization_rows(splits.parent):
        key = hashlib.sha256((row["sha256"] + policy).encode()).hexdigest()
        digest.update(key.encode())
        digest.update(file_digest(splits.parent / "features" / f"{key}.npz").encode())
    return digest.hexdigest()


def baseline_check(root, splits, model_path, cache, splits_hash, model_hash):
    rows, artifact = fixed_context(root, splits, model_path, cache, splits_hash, model_hash)
    # Validate every existing cache offline. Missing caches fail; never extract audio.
    scores, labels = [], []
    for row in rows:
        values, duration = read_features(splits.parent, row)
        if not len(values) or not np.isfinite(duration).all():
            raise ValueError("Invalid cached windows or weights.")
        if row["split"] == "validation":
            scores.append(float(np.average(segment_scores(artifact["pipeline"], values), weights=duration)))
            labels.append(row["label"])
    measured = summary(labels, scores, artifact["threshold"])
    if measured != artifact["validation"]["candidate"]:
        raise ValueError("Candidate validation metrics do not reproduce exactly.")
    fixed_context(root, splits, model_path, cache, splits_hash, model_hash)
    return {"metrics": measured, "seed": artifact["seed"], "n_train": sum(r["split"] == "train" for r in rows),
            "n_validation": len(labels), "cache_sha256": cache_snapshot(splits),
            "baseline_configuration": {"feature_set": "all", "classifier": "logistic_regression",
                "parameters": {"C": float(artifact["pipeline"].named_steps["classifier"].C),
                               "class_weight": artifact["pipeline"].named_steps["classifier"].class_weight}},
            "test_accessed": False, "diagnostic_used": False}


def fit_candidate(root, splits, model_path, cache, splits_hash, model_hash, proposal,
                  export_path, provenance, expected_cache_hash):
    proposal = validate_logistic(proposal)
    rows, artifact = fixed_context(root, splits, model_path, cache, splits_hash, model_hash)
    if cache_snapshot(splits) != expected_cache_hash:
        raise ValueError("Optimization cache changed since preflight.")
    columns = feature_indices(proposal["feature_set"])
    arrays, labels, weights, validation = [], [], [], []
    for row in rows:
        values, duration = read_features(splits.parent, row)
        if row["split"] == "train":
            arrays.append(values[:, columns])
            labels.extend([row["label"]] * len(values))
            weights.extend(duration / duration.sum())
        else:
            validation.append((row, values[:, columns], duration))
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(
        **proposal["parameters"], random_state=artifact["seed"], solver="lbfgs", max_iter=1000))])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(np.concatenate(arrays), np.asarray(labels), scaler__sample_weight=np.asarray(weights),
                     classifier__sample_weight=np.asarray(weights))
    scores = np.array([np.average(segment_scores(pipeline, values), weights=duration)
                       for _, values, duration in validation])
    labels = np.array([row["label"] for row, _, _ in validation])
    threshold = select_threshold(labels, scores)
    measured = summary(labels, scores, threshold)
    # Keep speaker identities local. Cloud prompts contain aggregate metrics only.
    diagnostics = {}
    for speaker in sorted({row["speaker"] for row, _, _ in validation}):
        mask = np.array([row["speaker"] == speaker for row, _, _ in validation])
        diagnostics[speaker] = summary(labels[mask], scores[mask], threshold)
    fixed_context(root, splits, model_path, cache, splits_hash, model_hash)
    if cache_snapshot(splits) != expected_cache_hash:
        raise ValueError("Optimization cache changed during fit.")
    report = {"metrics": measured, "validation_diagnostics": diagnostics, "seed": artifact["seed"],
              "feature_count": len(columns), "n_train": sum(r["split"] == "train" for r in rows),
              "saved_artifact": str(export_path), "test_accessed": False, "diagnostic_used": False}
    saved = {**artifact, "pipeline": pipeline, "threshold": threshold,
             "model_version": "inthewild-luminal-research-v1", "training_configuration": proposal,
             "feature_indices": columns, "validation": report, "sklearn_version": sklearn.__version__,
             "provenance": {**artifact["provenance"], **provenance, "baseline_sha256": model_hash,
                            "cache_sha256": expected_cache_hash},
             "deployment_status": "research_only_not_deployed"}
    export_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = export_path.with_suffix(".tmp.joblib")
    try:
        joblib.dump(saved, temporary)
        os.link(temporary, export_path)  # Exclusive publication preserves older artifacts.
    finally:
        temporary.unlink(missing_ok=True)
    return report


def prior_history(path, comparison):
    """Retain only known numeric metrics and allowlisted proposals for cloud prompts."""
    history = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["event"] != "experiment":
            continue
        proposal = event.get("proposal")
        if proposal is not None:
            proposal = validate_proposal(proposal, proposal["role"])
        metrics = event.get("metrics")
        measured = None
        if metrics:
            measured = {key: metrics[key] for key in ("balanced_accuracy", "threshold", "f1_synthetic") if key in metrics}
            tn, fp = metrics["confusion_matrix_real_fake"][0]
            fn, tp = metrics["confusion_matrix_real_fake"][1]
            measured.update(false_positive_rate=fp/(tn+fp), false_negative_rate=fn/(fn+tp))
        history.append({"dataset": "Gary Stafford (different dataset; not directly comparable)",
                        "source": "offline_static" if event["dry_run"] else "live_luminal",
                        "proposal": proposal, "metrics": measured,
                        "status": event["status"] if event["status"] in ("ok", "rejected", "failed", "timeout") else "unknown"})
    previous = json.loads(comparison.read_text())
    history.append({"dataset": "InTheWild fixed speaker-disjoint validation", "source": "local_candidate_training",
                    "proposal": {"role": "classifier", "feature_set": "all", "classifier": "logistic_regression",
                                 "parameters": {"C": 1.0, "class_weight": None}}, "metrics": previous["candidate"]})
    return history
