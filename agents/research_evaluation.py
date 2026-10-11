"""Cached training-speaker development folds, inner calibration and final gate.

All dataset reads derive from optimization.json. No sealed/test path or audio
decoder is imported. The old training/validation memberships are never edited.
"""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from agents.inthewild_context import cache_snapshot
from agents.research_proposals import canonical, columns, fingerprint, validate_configuration
from src.features import FEATURE_NAMES
from src.inthewild import optimization_rows, read_features, summary
from src.model import FEATURE_SETTINGS, file_digest
from src.training import metrics as frontend_metrics

PROTOCOL = "train-only-3-speaker-folds-inner-calibration-v2"
OBJECTIVE = {"fold_balanced_accuracy_weight": .6, "speaker_macro_balanced_accuracy_weight": .3,
             "worst_class_recall_weight": .1, "fold_std_penalty": .1,
             "max_mean_ba_regression": .005, "max_fpr_regression": .02, "max_fnr_regression": .02}


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare_bundle(experiment, cache_root):
    """Read-only source caches -> new bounded arrays and fixed training-only folds."""
    rows = optimization_rows(experiment)
    source_hash = file_digest(experiment / "optimization.json")
    cache_hash = cache_snapshot(experiment / "optimization.json")
    identity = {"splits_sha256": source_hash, "cache_sha256": cache_hash, "feature_settings": FEATURE_SETTINGS,
                "protocol": PROTOCOL, "seed": 42, "evaluation_source_sha256": file_digest(Path(__file__))}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = cache_root / key
    if (target / "bundle.json").exists():
        meta = json.loads((target / "bundle.json").read_text())
        if meta["identity"] != identity or any(file_digest(target/p) != h for p, h in meta["array_hashes"].items()):
            raise ValueError("Prepared feature bundle changed.")
        return target, meta
    target.mkdir(parents=True, exist_ok=False)
    counts = {}; hashes = {}
    for partition in ("train", "validation"):
        selected = [r for r in rows if r["split"] == partition]
        speaker_codes = {s: i for i, s in enumerate(sorted({r["speaker"] for r in selected}))}
        arrays, durations, offsets = [], [], [0]
        for row in selected:
            values, weights = read_features(experiment, row)
            if not len(values) or not np.isfinite(weights).all():
                raise ValueError("Invalid source feature cache.")
            arrays.append(values); durations.append(weights); offsets.append(offsets[-1]+len(values))
        output = {"values": np.concatenate(arrays), "durations": np.concatenate(durations),
                  "offsets": np.asarray(offsets, dtype=np.int64),
                  "labels": np.asarray([r["label"] for r in selected], dtype=np.int64),
                  "groups": np.asarray([speaker_codes[r["speaker"]] for r in selected], dtype=np.int64)}
        for name, values in output.items():
            p = target / f"{partition}-{name}.npy"
            with p.open("xb") as stream:
                np.save(stream, values, allow_pickle=False)
            hashes[p.name] = file_digest(p)
        counts[partition] = {"clips": len(selected), "windows": offsets[-1], "speakers": len(speaker_codes)}
    train = load_data(target, "train")
    plan = development_plan(train["labels"], train["groups"], 42)
    save_json(target / "folds.json", plan)
    if cache_snapshot(experiment / "optimization.json") != cache_hash or file_digest(experiment / "optimization.json") != source_hash:
        raise ValueError("Source caches/manifest changed while preparing bundle.")
    meta = {"identity": identity, "counts": counts, "array_hashes": hashes,
            "folds_sha256": file_digest(target / "folds.json"), "test_accessed": False, "diagnostic_used": False}
    save_json(target / "bundle.json", meta)
    return target, meta


def load_data(bundle, partition):
    if partition not in ("train", "validation"):
        raise ValueError("Only optimization partitions are supported.")
    return {name: np.load(bundle / f"{partition}-{name}.npy", allow_pickle=False, mmap_mode="r")
            for name in ("values", "durations", "offsets", "labels", "groups")}


def development_plan(labels, groups, seed):
    folds = []
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for number, (pool, heldout) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        if set(labels[heldout]) != {0, 1}:
            raise ValueError("Each development fold requires both classes.")
        inner = GroupShuffleSplit(n_splits=50, test_size=.2, random_state=seed+100+number)
        for fit_local, calibration_local in inner.split(pool, labels[pool], groups[pool]):
            fit, calibration = pool[fit_local], pool[calibration_local]
            if set(labels[fit]) == {0, 1} and set(labels[calibration]) == {0, 1}:
                break
        else:
            raise ValueError("Cannot create both-class inner calibration groups.")
        parts = (fit, calibration, heldout)
        if any(set(groups[a]) & set(groups[b]) for i, a in enumerate(parts) for b in parts[i+1:]):
            raise ValueError("Development speaker leakage.")
        folds.append({"fold": number+1, "fit": fit.tolist(), "calibration": calibration.tolist(), "heldout": heldout.tolist()})
    if sorted(i for f in folds for i in f["heldout"]) != list(range(len(labels))):
        raise ValueError("Development folds must cover training clips exactly once.")
    return {"protocol": PROTOCOL, "seed": seed, "folds": folds, "objective": OBJECTIVE,
            "validation_used_for_selection": False, "test_accessed": False}


def fast_threshold(labels, scores):
    """Same balanced-accuracy and tie objective as the original exhaustive search."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Both calibration classes and finite scores required.")
    unique = np.unique(scores)
    candidates = np.unique(np.r_[0., .5, 1., (unique[:-1]+unique[1:])/2])
    order = np.argsort(scores, kind="stable")
    positive = np.r_[0, np.cumsum(labels[order])]
    negative = np.r_[0, np.cumsum(1-labels[order])]
    index = np.searchsorted(scores[order], candidates, side="left")
    ba = .5*(negative[index]/negative[-1] + (positive[-1]-positive[index])/positive[-1])
    best = max(range(len(candidates)), key=lambda i: (ba[i], -abs(candidates[i]-.5), -candidates[i]))
    return float(candidates[best])


def clip_scores(pipeline, data, selected_columns):
    window = pipeline.predict_proba(np.asarray(data["values"][:, selected_columns]))[:, 1]
    offsets, durations = data["offsets"], data["durations"]
    return np.add.reduceat(window*durations, offsets[:-1])/np.add.reduceat(durations, offsets[:-1])


def pipeline_for(proposal, seed):
    proposal = validate_configuration(proposal)
    cls = {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier,
           "hist_gradient_boosting": HistGradientBoostingClassifier}[proposal["classifier"]]
    fixed = {"random_state": seed}
    if proposal["classifier"] == "logistic_regression":
        fixed.update(solver="lbfgs", max_iter=1000)
    elif proposal["classifier"] == "random_forest":
        fixed.update(n_jobs=1)
    else:
        fixed.update(early_stopping=False, max_depth=3)
    return Pipeline([("scaler", StandardScaler()), ("classifier", cls(**proposal["parameters"], **fixed))])


def journal(path, event, **fields):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def fit_pipeline(pipeline, data, clips, selected_columns, fit_journal, fold):
    lengths = np.diff(data["offsets"])
    mask = np.repeat(np.isin(np.arange(len(data["labels"])), clips), lengths)
    labels = np.repeat(data["labels"], lengths)[mask]
    clip_duration = np.add.reduceat(data["durations"], data["offsets"][:-1])
    weights = (data["durations"]/np.repeat(clip_duration, lengths))[mask]
    journal(fit_journal, "fit_started", fold=fold, training_clips=len(clips))
    started = time.monotonic()
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(np.asarray(data["values"][mask][:, selected_columns]), labels,
                         scaler__sample_weight=weights, classifier__sample_weight=weights)
    except Exception:
        journal(fit_journal, "fit_failed", fold=fold, training_runtime_s=time.monotonic()-started)
        raise
    elapsed = time.monotonic()-started
    journal(fit_journal, "fit_completed", fold=fold, training_runtime_s=elapsed)
    return elapsed


def selection_score(folds, speakers, pooled):
    mean = float(np.mean([m["balanced_accuracy"] for m in folds]))
    std = float(np.std([m["balanced_accuracy"] for m in folds]))
    speaker_ba = [s["balanced_accuracy"] for s in speakers.values() if s["balanced_accuracy"] is not None]
    if not speaker_ba:
        raise ValueError("Both-class speakers required for robust selection.")
    macro = float(np.mean(speaker_ba))
    score = .6*mean + .3*macro + .1*(1-max(pooled["false_positive_rate"], pooled["false_negative_rate"])) - .1*std
    return {"score": float(score), "mean_fold_balanced_accuracy": mean, "fold_balanced_accuracy_std": std,
            "speaker_macro_balanced_accuracy": macro, "speaker_balanced_accuracy_std": float(np.std(speaker_ba)),
            "both_class_speakers": len(speaker_ba)}


def passes_guardrails(candidate, baseline):
    return (candidate["objective"]["mean_fold_balanced_accuracy"] >= baseline["objective"]["mean_fold_balanced_accuracy"]-.005
            and candidate["metrics"]["false_positive_rate"] <= baseline["metrics"]["false_positive_rate"]+.02
            and candidate["metrics"]["false_negative_rate"] <= baseline["metrics"]["false_negative_rate"]+.02)


def diagnostics(data, predictions, pipelines, selected_columns):
    # Out-of-fold errors only; no microphone or main-validation error feedback.
    lengths = np.diff(data["offsets"])
    clip_values = np.add.reduceat(np.asarray(data["values"])*data["durations"][:, None], data["offsets"][:-1], axis=0)
    clip_values /= np.add.reduceat(data["durations"], data["offsets"][:-1])[:, None]
    fp = (data["labels"] == 0) & predictions
    tn = (data["labels"] == 0) & ~predictions
    association = []
    if fp.any() and tn.any():
        difference = (clip_values[fp].mean(axis=0)-clip_values[tn].mean(axis=0))/np.maximum(clip_values.std(axis=0), 1e-8)
        association = [{"feature": FEATURE_NAMES[i], "standardized_fp_minus_tn": float(difference[i])}
                       for i in np.argsort(-np.abs(difference))[:8]]
    contribution = defaultdict(list)
    for pipeline in pipelines:
        cls = pipeline.named_steps["classifier"]
        importance = np.abs(cls.coef_[0]) if hasattr(cls, "coef_") else getattr(cls, "feature_importances_", None)
        if importance is not None:
            for j, original in enumerate(selected_columns):
                name = FEATURE_NAMES[original]
                family = "mfcc_"+name.rsplit("_", 1)[-1] if name.startswith("mfcc_") else name.rsplit("_", 1)[0]
                contribution[family].append(float(importance[j]))
    return {"feature_family_contribution": {k: float(np.mean(v)) for k, v in contribution.items()},
            "contribution_semantics": "absolute standardized logistic coefficients or forest importance; descriptive, not causal",
            "false_positive_feature_associations": association,
            "error_source": "training-speaker out-of-fold predictions, adaptively reused development data"}


def evaluation_key(bundle, proposal):
    return hashlib.sha256((file_digest(bundle/"bundle.json") + file_digest(bundle/"folds.json")
                           + sklearn.__version__ + fingerprint(proposal)).encode()).hexdigest()


def cached_evaluation(bundle, proposal, evaluation_cache):
    target = evaluation_cache/evaluation_key(bundle, proposal)
    if (target/"report.json").exists():
        report = json.loads((target/"report.json").read_text())
        if report["configuration"] != canonical(proposal) or file_digest(target/"fold-models.joblib") != report["artifact_sha256"]:
            raise ValueError("Cached development evaluation changed.")
        return report
    return None


def evaluate_configuration(bundle, proposal, evaluation_cache, fit_journal, max_fits=3):
    proposal = validate_configuration(proposal)
    existing = cached_evaluation(bundle, proposal, evaluation_cache)
    if existing is not None:
        return {"cache_hit": True, "report": existing}
    if max_fits < 3:
        raise ValueError("Three actual fold fits must be reserved before evaluation.")
    data = load_data(bundle, "train")  # Main validation is inaccessible in this function.
    plan = json.loads((bundle/"folds.json").read_text())
    if len(plan["folds"]) != 3:
        raise ValueError("Only three reserved development fits are permitted.")
    selected_columns = columns(proposal["feature_set"])
    scores = np.full(len(data["labels"]), np.nan)
    predictions = np.zeros(len(scores), dtype=bool)
    pipelines, reports, thresholds, training_times, compatibility_metrics = [], [], [], [], []
    with threadpool_limits(limits=1):
        for fold in plan["folds"]:
            pipeline = pipeline_for(proposal, plan["seed"])
            elapsed = fit_pipeline(pipeline, data, fold["fit"], selected_columns, fit_journal, fold["fold"])
            all_scores = clip_scores(pipeline, data, selected_columns)
            threshold = fast_threshold(data["labels"][fold["calibration"]], all_scores[fold["calibration"]])
            heldout = fold["heldout"]
            scores[heldout] = all_scores[heldout]
            predictions[heldout] = all_scores[heldout] >= threshold
            reports.append(summary(data["labels"][heldout], all_scores[heldout], threshold))
            compatibility_metrics.append(frontend_metrics(data["labels"][heldout], all_scores[heldout], threshold))
            pipelines.append(pipeline); thresholds.append(threshold); training_times.append(elapsed)
    if not np.isfinite(scores).all():
        raise ValueError("Missing held-out speaker predictions.")
    pooled = summary(data["labels"], predictions.astype(float), .5)
    pooled["threshold_semantics"] = "each held-out fold uses its inner-calibration threshold; 0.5 is applied to pooled binary decisions only"
    speakers = {f"speaker_{int(group):02d}": summary(data["labels"][data["groups"] == group], predictions[data["groups"] == group].astype(float), .5)
                for group in np.unique(data["groups"])}
    objective = selection_score(reports, speakers, pooled)
    target = evaluation_cache/evaluation_key(bundle, proposal)
    target.mkdir(parents=True, exist_ok=False)
    artifact = target/"fold-models.joblib"
    with artifact.open("xb") as stream:
        joblib.dump({"pipelines": pipelines, "thresholds": thresholds, "columns": selected_columns,
                     "configuration": canonical(proposal), "scores": scores, "predictions": predictions,
                     "protocol": PROTOCOL, "bundle_sha256": file_digest(bundle/"bundle.json")}, stream)
    report = {"configuration": canonical(proposal), "metrics": pooled, "fold_metrics": reports,
              "evaluated_proposal": proposal, "bundle_sha256": file_digest(bundle/"bundle.json"),
              "heldout_speakers": speakers, "objective": objective, "training_runtime_s": float(sum(training_times)),
              "frontend_fold_metrics": compatibility_metrics,
              "fold_training_runtime_s": training_times, "diagnostics": diagnostics(data, predictions, pipelines, selected_columns),
              "artifact": str(artifact), "artifact_sha256": file_digest(artifact), "feature_count": len(selected_columns),
              "seed": plan["seed"], "protocol": PROTOCOL, "test_accessed": False,
              "main_validation_accessed": False, "independent_generalization_verified": False}
    save_json(target/"report.json", report)
    return {"cache_hit": False, "report": report}


def known_evaluations(bundle, evaluation_cache):
    """Reuse only identical frozen development protocols, not legacy validation fits."""
    identity = file_digest(bundle/"bundle.json")
    reports = []
    for path in sorted(evaluation_cache.glob("*/report.json")):
        saved = json.loads(path.read_text())
        if saved.get("bundle_sha256") == identity:
            checked = cached_evaluation(bundle, saved["evaluated_proposal"], evaluation_cache)
            if checked is not None:
                reports.append(checked)
    return reports


def final_candidate(bundle, proposal, fit_journal, export_path, provenance):
    """One final full-training fit and one existing-validation gate after selection."""
    proposal = validate_configuration(proposal)
    data = load_data(bundle, "train"); val = load_data(bundle, "validation")
    pipeline = pipeline_for(proposal, 42); selected_columns = columns(proposal["feature_set"])
    elapsed = fit_pipeline(pipeline, data, np.arange(len(data["labels"])), selected_columns, fit_journal, "final")
    with threadpool_limits(limits=1):
        scores = clip_scores(pipeline, val, selected_columns)
    threshold = fast_threshold(val["labels"], scores)
    report = {"metrics": summary(val["labels"], scores, threshold),
              "frontend_metrics": frontend_metrics(val["labels"], scores, threshold),
              "training_runtime_s": elapsed, "artifact": str(export_path), "test_accessed": False,
              "main_validation_accessed": True, "independent_generalization_verified": False,
              "threshold_selection": "one final validation calibration after training-only development selection"}
    with export_path.open("xb") as stream:
        joblib.dump({"pipeline": pipeline, "feature_settings": FEATURE_SETTINGS, "feature_indices": selected_columns,
                     "threshold": threshold, "model_version": "inthewild-luminal-research-v2", "seed": 42,
                     "configuration": canonical(proposal), "validation": report, "provenance": provenance,
                     "deployment_status": "research_only_not_deployed"}, stream)
    return report
