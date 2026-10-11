"""Offline export mechanics on technical tones, plus optional real local checkpoint verification."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from agents.experiment_runner import fit_candidate
from agents.luminal_client import LuminalClient
from scripts.compare_models import compare
from src import evaluation, promotion
from src.dataset import load_splits
from src.inference import predict_audio
from src.model import file_digest, load_model, score_rows, cached_features
from test_dataset import technical_dataset  # noqa: F401
from test_model import fitted  # noqa: F401


@pytest.fixture
def export_case(fitted, monkeypatch):
    root, splits, baseline, _, cache = fitted
    monkeypatch.setattr(LuminalClient, "complete", lambda *a, **k: pytest.fail("Paid requests forbidden"))
    config = {"role": "classifier", "feature_set": "all", "classifier": "logistic_regression",
              "parameters": {"C": 0.1, "class_weight": None}}
    result = fit_candidate(root, splits, baseline, cache, file_digest(splits), file_digest(baseline), config)
    # Mock event envelope for test tones only; metrics come from the actual fit.
    event = {"run_id": "technical-fixture", "event": "experiment", "status": "ok", "dry_run": False,
             "mode": "cloud", "decision": "accepted", "proposal": config, "role": "classifier",
             "round": 2, "timestamp": "technical-fixture", "seed": 42,
             "metrics": result["metrics"], "splits_sha256": file_digest(splits), "baseline_sha256": file_digest(baseline)}
    finish = dict(event, event="run_finished", best_proposal=config,
                  best_balanced_accuracy=result["metrics"]["balanced_accuracy"])
    history = splits.parent / "events.jsonl"
    history.write_text(json.dumps(event) + "\n" + json.dumps(finish) + "\n")
    return dict(root=root, splits=splits, baseline_path=baseline, history=history, run_id="technical-fixture",
                model_path=splits.parent / "winner.joblib", run_dir=splits.parent / "winner-run", cache=cache)


def test_export_preserves_inputs_scaler_and_real_inference(export_case, monkeypatch):
    args = export_case
    inputs = {key: file_digest(args[key]) for key in ("baseline_path", "splits", "history")}
    report = promotion.reproduce(**args)
    artifact = load_model(args["model_path"])
    assert artifact["model_version"] == "optimized-v1"
    assert artifact["label_mapping"] == {"real": 0, "fake": 1}
    assert artifact["pipeline"].named_steps["classifier"].C == 0.1
    assert artifact["validation"]["metrics"] == report["expected_metrics"]
    rows = load_splits(args["splits"])
    values, weights = [], []
    for row in rows:
        if row["split"] == "train":
            features, duration = cached_features(args["root"], row, args["cache"])
            values.extend(features)
            weights.extend(duration / duration.sum())
    np.testing.assert_allclose(artifact["pipeline"].named_steps["scaler"].mean_,
                               np.average(values, axis=0, weights=weights))
    row = next(row for row in rows if row["split"] == "validation")
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(args["model_path"]))
    result = predict_audio((args["root"] / row["path"]).read_bytes())
    assert result["status"] == "ok" and result["model_version"] == "optimized-v1"
    assert result["synthetic_score"] == pytest.approx(score_rows(artifact["pipeline"], args["root"], [row], args["cache"])[0])
    assert result["threshold"] == report["expected_metrics"]["threshold"]
    assert result["segments"] and 0 <= result["synthetic_score"] <= 1
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(args["baseline_path"]))
    assert predict_audio((args["root"] / row["path"]).read_bytes())["model_version"] == "baseline-v1"
    assert {key: file_digest(args[key]) for key in inputs} == inputs
    assert len(list(args["cache"].glob("*.npz"))) == sum(row["split"] != "test" for row in rows)
    monkeypatch.setattr(promotion, "fit_candidate_model", lambda *a: pytest.fail("Replay must not fit"))
    assert promotion.reproduce(**args) == report


def test_validation_mismatch_prevents_export(export_case):
    args = export_case
    lines = [json.loads(line) for line in args["history"].read_text().splitlines()]
    lines[0]["metrics"]["threshold"] += 0.01
    args["history"].write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    with pytest.raises(ValueError, match="mismatch in threshold"):
        promotion.reproduce(**args)
    assert not args["model_path"].exists()


def test_metadata_mismatch_and_baseline_overwrite_rejected(export_case):
    with pytest.raises(ValueError, match="separate path"):
        promotion.reproduce(**dict(export_case, model_path=export_case["baseline_path"]))
    promotion.reproduce(**export_case)
    artifact = load_model(export_case["model_path"])
    artifact["label_mapping"] = {"real": 1, "fake": 0}
    joblib.dump(artifact, export_case["model_path"])
    with pytest.raises(ValueError, match="metadata"):
        load_model(export_case["model_path"])


def test_comparison_replays_each_frozen_model_and_records_prior_test(export_case, monkeypatch):
    args = export_case
    promotion.reproduce(**args)
    baseline_run = args["splits"].parent / "prior-baseline-run"
    previous = evaluation.evaluate(args["root"], args["splits"], args["baseline_path"], baseline_run, args["cache"])
    kwargs = dict(root=args["root"], splits=args["splits"], baseline=args["baseline_path"], winner=args["model_path"],
                  baseline_run=baseline_run, winner_run=args["run_dir"], cache=args["cache"], history=args["history"])
    report = compare(**kwargs)
    assert report["baseline"] == previous["metrics"]
    assert any(item["path"] == str(baseline_run / "test_metrics.json")
               for item in report["history_before_comparison"]["prior_evaluation_records"])
    monkeypatch.setattr(evaluation, "score_rows", lambda *a: pytest.fail("Final test must replay"))
    assert compare(**kwargs) == report


def test_actual_saved_winner_loads_and_predicts_locally(monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    checkpoint = repository / "models/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e.joblib"
    splits = repository / "data/splits.csv"
    if not checkpoint.exists() or not splits.exists():
        pytest.skip("Optional local speech checkpoint/data are excluded from Git")
    artifact = load_model(checkpoint)
    assert artifact["validation"]["metrics"]["balanced_accuracy"] == 0.9568335941575378
    row = next(row for row in load_splits(splits, optimization_only=True) if row["split"] == "validation")
    root = repository / "data/deepfake-audio-detection"
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(checkpoint))
    result = predict_audio((root / row["path"]).read_bytes())
    assert result["status"] == "ok" and result["model_version"] == "optimized-v1"
    assert result["synthetic_score"] == pytest.approx(score_rows(artifact["pipeline"], root, [row], repository / "data/features")[0])
