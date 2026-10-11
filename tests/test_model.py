"""End-to-end mechanics on generated test tones; these are not speech metrics."""

import json
from io import BytesIO

import joblib
import numpy as np
import pytest
import soundfile as sf

from src import evaluation, model, training
from src.dataset import load_splits, prepare
from src.inference import predict_audio
from test_dataset import technical_dataset  # noqa: F401 -- shared pytest fixture


@pytest.fixture
def fitted(technical_dataset):
    root, splits = technical_dataset
    prepare(root, splits)
    model_path = splits.parent / "baseline.joblib"
    run_dir = splits.parent / "run"
    cache = splits.parent / "cache"
    training.train(root, splits, model_path, run_dir, cache)
    return root, splits, model_path, run_dir, cache


def test_saved_model_scaler_train_only_and_test_untouched(fitted):
    root, splits, model_path, run_dir, cache = fitted
    artifact = model.load_model(model_path)
    rows = load_splits(splits)
    assert artifact["feature_settings"] == model.FEATURE_SETTINGS
    assert artifact["pipeline"].classes_.tolist() == [0, 1]
    train = [row for row in rows if row["split"] == "train"]
    values, weights = [], []
    for row in train:
        features, duration = model.cached_features(root, row, cache)
        values.extend(features)
        weights.extend(duration / duration.sum())
    np.testing.assert_allclose(artifact["pipeline"].named_steps["scaler"].mean_,
                               np.average(values, axis=0, weights=weights))
    used = [row for row in rows if row["split"] != "test"]
    assert len(list(cache.glob("*.npz"))) == len(used)
    report = json.loads((run_dir / "validation_metrics.json").read_text())
    assert {clip["path"] for clip in report["clips"]} == {row["path"] for row in rows if row["split"] == "validation"}
    scores = np.array([clip["synthetic_score"] for clip in report["clips"]])
    labels = np.array([clip["label"] for clip in report["clips"]])
    assert artifact["threshold"] == training.select_threshold(labels, scores)
    assert report["metrics"] == training.metrics(labels, scores, artifact["threshold"])
    assert not (run_dir / "test_metrics.json").exists()


def test_final_evaluation_measures_once_and_replays(fitted, monkeypatch):
    root, splits, model_path, run_dir, cache = fitted
    report = evaluation.evaluate(root, splits, model_path, run_dir, cache)
    test_rows = [row for row in load_splits(splits) if row["split"] == "test"]
    assert report["metrics"]["n_clips"] == len(test_rows)
    assert report["metrics"]["threshold"] == model.load_model(model_path)["threshold"]
    assert report == json.loads((run_dir / "test_metrics.json").read_text())
    assert {clip["path"] for clip in report["clips"]} == {row["path"] for row in test_rows}

    def forbidden(*args):
        pytest.fail("Saved final test must not be rescored.")

    monkeypatch.setattr(evaluation, "score_rows", forbidden)
    assert evaluation.evaluate(root, splits, model_path, run_dir, cache) == report
    with pytest.raises(ValueError, match="Preserve existing"):
        training.train(root, splits, model_path, run_dir, cache)
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="already evaluated"):
        evaluation.evaluate(root, splits, model_path, run_dir, cache)


def test_exact_inference_contract_real_segment_scores(fitted, monkeypatch):
    root, splits, model_path, _, cache = fitted
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(model_path))
    row = load_splits(splits)[0]
    result = predict_audio((root / row["path"]).read_bytes(), row["path"])
    assert set(result) == {"status", "label", "synthetic_score", "threshold", "segments", "model_version", "message"}
    assert result["status"] == "ok"
    assert result["label"] in {"likely_human", "likely_synthetic", "inconclusive"}
    artifact = model.load_model(model_path)
    expected = model.score_rows(artifact["pipeline"], root, [row], cache)[0]
    assert result["synthetic_score"] == pytest.approx(expected)
    assert result["threshold"] == artifact["threshold"]
    assert result["segments"][0]["start_s"] == 0
    assert result["segments"][-1]["end_s"] == 2.5
    assert json.loads(json.dumps(result, allow_nan=False)) == result

    time = np.arange(84000) / 16000
    buffer = BytesIO()
    sf.write(buffer, 0.2 * np.sin(2 * np.pi * 440 * time), 16000, format="WAV")
    multi = predict_audio(buffer.getvalue())
    assert [(s["start_s"], s["end_s"]) for s in multi["segments"]] == [(0, 4), (4, 5.25)]
    assert multi["synthetic_score"] == pytest.approx(np.average(
        [s["synthetic_score"] for s in multi["segments"]], weights=[4, 1.25]))


@pytest.mark.parametrize("payload", [b"", b"broken", None])
def test_invalid_audio_without_model(payload, tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(tmp_path / "absent.joblib"))
    result = predict_audio(payload)
    assert result["status"] == "invalid"
    assert result["label"] == "inconclusive" and result["synthetic_score"] is None
    assert result["threshold"] is None and result["segments"] == []


def test_unavailable_and_silent_audio(fitted, monkeypatch):
    _, _, model_path, _, _ = fitted
    buffer = BytesIO()
    sf.write(buffer, np.zeros(16000), 16000, format="WAV")
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(model_path.parent / "absent.joblib"))
    assert predict_audio(buffer.getvalue())["status"] == "unavailable"
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(model_path))
    result = predict_audio(buffer.getvalue())
    assert result["status"] == "invalid" and result["segments"] == []
    assert result["label"] == "inconclusive" and result["synthetic_score"] is None


def test_incompatible_model_unavailable(fitted, monkeypatch):
    root, splits, model_path, _, _ = fitted
    artifact = model.load_model(model_path)
    artifact["feature_settings"] = dict(artifact["feature_settings"], sample_rate=8000)
    joblib.dump(artifact, model_path)
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(model_path))
    result = predict_audio((root / load_splits(splits)[0]["path"]).read_bytes())
    assert result["status"] == "unavailable" and result["synthetic_score"] is None


def test_modified_audio_rejects_even_cached_features(fitted):
    root, splits, _, _, cache = fitted
    row = next(row for row in load_splits(splits) if row["split"] == "train")
    path = root / row["path"]
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="changed since preparation"):
        model.cached_features(root, row, cache)


def test_changed_partitions_refuse_test(fitted):
    root, splits, model_path, run_dir, cache = fitted
    splits.write_text(splits.read_text().replace(",42,", ",43,"))
    with pytest.raises(ValueError, match="changed since training"):
        evaluation.evaluate(root, splits, model_path, run_dir, cache)
    assert not (run_dir / "test_started.json").exists()
