"""One-time evaluation safeguards and actual frozen-checkpoint inference mechanics."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf

from src import inthewild_final as final
from src.audio import decode_audio
from src.inference import predict_audio
from src.model import file_digest, load_model
from scripts.launch_final_demo import local_path, selected_path
from test_dataset import technical_dataset  # noqa: F401
from test_model import fitted  # noqa: F401


def wav(frequency=440, silent=False):
    buffer = BytesIO()
    samples = np.zeros(16000) if silent else .2 * np.sin(2*np.pi*frequency*np.arange(16000)/16000)
    sf.write(buffer, samples, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_metric_class_orientation():
    metrics = final.measured_metrics([0, 0, 0, 1, 1], [.1, .2, .9, .1, .9], .5)
    assert metrics["confusion_matrix_real_fake"] == [[2, 1], [1, 1]]
    assert metrics["precision_synthetic"] == .5
    assert metrics["recall_synthetic"] == .5
    assert metrics["false_positive_rate"] == pytest.approx(1/3)
    assert metrics["false_negative_rate"] == .5
    assert metrics["balanced_accuracy"] == pytest.approx(7/12)


def test_final_scores_once_then_replays_without_test_access(fitted, tmp_path, monkeypatch):
    _, _, model, _, _ = fitted
    experiment = tmp_path / "experiment"
    (experiment / "sealed").mkdir(parents=True)
    (experiment / "sealed/test.json").write_text("[]")
    rows, payloads = [], []
    for label in (0, 1):
        payload = wav(440 + 200*label)
        payloads.append(payload)
        rows.append(dict(path=f"{label}.wav", label=label, speaker=f"speaker-{label}",
                         utterance_id=f"id-{label}", sha256=hashlib.sha256(payload).hexdigest(),
                         pcm_sha256=hashlib.sha256(decode_audio(payload).samples.tobytes()).hexdigest()))
    run = tmp_path / "final-run"
    run.mkdir()
    artifact = load_model(model)
    artifact["provenance"] = {"fixture": "generated tones; no speech benchmark"}
    joblib.dump(artifact, model)
    freeze = dict(model_path=str(model), model_sha256=file_digest(model), experiment=str(experiment),
                  threshold=artifact["threshold"], feature_settings=artifact["feature_settings"],
                  protected_sha256={str(model): file_digest(model)})
    final.save(run / "freeze.json", freeze)
    monkeypatch.setattr(final, "audit_partition", lambda *args: rows)
    monkeypatch.setattr(final, "test_audio", lambda *args: iter(zip(rows, payloads)))
    original_score = final.segment_scores
    calls = []

    def score(*args):
        assert (experiment / "sealed/test_started.json").exists()
        calls.append(1)
        return original_score(*args)

    monkeypatch.setattr(final, "segment_scores", score)
    report = final.evaluate_once(run / "freeze.json", tmp_path)
    assert len(calls) == 2 and report["recordings_scored"] == 2
    assert report["metrics"]["threshold"] == artifact["threshold"]
    assert report["model_fits"] == 0 and report["threshold_tuned_on_test"] is False

    def forbidden(*args):
        pytest.fail("Successful replay must never reopen or rescore test data.")

    monkeypatch.setattr(final, "test_audio", forbidden)
    monkeypatch.setattr(final, "audit_partition", forbidden)
    monkeypatch.setattr(final, "segment_scores", forbidden)
    assert final.evaluate_once(run / "freeze.json", tmp_path) == report
    freeze["model_sha256"] = "another-candidate"
    (run / "freeze.json").write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="another frozen candidate"):
        final.evaluate_once(run / "freeze.json", tmp_path)


def test_partial_attempt_blocks_even_a_different_run_directory(tmp_path):
    experiment = tmp_path / "experiment"
    (experiment / "sealed").mkdir(parents=True)
    (experiment / "sealed/test_started.json").write_text("{}")
    for number in (1, 2):
        run = tmp_path / str(number)
        run.mkdir()
        final.save(run / "freeze.json", dict(experiment=str(experiment), model_sha256="frozen"))
        with pytest.raises(ValueError, match="no automatic retry"):
            final.evaluate_once(run / "freeze.json", tmp_path)


def test_partition_audit_rejects_speaker_leakage_and_test_cache(tmp_path):
    experiment = tmp_path / "experiment"
    (experiment / "sealed").mkdir(parents=True)
    (experiment / "features").mkdir()
    opt = [dict(path=f"opt-{i}", utterance_id=f"opt-{i}", sha256=f"opt-{i}", pcm_sha256=f"pcm-opt-{i}",
                speaker=split, label=i%2, split=split, seed=42)
           for i, split in enumerate(("train", "train", "validation", "validation"))]
    rows = [dict(path=f"test-{i}", utterance_id=f"test-{i}", sha256=f"test-{i}", pcm_sha256=f"pcm-test-{i}",
                 speaker="test-speaker", label=i%2, split="test", seed=42) for i in range(1800)]
    final.save(experiment / "optimization.json", opt)
    final.save(experiment / "sealed/test.json", rows)
    final.save(experiment / "preparation.json", {"partitions": {"test": final.distribution(rows)}})
    for row in opt:
        (experiment / "features" / final.feature_key(row)).touch()
    artifact = dict(splits_sha256=file_digest(experiment / "optimization.json"), seed=42,
                    train_groups=["train"], validation_groups=["validation"])
    assert final.audit_partition(experiment, artifact) == rows
    cached_test = experiment / "features" / final.feature_key(rows[0])
    cached_test.touch()
    with pytest.raises(ValueError, match="exclusively train/validation"):
        final.audit_partition(experiment, artifact)
    cached_test.unlink()
    rows[0]["speaker"] = "train"
    (experiment / "sealed/test.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="speaker overlaps"):
        final.audit_partition(experiment, artifact)


def test_test_shard_reader_checks_row_identity_and_bytes(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    payload = wav()
    items = [dict(path=f"{i}.wav", label=i, audio={"bytes": payload},
                  notes=json.dumps(dict(label="spoof" if i else "bona-fide",
                                        speaker=f"speaker-{i}", utterance_id=f"id-{i}")))
             for i in (0, 1)]
    pq.write_table(pa.Table.from_pylist(items), tmp_path / "fixture.parquet")
    monkeypatch.setattr(final, "verify_schema", lambda schema: None)
    row = dict(shard="fixture.parquet", row=1, path="1.wav", label=1,
               speaker="speaker-1", utterance_id="id-1", sha256=hashlib.sha256(payload).hexdigest())
    assert list(final.test_audio(tmp_path, [row])) == [(row, payload)]
    with pytest.raises(ValueError, match="identity/label mismatch"):
        list(final.test_audio(tmp_path, [dict(row, label=0)]))
    with pytest.raises(ValueError, match="audio changed"):
        list(final.test_audio(tmp_path, [dict(row, sha256="changed")]))


@pytest.fixture
def frozen_local():
    if not final.FREEZE.exists():
        pytest.skip("Actual frozen checkpoint is produced only by the authorized freeze command.")
    return json.loads(final.FREEZE.read_text())


def test_actual_selected_checkpoint_and_rollback_loading(frozen_local):
    selected = selected_path()
    rollback = selected_path(rollback=True)
    assert load_model(selected)["model_version"] == "inthewild-v1"
    assert load_model(rollback)["model_version"] == "baseline-v1"
    assert file_digest(final.ROOT / "models/baseline.joblib") == frozen_local["baseline_sha256"]
    assert file_digest(final.ROOT / "models/inthewild-v1-12000.joblib") == frozen_local["model_sha256"]
    assert selected != rollback


def test_actual_candidate_inference_and_invalid_unavailable(frozen_local, tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(selected_path()))
    result = predict_audio(wav(), "mechanics-tone.wav")
    assert result["status"] == "ok" and result["model_version"] == "inthewild-v1"
    assert result["threshold"] == frozen_local["threshold"]
    assert set(result) == {"status", "label", "synthetic_score", "threshold", "segments", "model_version", "message"}
    assert predict_audio(b"broken")["status"] == "invalid"
    assert predict_audio(wav(silent=True))["status"] == "invalid"
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(tmp_path / "missing.joblib"))
    assert predict_audio(wav())["status"] == "unavailable"
    artifact = load_model(selected_path())
    artifact["feature_settings"] = dict(artifact["feature_settings"], sample_rate=8000)
    bad = tmp_path / "incompatible.joblib"
    joblib.dump(artifact, bad)
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(bad))
    assert predict_audio(wav())["status"] == "unavailable"


def test_app_adapter_and_spectrogram_with_selected_model(frozen_local, monkeypatch):
    from ui.detector import run_detector
    from ui.viz import spectrogram_fig
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(selected_path()))
    result = run_detector(wav(), "mechanics-tone.wav")
    assert result["status"] == "ok" and result["model_version"] == "inthewild-v1"
    assert spectrogram_fig(decode_audio(wav())).data
    app = AppTest.from_file(str(final.ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception


def test_freeze_paths_from_another_machine_rebase_and_stay_hash_checked(tmp_path):
    other = "/Users/someone/elsewhere/Project-Spectra"
    freeze = dict(model_path=f"{other}/models/inthewild-final-frozen.joblib",
                  rollback_path=f"{other}/models/baseline-rollback-20261011.joblib",
                  model_sha256="not-the-real-digest", baseline_sha256="not-the-real-digest")
    assert local_path(freeze["rollback_path"], freeze) == final.ROOT / "models/baseline-rollback-20261011.joblib"
    (tmp_path / "freeze.json").write_text(json.dumps(freeze))
    if (final.ROOT / "models/inthewild-final-frozen.joblib").exists():
        with pytest.raises(ValueError, match="changed"):
            selected_path(freeze_path=tmp_path / "freeze.json")
