"""Freeze and evaluate the selected local InTheWild model exactly once.

No fitting, threshold selection, cloud calls, or database access. A partition-level
marker survives failures and changing the output directory cannot bypass it.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.audio import decode_audio
from src.inthewild import LIMITATIONS, distribution, optimization_rows, summary, verify_schema
from src.model import FEATURE_SETTINGS, file_digest, load_model, segment_scores, window_features

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "data/inthewild/experiment-12000-v1"
FINAL_RUN = ROOT / "runs/inthewild-final-20261011"
FREEZE = FINAL_RUN / "freeze.json"
POLICY = json.dumps(FEATURE_SETTINGS, sort_keys=True)


def save(path, value):
    """Publish a complete JSON file exclusively, without overwriting evidence."""
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def feature_key(row):
    return hashlib.sha256((row["sha256"] + POLICY).encode()).hexdigest() + ".npz"


def freeze_selected():
    """Check historical hashes and snapshot the already-selected model; no test read."""
    candidate = ROOT / "models/inthewild-v1-12000.joblib"
    baseline = ROOT / "models/baseline.joblib"
    verification = json.loads((EXPERIMENT / "verification.json").read_text())
    research = json.loads((ROOT / "runs/inthewild-research-preparation-20261011-v2/research-result.json").read_text())
    before = json.loads((ROOT / "runs/inthewild-research-preparation-20261011-v2/protected-before.json").read_text())
    if research["best_artifact"] != "models/inthewild-v1-12000.joblib" or research["test_accessed"]:
        raise ValueError("Research history disagrees with the selected, test-sealed candidate.")
    if file_digest(candidate) != verification["candidate_model_sha256"]:
        raise ValueError("Candidate changed since its saved validation verification.")
    for path in (candidate, baseline, EXPERIMENT / "optimization.json", EXPERIMENT / "comparison.json"):
        if file_digest(path) != before[str(path.relative_to(ROOT))]:
            raise ValueError(f"Protected input changed: {path}")
    artifact = load_model(candidate)
    classifier = artifact["pipeline"].named_steps["classifier"]
    if (classifier.C != 1 or classifier.class_weight is not None
            or classifier.random_state != 42 or classifier.solver != "lbfgs"
            or classifier.max_iter != 1000
            or artifact["validation"]["candidate"]["balanced_accuracy"] != 0.9292057374443738
            or artifact["threshold"] != 0.41431927302943794):
        raise ValueError("Checkpoint is not the selected validation candidate.")
    rows = optimization_rows(EXPERIMENT)
    expected = {feature_key(row) for row in rows}
    actual = {path.name for path in (EXPERIMENT / "features").glob("*.npz")}
    if expected != actual or len(rows) != 10200:
        raise ValueError("Feature cache contains missing or non-optimization entries.")
    if (EXPERIMENT / "sealed/test_started.json").exists() or (EXPERIMENT / "sealed/test_metrics.json").exists():
        raise ValueError("Final test already started; preserve and inspect its evidence.")
    FINAL_RUN.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / "models/inthewild-final-frozen.joblib"
    rollback = ROOT / "models/baseline-rollback-20261011.joblib"
    for source, target in ((candidate, frozen), (baseline, rollback)):
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        if file_digest(source) != file_digest(target):
            raise ValueError("Frozen checkpoint copy failed integrity verification.")
    paths = [candidate, baseline, frozen, rollback, EXPERIMENT / "optimization.json",
             EXPERIMENT / "comparison.json", EXPERIMENT / "preparation.json",
             ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py")),
             *[ROOT / f"src/{name}.py" for name in
               ("audio", "features", "spectrogram", "model", "inference", "inthewild", "inthewild_final")]]
    freeze = dict(frozen_at=datetime.now(timezone.utc).isoformat(),
                  model_path=str(frozen), rollback_path=str(rollback),
                  model_sha256=file_digest(frozen), baseline_sha256=file_digest(rollback),
                  protected_sha256={str(path): file_digest(path) for path in paths},
                  experiment=str(EXPERIMENT), feature_settings=FEATURE_SETTINGS,
                  threshold=float(artifact["threshold"]),
                  validation=artifact["validation"]["candidate"],
                  classifier_parameters=classifier.get_params(),
                  optimization_cache_count=len(actual), test_scored_before_freeze=False,
                  prior_evidence=[str(EXPERIMENT / "verification.json"),
                      str(ROOT / "runs/inthewild-luminal-20261011-v1/summary.json"),
                      str(ROOT / "runs/inthewild-research-preparation-20261011-v2/postrun-preservation.json")],
                  seal_audit_caveat="No historical digest of sealed/test.json was recorded. "
                      "Existing logs and optimization-only caches support no prior test scoring; "
                      "they are not an OS access audit. Preparation decoded all clips for PCM "
                      "identity checks; shard extraction also read batches containing unselected "
                      "audio, without extracting/scoring test features.")
    save(FREEZE, freeze)
    return freeze


def check_frozen(freeze):
    for path, expected in freeze["protected_sha256"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    artifact = load_model(Path(freeze["model_path"]))
    if artifact["threshold"] != freeze["threshold"] or artifact["feature_settings"] != freeze["feature_settings"]:
        raise ValueError("Frozen preprocessing/threshold mismatch.")
    return artifact


def audit_partition(experiment, artifact):
    """Inspect authorized test membership, hashes and cache absence before scoring."""
    optimization = optimization_rows(experiment)
    if artifact["splits_sha256"] != file_digest(experiment / "optimization.json"):
        raise ValueError("Optimization membership changed since candidate fitting.")
    for split, key in (("train", "train_groups"), ("validation", "validation_groups")):
        if sorted({r["speaker"] for r in optimization if r["split"] == split}) != artifact[key]:
            raise ValueError("Saved training/validation speakers disagree with membership.")
    test = json.loads((experiment / "sealed/test.json").read_text())
    if len(test) != 1800 or {r["label"] for r in test} != {0, 1}:
        raise ValueError("Expected the sealed 1,800-recording, two-class partition.")
    if any(r["split"] != "test" or r["seed"] != artifact["seed"] for r in test):
        raise ValueError("Invalid sealed membership/seed.")
    all_rows = optimization + test
    for key in ("path", "utterance_id", "sha256", "pcm_sha256"):
        if len({r[key] for r in all_rows}) != len(all_rows):
            raise ValueError(f"Duplicate/cross-partition identity: {key}")
    if {r["speaker"] for r in test} & {r["speaker"] for r in optimization}:
        raise ValueError("Test speaker overlaps development partitions.")
    preparation = json.loads((experiment / "preparation.json").read_text())
    if distribution(test) != preparation["partitions"]["test"]:
        raise ValueError("Sealed membership differs from saved preparation distribution.")
    cache_files = {p.name for p in (experiment / "features").glob("*.npz")}
    if cache_files != {feature_key(r) for r in optimization}:
        raise ValueError("Pre-evaluation cache is not exclusively train/validation.")
    return test


def test_audio(root, rows):
    """Yield only selected test clips, verifying published identity and raw bytes."""
    import pyarrow.parquet as pq
    from src.inthewild import metadata_rows
    wanted = {(r["shard"], r["row"]): r for r in rows}
    for shard in sorted({r["shard"] for r in rows}):
        parquet = pq.ParquetFile(root / shard)
        verify_schema(parquet.schema_arrow)
        offset = 0
        for batch in parquet.iter_batches(batch_size=16, columns=["path", "audio", "label", "notes"]):
            for index, item in enumerate(batch.to_pylist(), start=offset):
                row = wanted.get((shard, index))
                if row is None:
                    continue
                metadata = metadata_rows(pq_to_table(item), shard)[0]
                if any(metadata[key] != row[key] for key in ("path", "label", "speaker", "utterance_id")):
                    raise ValueError("Published test identity/label mismatch.")
                payload = item["audio"]["bytes"]
                if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    raise ValueError("Test audio changed since preparation.")
                yield row, payload
            offset += batch.num_rows


def pq_to_table(item):
    import pyarrow as pa
    return pa.Table.from_pylist([{k: item[k] for k in ("path", "label", "notes")}])


def measured_metrics(labels, scores, threshold):
    from sklearn.metrics import precision_score, recall_score
    result = summary(labels, scores, threshold)
    decisions = np.asarray(scores) >= threshold
    result.update(precision_synthetic=float(precision_score(labels, decisions, zero_division=0)),
                  recall_synthetic=float(recall_score(labels, decisions, zero_division=0)))
    return result


def evaluate_once(freeze_path=FREEZE, root=ROOT / "data/inthewild"):
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    experiment = Path(freeze["experiment"])
    sealed = experiment / "sealed"
    report_path, marker = sealed / "test_metrics.json", sealed / "test_started.json"
    identity = dict(model_sha256=freeze["model_sha256"], freeze_sha256=file_digest(freeze_path))
    # Successful replay never opens test membership/audio or invokes inference.
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if any(report[k] != v for k, v in identity.items()):
            raise ValueError("Final test belongs to another frozen candidate; no rescoring allowed.")
        return report
    if marker.exists():
        raise ValueError("Final test already started; no automatic retry or rescoring allowed.")
    artifact = check_frozen(freeze)
    rows = audit_partition(experiment, artifact)
    seal_hash = file_digest(sealed / "test.json")
    save(freeze_path.parent / "test_preflight.json", dict(identity,
         test_manifest_sha256=seal_hash, test_distribution=distribution(rows),
         speakers_disjoint=True, exact_encoded_and_pcm_duplicates_absent=True,
         test_features_cached_before_evaluation=False, historical_evidence=freeze.get("prior_evidence", []),
         caveat=freeze.get("seal_audit_caveat", "")))
    # Atomic exclusive marker on the PARTITION, before any test feature extraction.
    with marker.open("x") as stream:
        json.dump(dict(identity, test_manifest_sha256=seal_hash,
                       started_at=datetime.now(timezone.utc).isoformat()), stream)
        stream.flush()
        os.fsync(stream.fileno())
    clips = []
    with (sealed / "test_predictions.jsonl").open("x") as journal:
        for row, payload in test_audio(root, rows):
            recording = decode_audio(payload)
            if hashlib.sha256(recording.samples.tobytes()).hexdigest() != row["pcm_sha256"]:
                raise ValueError("Decoded test waveform differs from identity preparation.")
            features, weights, windows = window_features(recording)
            scores = segment_scores(artifact["pipeline"], features)
            clip = dict(path=row["path"], label=row["label"], speaker=row["speaker"],
                        utterance_id=row["utterance_id"], sha256=row["sha256"],
                        synthetic_score=float(np.average(scores, weights=weights)),
                        segments=[dict(start_s=w.start_s, end_s=w.end_s, synthetic_score=float(s))
                                  for w, s in zip(windows, scores)])
            journal.write(json.dumps(clip, allow_nan=False) + "\n")
            journal.flush()
            clips.append(clip)
            if len(clips) % 100 == 0:
                print(f"Final test: {len(clips)}/{len(rows)} recordings scored", flush=True)
    if len(clips) != len(rows) or {c["path"] for c in clips} != {r["path"] for r in rows}:
        raise ValueError("Test recordings missing or repeated; preserve partial journal.")
    check_frozen(freeze)
    if file_digest(sealed / "test.json") != seal_hash:
        raise ValueError("Sealed membership changed during evaluation.")
    labels = np.array([c["label"] for c in clips])
    scores = np.array([c["synthetic_score"] for c in clips])
    report = dict(identity, test_manifest_sha256=seal_hash, partition="local_sealed_test",
                  evaluated_at=datetime.now(timezone.utc).isoformat(),
                  model_version=artifact["model_version"], seed=artifact["seed"],
                  provenance=artifact["provenance"],
                  metrics=measured_metrics(labels, scores, artifact["threshold"]),
                  confusion_matrix_semantics="Rows: actual [human, synthetic]; columns: predicted [human, synthetic].",
                  decision_policy="Binary score >= frozen validation threshold; UI abstention band is excluded from these metrics.",
                  per_speaker={}, test_speakers=sorted({c["speaker"] for c in clips}),
                  limitations=LIMITATIONS, seal_audit_caveat=freeze.get("seal_audit_caveat", ""),
                  model_fits=0, luminal_calls=0, threshold_tuned_on_test=False,
                  recordings_scored=len(clips), predictions_sha256=file_digest(sealed / "test_predictions.jsonl"))
    for speaker in report["test_speakers"]:
        mask = np.array([c["speaker"] == speaker for c in clips])
        report["per_speaker"][speaker] = measured_metrics(labels[mask], scores[mask], artifact["threshold"])
    save(report_path, report)
    save(freeze_path.parent / "test_metrics.json", report)
    return report
