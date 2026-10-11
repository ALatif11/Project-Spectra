"""No-network checks of the InTheWild adapter and paid-run guardrails."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from agents import inthewild_context as context, orchestrator
from agents.experiment_runner import Limits
from agents.proposals import static_proposal, feature_indices
from src.dataset import load_splits
from src.inthewild import LABELS, REPO, REVISION, summary
from src.model import cached_features, file_digest, load_model, segment_scores
from test_dataset import technical_dataset  # noqa: F401
from test_model import fitted  # noqa: F401


@pytest.fixture
def itw_context(fitted, tmp_path):
    root, old_splits, old_model, _, cache = fitted
    rows = load_splits(old_splits, optimization_only=True)
    experiment = tmp_path / "inthewild"
    experiment.mkdir()
    (experiment / "features").mkdir()
    artifact = load_model(old_model)
    scores, labels = [], []
    for row in rows:
        values, weights = cached_features(root, row, cache)
        row["speaker"] = row["group"]
        row["seed"] = int(row["seed"])
        row["label"] = int(row["label"])
        policy = json.dumps(artifact["feature_settings"], sort_keys=True)
        import hashlib
        key = hashlib.sha256((row["sha256"] + policy).encode()).hexdigest()
        np.savez_compressed(experiment / "features" / f"{key}.npz", features=values, weights=weights)
        if row["split"] == "validation":
            scores.append(float(np.average(segment_scores(artifact["pipeline"], values), weights=weights)))
            labels.append(row["label"])
    splits = experiment / "optimization.json"
    splits.write_text(json.dumps(rows))
    artifact.update(model_version="inthewild-v1", label_mapping=LABELS,
                    provenance={"repo": REPO, "revision": REVISION}, splits_sha256=file_digest(splits),
                    validation={"candidate": summary(labels, scores, artifact["threshold"])})
    model = experiment / "candidate.joblib"
    joblib.dump(artifact, model)
    return (root, splits, model, experiment / "features", file_digest(splits), file_digest(model))


def test_offline_reproduction_and_fit_access_only_manifest_and_cache(itw_context, tmp_path, monkeypatch):
    root, splits, model, cache, *_ = itw_context
    original_open = Path.open

    def restricted(path, *args, **kwargs):
        assert "sealed" not in path.parts
        assert path.suffix not in (".wav", ".flac")
        assert not path.is_relative_to(root)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", restricted)
    checked = context.baseline_check(*itw_context)
    assert checked["metrics"] == load_model(model)["validation"]["candidate"]
    proposal = static_proposal("features", 1)
    exported = tmp_path / "research.joblib"
    result = context.fit_candidate(*itw_context, proposal, exported, {"run_id": "test", "mode": "cloud"}, checked["cache_sha256"])
    assert result["metrics"]["n"] == checked["n_validation"]
    assert result["test_accessed"] is False and result["diagnostic_used"] is False
    saved = joblib.load(exported)
    assert saved["feature_indices"] == feature_indices("mfcc")
    assert saved["provenance"]["run_id"] == "test"
    arrays, weights = [], []
    for row in json.loads(splits.read_text()):
        if row["split"] == "train":
            values, duration = context.read_features(splits.parent, row)
            arrays.extend(values[:, feature_indices("mfcc")])
            weights.extend(duration/duration.sum())
    np.testing.assert_allclose(saved["pipeline"].named_steps["scaler"].mean_, np.average(arrays, axis=0, weights=weights))
    assert file_digest(model) == itw_context[-1]
    with pytest.raises(FileExistsError):
        context.fit_candidate(*itw_context, proposal, exported, {}, checked["cache_sha256"])


def test_missing_cache_fails_without_extraction(itw_context):
    next(itw_context[3].glob("*.npz")).unlink()
    with pytest.raises(FileNotFoundError):
        context.baseline_check(*itw_context)


@pytest.mark.parametrize("mutation", ["test", "speaker", "duplicate"])
def test_invalid_optimization_membership_refused(itw_context, mutation):
    args = list(itw_context)
    rows = json.loads(args[1].read_text())
    if mutation == "test":
        rows[0]["split"] = "test"
    elif mutation == "speaker":
        opposite = next(row for row in rows if row["split"] != rows[0]["split"])
        opposite["speaker"] = rows[0]["speaker"]
    else:
        rows.append(rows[0])
    args[1].write_text(json.dumps(rows))
    args[4] = file_digest(args[1])
    artifact = joblib.load(args[2])
    artifact["splits_sha256"] = args[4]
    joblib.dump(artifact, args[2])
    args[5] = file_digest(args[2])
    with pytest.raises(ValueError):
        context.baseline_check(*args)


def test_logistic_only_and_cache_change_guard(itw_context, tmp_path):
    checked = context.baseline_check(*itw_context)
    with pytest.raises(context.ProposalError):
        context.validate_logistic(static_proposal("classifier", 2))
    path = next(itw_context[3].glob("*.npz"))
    with np.load(path) as data:
        f, w = data["features"].copy(), data["weights"].copy()
    f[0, 0] += 1
    np.savez_compressed(path, features=f, weights=w)
    with pytest.raises(ValueError, match="cache changed"):
        context.fit_candidate(*itw_context, static_proposal("classifier", 1), tmp_path / "never.joblib", {}, checked["cache_sha256"])


@pytest.mark.parametrize("limits", [Limits(rounds=3), Limits(rounds=2, experiments=7), Limits(rounds=2, llm_calls=13)])
def test_itw_approval_hard_caps_before_calls(itw_context, tmp_path, limits):
    with pytest.raises(ValueError, match="approval caps"):
        orchestrator.run_experiments(inthewild=True, limits=limits, output=tmp_path / "history.jsonl",
                                    artifact_dir=tmp_path / "artifacts")
    assert not (tmp_path / "history.jsonl").exists()


def test_three_roles_receive_history_and_record_live_provenance(itw_context, tmp_path, monkeypatch):
    from types import SimpleNamespace
    checked = context.baseline_check(*itw_context)
    received = []

    class MockClient:
        config = SimpleNamespace(base_url="https://fixture.test/v1", model_id="fixture", max_output_tokens=1000, timeout_s=30)

        def propose_recorded(self, role, payload):
            received.append((role, payload))
            return {"proposal": static_proposal(role, 1), "provider_response": {"completion_tokens": 100}}

    def no_process(action, args, timeout):
        if action is context.baseline_check:
            result = checked
        elif action is context.fit_candidate:
            result = {"metrics": checked["metrics"], "validation_diagnostics": {}, "saved_artifact": str(args[7])}
        else:
            result = action(*args)
        return {"status": "ok", "result": result, "reason": None, "runtime_s": 0.01}

    monkeypatch.setattr(orchestrator, "bounded_call", no_process)
    root, splits, model, cache, *_ = itw_context
    output = tmp_path / "separate" / "experiments.jsonl"
    result = orchestrator.run_experiments(root=root, splits=splits, model_path=model, cache=cache, output=output,
        inthewild=True, client=MockClient(), prior_history=[{"source": "earlier_experiment", "metrics": checked["metrics"]}],
        limits=Limits(rounds=1, experiments=6, llm_calls=12), artifact_dir=tmp_path / "exports")
    assert result["llm_calls"] == 3 and result["model_experiments"] == 3
    assert [role for role, _ in received] == ["features", "classifier", "robustness"]
    for _, payload in received:
        assert payload["baseline_metrics"] == checked["metrics"]
        assert payload["experiments"][0]["source"] == "earlier_experiment"
        assert all(r["speaker"] not in json.dumps(payload) for r in json.loads(splits.read_text()))
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(e["event"] == "call_started" for e in events) == 3
    assert sum(e["event"] == "proposal" for e in events) == 3
    assert all(e["provider"]["automatic_retries"] == 0 for e in events)
    with pytest.raises(ValueError, match="separate new"):
        orchestrator.run_experiments(inthewild=True, limits=Limits(rounds=2, experiments=6, llm_calls=12),
                                    output=output, artifact_dir=tmp_path / "exports")
