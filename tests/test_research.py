"""Expanded research lifecycle tests; generated fixtures are not speech evidence."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import jsonschema
from referencing import Registry, Resource

from agents import research_evaluation as evaluation, research_loop as loop
from agents.luminal_client import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, LuminalClient, LuminalConfig
from agents.research_budget import Budget, Pricing, ResearchPolicy
from agents.research_proposals import BASELINE, columns, fingerprint, validate_configuration, validate_response
from src.inthewild import summary
from src.model import file_digest
from src.training import select_threshold
from test_agents import chat_response, mock_sdk


@pytest.mark.parametrize("changed", [{"rounds": 7}, {"llm_calls": 61}, {"experiments": 37}, {"max_output_tokens": 2501},
    {"run_timeout_s": 901}, {"estimated_cost_usd": 5.01}, {"estimated_cost_usd": float("nan")},
    {"agents_per_round": 4}, {"patience": 3}, {"development_folds": 4}, {"train_timeout_s": 121},
    {"request_timeout_s": 121}, {"rounds": True}, {"max_output_tokens": True}])
def test_research_hard_limits(changed):
    with pytest.raises(ValueError):
        replace(ResearchPolicy(), **changed)


def test_larger_supported_budget_and_cost_reservation():
    policy = ResearchPolicy()
    assert (policy.rounds, policy.agents_per_round, policy.llm_calls, policy.experiments, policy.max_output_tokens) == (6, 3, 60, 36, 2500)
    now = [0.]
    budget = Budget(policy, Pricing(), clock=lambda: now[0])
    usage = {"prompt_tokens": 100, "cached_prompt_tokens": 50, "completion_tokens": 2500, "reasoning_tokens": 2000}
    reservation = budget.reserve_call([{"role": "user", "content": "test"}])
    cost = budget.settle_call(reservation, usage)
    expected = (50*.1425+50*.0029+2500*.57)/1e6
    assert cost["estimated_cost_usd"] == pytest.approx(expected)
    assert float(budget.cost) == pytest.approx(expected)  # Reasoning is not double billed.
    for _ in range(59):
        reserved = budget.reserve_call([{"role": "user", "content": "test"}])
        budget.settle_call(reserved, usage)
    with pytest.raises(ValueError, match="llm_call_limit"):
        budget.reserve_call([])
    now[0] = 901
    assert budget.remaining() == 0 and not budget.can_fit(1)


def test_spending_unknown_usage_and_fit_reservations():
    tiny = Budget(replace(ResearchPolicy(), estimated_cost_usd=.000001), Pricing())
    with pytest.raises(ValueError, match="estimated_cost_limit"):
        tiny.reserve_call([{"role": "user", "content": "test"}])
    assert tiny.calls == 0
    budget = Budget(replace(ResearchPolicy(), experiments=4), Pricing())
    reservation = budget.reserve_call([])
    assert budget.settle_call(reservation, {})["cost_basis"] == "reserved_upper_estimate"
    assert budget.uncertain_requests == 1 and float(budget.cost) > 0
    assert budget.can_fit(3)
    budget.account_fits([{"event": "fit_started"}, {"event": "fit_completed"}, {"event": "fit_started"}])
    assert budget.fits == 2 and budget.fits_completed == 1
    assert not budget.can_fit(3) and budget.can_fit(1)


@pytest.mark.parametrize("changes", [{"classifier": "cnn"}, {"feature_set": "raw_audio"}, {"shell": "anything"},
                                     {"parameters": {"C": True, "class_weight": None}},
                                     {"parameters": {"C": 1000, "class_weight": None}}])
def test_research_proposals_reject_executable_or_excessive_values(changes):
    with pytest.raises(ValueError):
        validate_configuration({**BASELINE, **changes})


def test_proposal_hypotheses_canonical_deduplication_and_family_columns():
    assert fingerprint(BASELINE) == fingerprint({**BASELINE, "role": "robustness", "parameters": {"C": 1, "class_weight": None}})
    assert len(columns("mfcc_mean")) == 20 and len(columns("spectral_shape")) == 8
    value = {"hypothesis": "Reduce false positives by removing recording-level feature shortcuts.", "proposal": {**BASELINE, "role": "features", "feature_set": "no_energy"}}
    assert validate_response(json.dumps(value), "features")["proposal"] == value["proposal"]
    with pytest.raises(ValueError):
        validate_response(json.dumps({**value, "hypothesis": "```python arbitrary```"}), "features")
    forest = {"role": "classifier", "feature_set": "all", "classifier": "random_forest", "parameters": {
        "n_estimators": 100, "max_depth": 6, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced"}}
    validate_configuration(forest)
    with pytest.raises(ValueError):
        validate_configuration({**forest, "parameters": {**forest["parameters"], "max_depth": 10}})


def test_sdk_research_hypothesis_usage_and_2500_tokens_without_retries(monkeypatch):
    captured = []
    proposal = {"hypothesis": "Stronger regularization may reduce variation across held-out speaker groups.",
                "proposal": {**BASELINE, "parameters": {"C": .1, "class_weight": None}}}

    def response(request):
        captured.append(json.loads(request.content))
        import httpx
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(proposal)}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 400, "completion_tokens": 200, "total_tokens": 600,
                    "prompt_tokens_details": {"cached_tokens": 100}, "completion_tokens_details": {"reasoning_tokens": 100}}})

    settings = mock_sdk(monkeypatch, response)
    client = LuminalClient(LuminalConfig("private-fixture-key", "https://configured.test/v1", DEFAULT_MODEL_ID, 5, 2500))
    result = client.propose_research("classifier", {"metrics": {"balanced_accuracy": .9}})
    assert result["status"] == "ok" and result["hypothesis"] == proposal["hypothesis"]
    assert result["provider_response"]["cached_prompt_tokens"] == 100
    assert len(captured) == 1 and captured[0]["max_tokens"] == 2500 and settings[0]["max_retries"] == 0
    assert "private-fixture-key" not in json.dumps(captured)


def test_research_cloud_failure_retains_usage_without_static_fallback(monkeypatch):
    import httpx
    mock_sdk(monkeypatch, lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 2500, "total_tokens": 2700,
                  "completion_tokens_details": {"reasoning_tokens": 2500}}}))
    client = LuminalClient(LuminalConfig("fixture-key", "https://configured.test/v1", DEFAULT_MODEL_ID, 5, 2500))
    result = client.propose_research("features", {})
    assert result["status"] == "failed" and result["proposal"] is None
    assert result["provider_response"]["completion_tokens"] == 2500


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/"bundle"; root.mkdir()
    rng = np.random.default_rng(42)
    hashes = {}
    for partition, speakers in (("train", 18), ("validation", 4)):
        labels = np.tile([0, 1, 0, 1, 0, 1], speakers)
        values = rng.normal(size=(len(labels)*2, 52)).astype(np.float32)
        values[:, :5] += np.repeat(labels, 2)[:, None]*.8
        fields = {"values": values, "durations": np.tile([2., 1.], len(labels)), "labels": labels,
                  "groups": np.repeat(np.arange(speakers), 6), "offsets": np.arange(0, len(values)+1, 2)}
        for key, value in fields.items():
            path = root/f"{partition}-{key}.npy"; np.save(path, value, allow_pickle=False); hashes[path.name] = file_digest(path)
    data = evaluation.load_data(root, "train")
    evaluation.save_json(root/"folds.json", evaluation.development_plan(data["labels"], data["groups"], 42))
    evaluation.save_json(root/"bundle.json", {"array_hashes": hashes, "folds_sha256": file_digest(root/"folds.json"), "identity": {"fixture": True}})
    return root


def test_development_calibration_groups_and_fast_threshold_match(bundle):
    data = evaluation.load_data(bundle, "train")
    plan = json.loads((bundle/"folds.json").read_text())
    all_heldout = []
    for fold in plan["folds"]:
        groups = [set(data["groups"][fold[key]]) for key in ("fit", "calibration", "heldout")]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
        all_heldout.extend(fold["heldout"])
    assert sorted(all_heldout) == list(range(len(data["labels"])))
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = np.tile([0, 1], 12); scores = np.round(rng.uniform(size=24), 1)
        assert evaluation.fast_threshold(y, scores) == select_threshold(y, scores)


def test_actual_three_fold_fits_train_only_cache_and_scaler_fit_membership(bundle, tmp_path, monkeypatch):
    original_load = evaluation.load_data
    monkeypatch.setattr(evaluation, "load_data", lambda path, partition: original_load(path, partition) if partition == "train" else pytest.fail("Main validation forbidden during selection"))
    journal = tmp_path/"fits.jsonl"; cache = tmp_path/"eval"
    result = evaluation.evaluate_configuration(bundle, BASELINE, cache, journal)
    report = result["report"]
    assert result["cache_hit"] is False and report["main_validation_accessed"] is False
    events = loop.fit_events(journal)
    assert sum(e["event"] == "fit_started" for e in events) == 3
    assert sum(e["event"] == "fit_completed" for e in events) == 3
    artifact = joblib.load(report["artifact"]); data = original_load(bundle, "train")
    plan = json.loads((bundle/"folds.json").read_text())
    for pipeline, fold in zip(artifact["pipelines"], plan["folds"]):
        lengths = np.diff(data["offsets"])
        mask = np.repeat(np.isin(np.arange(len(data["labels"])), fold["fit"]), lengths)
        weights = data["durations"]/np.repeat(np.add.reduceat(data["durations"], data["offsets"][:-1]), lengths)
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, np.average(data["values"][mask], axis=0, weights=weights[mask]))
    assert evaluation.evaluate_configuration(bundle, BASELINE, cache, journal)["cache_hit"] is True
    assert len(loop.fit_events(journal)) == len(events)
    Path(report["artifact"]).write_bytes(b"changed-fixture")
    with pytest.raises(ValueError, match="changed"):
        evaluation.cached_evaluation(bundle, BASELINE, cache)


@pytest.fixture
def loop_config(bundle, tmp_path):
    cache = tmp_path/"evaluations"
    report = evaluation.evaluate_configuration(bundle, BASELINE, cache, tmp_path/"baseline-journal.jsonl")["report"]
    run = tmp_path/"run"; run.mkdir()
    return {"run_dir": str(run), "bundle": str(bundle), "evaluation_cache": str(cache), "run_id": "fixture-parent",
            "policy": ResearchPolicy(), "pricing": Pricing(), "protected_hashes": {}, "source_hashes": {},
            "model": "preserved-fixture.joblib", "model_hash": "fixture", "splits_hash": "fixture",
            "baseline_validation": summary([0, 0, 1, 1], [.1, .2, .8, .9], .5),
            "history_overview": {"fixture": "no actual speech results"}, "baseline": report}


class FixtureClient:
    config = SimpleNamespace(base_url=DEFAULT_BASE_URL, model_id=DEFAULT_MODEL_ID, max_output_tokens=2500)

    def __init__(self, *, fails=False, changes=False):
        self.received = []; self.fails = fails; self.changes = changes

    def propose_research(self, role, summary):
        self.received.append((role, summary))
        if self.fails:
            return {"status": "failed", "reason": "fixture-empty-response", "provider_response": {"prompt_tokens": 1000, "completion_tokens": 2500}, "hypothesis": None, "proposal": None}
        p = {**BASELINE, "role": role}
        if self.changes:
            if role == "features":
                p["feature_set"] = "mfcc" if summary["round"] == 1 else "spectral"
            else:
                p["parameters"] = {"C": .1*summary["round"] if role == "classifier" else .3*summary["round"], "class_weight": None}
        return {"status": "ok", "proposal": p, "hypothesis": "Fixture hypothesis: this controlled test changes a bounded configuration.",
                "reason": None, "provider_response": {"prompt_tokens": 1000, "cached_prompt_tokens": 500, "completion_tokens": 200}}


def synchronous(monkeypatch):
    monkeypatch.setattr(loop, "bounded_call", lambda action, args, timeout: {"status": "ok", "result": action(*args), "reason": None, "runtime_s": 0.01})


def test_two_complete_plateau_rounds_deduplicate_all_roles_and_preserve_frontend_schema(loop_config, monkeypatch):
    synchronous(monkeypatch); client = FixtureClient()
    schema_path = Path("agents/event.schema.json"); before = schema_path.read_bytes()
    result = loop.run_research(loop_config, client=client)
    assert result["reason"] == "early_stop_two_complete_rounds_without_meaningful_improvement"
    assert result["completed_rounds"] == 2 and result["budget"]["api_calls"] == 6 and result["budget"]["ml_fit_attempts"] == 0
    assert result["final_validation"] is None and schema_path.read_bytes() == before
    schema = json.loads(before); proposal_schema = json.loads(Path("agents/proposal.schema.json").read_text())
    uri = "https://fixture.test/event.schema.json"
    registry = Registry().with_resource("https://fixture.test/proposal.schema.json", Resource.from_contents(proposal_schema))
    validator = jsonschema.Draft202012Validator({**schema, "$id": uri}, registry=registry)
    projected = [json.loads(l) for l in (Path(loop_config["run_dir"])/"frontend-events.jsonl").read_text().splitlines()]
    for event in projected:
        validator.validate(event)
    assert len({e["run_id"] for e in projected}) == 2  # No global counts are clipped.
    assert client.received[3][1]["recent_measured_outcomes"]  # Prior round fed back to every role.
    with pytest.raises(ValueError, match="already started"):
        loop.run_research(loop_config, client=client)


def test_failed_cloud_responses_never_substitute_proposals(loop_config, monkeypatch):
    synchronous(monkeypatch); client = FixtureClient(fails=True)
    result = loop.run_research(loop_config, client=client)
    assert result["budget"]["api_calls"] == 6 and result["budget"]["ml_fit_attempts"] == 0
    assert result["status"] == "partial_failure" and result["completed_configurations"] == 0
    assert not list(Path(loop_config["run_dir"]).glob("configuration-*-fits.jsonl"))


def test_spending_cap_stops_before_cloud_or_candidate_fit(loop_config, monkeypatch):
    synchronous(monkeypatch); client = FixtureClient()
    loop_config["policy"] = replace(ResearchPolicy(), estimated_cost_usd=.000001)
    result = loop.run_research(loop_config, client=client)
    assert result["reason"] == "estimated_cost_limit" and not client.received
    assert result["budget"]["api_calls"] == 0 and result["budget"]["ml_fit_attempts"] == 0


def test_improvement_feedback_actual_fit_counts_and_final_gate_only_after_selection(loop_config, monkeypatch):
    client = FixtureClient(changes=True); counter = [0]; final_calls = []
    base = loop_config["baseline"]

    def fake_evaluation(bundle, proposal, cache, journal, max_fits):
        counter[0] += 1
        for fold in range(3):
            evaluation.journal(journal, "fit_started", fold=fold)
            evaluation.journal(journal, "fit_completed", fold=fold)
        return {"cache_hit": False, "report": {**base, "objective": {**base["objective"], "score": base["objective"]["score"]+.004*counter[0]}}}

    def fake_final(bundle, proposal, journal, export, provenance):
        final_calls.append(len(client.received))
        evaluation.journal(journal, "fit_started", fold="final"); evaluation.journal(journal, "fit_completed", fold="final")
        return {"artifact": str(export), "metrics": loop_config["baseline_validation"],
                "frontend_metrics": base["frontend_fold_metrics"][0], "training_runtime_s": .01}

    monkeypatch.setattr(loop, "evaluate_configuration", fake_evaluation)
    monkeypatch.setattr(loop, "final_candidate", fake_final)
    loop_config["policy"] = replace(ResearchPolicy(), rounds=2)
    synchronous(monkeypatch)
    result = loop.run_research(loop_config, client=client)
    assert result["budget"]["ml_fit_attempts"] == 19 and result["completed_configurations"] == 6
    assert result["budget"]["ml_fits_completed"] == 19 and final_calls == [6]
    assert client.received[3][1]["best_objective"]["score"] > base["objective"]["score"]
    assert all("main_validation_accessed" not in json.dumps(s) for _, s in client.received)


def test_research_prepare_default_does_not_benchmark_or_start_live(tmp_path, monkeypatch, capsys):
    from scripts import optimize_inthewild, prepare_research
    calls = []

    def fake_prepare(run, profile, benchmark):
        calls.append(benchmark)
        return {k: "test-only" for k in ("run_id", "policy", "pricing", "baseline_validation", "estimate", "api_calls", "exact_live_command")}

    monkeypatch.setattr(prepare_research, "prepare", fake_prepare)
    monkeypatch.setattr(LuminalClient, "complete", lambda *a, **k: pytest.fail("Paid calls forbidden"))
    monkeypatch.setattr("sys.argv", ["optimize_inthewild", "--research", "--run-dir", str(tmp_path)])
    optimize_inthewild.main()
    assert calls == [False]


def test_partial_round_fit_cap_reserves_final_fit_without_counting_as_plateau(loop_config, monkeypatch):
    client = FixtureClient(changes=True); base = loop_config["baseline"]

    def fake_eval(bundle, proposal, cache, journal, max_fits):
        for i in range(3):
            evaluation.journal(journal, "fit_started", fold=i); evaluation.journal(journal, "fit_completed", fold=i)
        return {"cache_hit": False, "report": {**base, "objective": {**base["objective"], "score": base["objective"]["score"]+.01}}}

    def fake_final(bundle, proposal, journal, export, provenance):
        evaluation.journal(journal, "fit_started", fold="final"); evaluation.journal(journal, "fit_completed", fold="final")
        return {"artifact": str(export), "metrics": loop_config["baseline_validation"], "frontend_metrics": base["frontend_fold_metrics"][0], "training_runtime_s": .01}

    monkeypatch.setattr(loop, "evaluate_configuration", fake_eval); monkeypatch.setattr(loop, "final_candidate", fake_final)
    loop_config["policy"] = replace(ResearchPolicy(), experiments=8)
    synchronous(monkeypatch)
    result = loop.run_research(loop_config, client=client)
    assert result["reason"] == "fit_budget_reserved_for_final_gate" and result["completed_rounds"] == 0
    assert result["budget"]["api_calls"] == 2 and result["budget"]["ml_fit_attempts"] == 7


def test_provider_parameter_rejection_stops_without_paid_fallback(loop_config, monkeypatch):
    client = FixtureClient()
    monkeypatch.setattr(client, "propose_research", lambda *a: {"status": "failed", "reason": "Luminal HTTP 422. No automatic retry.",
                        "provider_response": {}, "proposal": None, "hypothesis": None})
    synchronous(monkeypatch)
    result = loop.run_research(loop_config, client=client)
    assert result["budget"]["api_calls"] == 1 and result["budget"]["ml_fit_attempts"] == 0
    assert result["reason"] == "provider_configuration_rejected_no_retry"


def test_partial_failed_fold_workers_count_only_actual_started_fits(loop_config, monkeypatch):
    client = FixtureClient(changes=True)

    def bounded(action, args, timeout):
        if action is loop.evaluate_configuration:
            evaluation.journal(args[3], "fit_started", fold=1)
            return {"status": "timeout", "reason": "operation_timeout", "result": None, "runtime_s": .01}
        return {"status": "ok", "result": action(*args), "reason": None, "runtime_s": .01}

    monkeypatch.setattr(loop, "bounded_call", bounded)
    loop_config["policy"] = replace(ResearchPolicy(), rounds=1)
    result = loop.run_research(loop_config, client=client)
    assert result["budget"]["ml_fit_attempts"] == 3 and result["budget"]["ml_fits_completed"] == 0
    assert result["completed_configurations"] == 0 and result["status"] == "partial_failure"
