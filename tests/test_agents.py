"""Agent safety/orchestration tests; technical tone fixtures are not speech evidence."""

from dataclasses import replace
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import httpx
import openai

import numpy as np
import pytest

from agents import experiment_runner as runner, orchestrator
from agents import configuration
from agents.luminal_client import LuminalClient, LuminalConfig, LuminalError
from agents.proposals import ProposalError, feature_indices, parse_json, static_proposal, validate_proposal
from src.dataset import load_splits
from src.model import cached_features, file_digest, load_model
from test_dataset import technical_dataset  # noqa: F401
from test_model import fitted  # noqa: F401


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    # Tests must never read a developer's real .env or send a paid request.
    monkeypatch.setattr(configuration, "REPOSITORY_ROOT", tmp_path)
    for key in ("LUMINAL_API_KEY", "LUMINAL_BASE_URL", "LUMINAL_MODEL_ID", "LUMINAL_TIMEOUT_S",
                "MAX_AGENT_ROUNDS", "MAX_LLM_CALLS", "MAX_MODEL_EXPERIMENTS", "MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def config():
    return LuminalConfig("private-test-key", "https://configured.test/v1", "configured-test-model", 5)


def mock_sdk(monkeypatch, handler):
    original = openai.OpenAI
    settings = []

    def factory(**kwargs):
        settings.append({key: value for key, value in kwargs.items() if key not in ("api_key", "http_client")})
        assert kwargs["http_client"].follow_redirects is False
        kwargs["http_client"].close()
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        return original(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", factory)
    return settings


def chat_response(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.mark.parametrize("role", ["features", "classifier", "robustness"])
def test_all_static_rounds_match_role_allowlist(role):
    for round_number in range(1, 4):
        proposal = static_proposal(role, round_number)
        assert validate_proposal(parse_json(json.dumps(proposal)), role) == proposal
    assert len(feature_indices("all")) == 52
    assert len(feature_indices("mfcc")) == 40
    assert len(feature_indices("spectral")) == 12
    assert len(feature_indices("no_energy")) == 48


@pytest.mark.parametrize("change", [
    {"python": "__import__('os').system('anything')"}, {"seed": 123}, {"role": "robustness"},
    {"feature_set": "unapproved"}, {"classifier": "cnn"},
    {"parameters": {"C": True, "class_weight": None}},
    {"parameters": {"C": float("nan"), "class_weight": None}},
    {"parameters": {"C": 1000000, "class_weight": None}},
    {"parameters": {"C": 10 ** 500, "class_weight": None}},
    {"parameters": {"C": 1, "class_weight": None, "shell": "anything"}},
])
def test_rejects_arbitrary_code_seed_changes_and_unbounded_settings(change):
    with pytest.raises(ProposalError):
        validate_proposal(dict(static_proposal("classifier", 1), **change), "classifier")


@pytest.mark.parametrize("text", ['{"C":1,"C":2}', '{"C":NaN}', '```json\n{}\n```', '[]' + ' ' * 17000])
def test_json_parser_rejects_ambiguous_or_non_json_responses(text):
    with pytest.raises(ProposalError):
        parse_json(text)


def test_role_constraints_and_random_forest_bounds():
    feature = static_proposal("features", 1)
    feature["parameters"]["C"] = 2
    robust = static_proposal("robustness", 1)
    robust["feature_set"] = "all"
    forest = static_proposal("classifier", 2)
    forest["parameters"]["n_estimators"] = 1000
    for proposal in (feature, robust, forest):
        with pytest.raises(ProposalError):
            validate_proposal(proposal, proposal["role"])


def test_configuration_validates_provider_and_hides_key(config):
    with pytest.raises(LuminalError, match="Configure LUMINAL"):
        LuminalConfig.from_env()
    assert "private-test-key" not in repr(config)
    for changes in ({"base_url": "http://configured.test"}, {"base_url": "https://user:secret@configured.test"},
                    {"base_url": "https://configured.test?key=secret"}, {"model_id": "model\nname"},
                    {"timeout_s": float("inf")}, {"max_output_tokens": 1001}, {"max_output_tokens": True}):
        with pytest.raises(LuminalError):
            replace(config, **changes)


def test_root_env_loaded_from_another_directory_and_host_key_preserved(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LUMINAL_API_KEY=local-test-key\nMAX_OUTPUT_TOKENS=80\nMAX_AGENT_ROUNDS=2\nMAX_LLM_CALLS=4\nMAX_MODEL_EXPERIMENTS=5\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("LUMINAL_API_KEY=wrong-test-key\n")
    monkeypatch.chdir(elsewhere)
    config = LuminalConfig.from_env()
    assert config.api_key == "local-test-key" and config.max_output_tokens == 80
    assert config.base_url == "https://flash.luminal.cloud/v1"
    assert config.model_id == "deepseek-ai/DeepSeek-V4.1-Flash"
    limits = runner.Limits.from_env()
    assert (limits.rounds, limits.llm_calls, limits.experiments) == (2, 4, 5)
    monkeypatch.setenv("LUMINAL_API_KEY", "host-test-key")
    assert LuminalConfig.from_env().api_key == "host-test-key"


@pytest.mark.parametrize("name,value", [("MAX_AGENT_ROUNDS", "4"), ("MAX_LLM_CALLS", "21"),
    ("MAX_MODEL_EXPERIMENTS", "13"), ("MAX_OUTPUT_TOKENS", "1001"), ("MAX_OUTPUT_TOKENS", "secret")])
def test_environment_cannot_raise_hard_caps(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("LUMINAL_API_KEY", "test-key")
    with pytest.raises((ValueError, LuminalError)) as error:
        LuminalConfig.from_env() if name == "MAX_OUTPUT_TOKENS" else runner.Limits.from_env()
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("role", ["features", "classifier", "robustness"])
@pytest.mark.parametrize("tokens", [1000, 80])
def test_sdk_uses_configured_url_model_and_token_cap_for_each_role(config, monkeypatch, role, tokens):
    requests = []

    def respond(request):
        requests.append(request)
        return chat_response(json.dumps(static_proposal(role, 1)))

    settings = mock_sdk(monkeypatch, respond)
    config = replace(config, max_output_tokens=tokens)
    assert LuminalClient(config).propose(role, {"n_train": 10}) == static_proposal(role, 1)
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://configured.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer private-test-key"
    assert payload["model"] == config.model_id and payload["max_tokens"] == tokens
    assert settings[0]["max_retries"] == 0 and settings[0]["timeout"] == 5
    assert "private-test-key" not in json.dumps(payload)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422, 429, 500, 302])
def test_provider_http_errors_are_actionable_redacted_and_never_retried(config, monkeypatch, code):
    requests = []

    def failed(request):
        requests.append(request)
        return httpx.Response(code, headers={"Location": "https://other.test"},
                              json={"error": {"message": "private-test-key private response"}})

    mock_sdk(monkeypatch, failed)
    with pytest.raises(LuminalError, match=f"HTTP {code}") as error:
        LuminalClient(config).propose("features", {})
    assert "private" not in str(error.value) and len(requests) == 1
    if code in (400, 422):
        assert "max_tokens" in str(error.value) and "No parameter fallback" in str(error.value)


@pytest.mark.parametrize("response", [httpx.Response(200, content=b"invalid"),
    httpx.Response(200, json={"choices": []}), chat_response("x" * 65537), chat_response(None),
    chat_response('{"python":"private response"}')])
def test_no_fabricated_response_or_raw_payload_on_failure(config, monkeypatch, response):
    mock_sdk(monkeypatch, lambda request: response)
    with pytest.raises((LuminalError, ProposalError)) as error:
        LuminalClient(config).propose("features", {})
    assert "private response" not in str(error.value)


@pytest.mark.parametrize("kind", [httpx.ConnectError, httpx.ReadTimeout])
def test_connection_failure_and_timeout_do_not_retry_or_leak(config, monkeypatch, kind):
    requests = []

    def failed(request):
        requests.append(request)
        raise kind("private-test-key", request=request)

    mock_sdk(monkeypatch, failed)
    with pytest.raises(LuminalError) as error:
        LuminalClient(config).complete([{"role": "user", "content": "OK"}])
    assert len(requests) == 1 and "private" not in str(error.value)


@pytest.mark.parametrize("code", [200, 400, 429, 500])
def test_connectivity_script_makes_exactly_one_small_request(config, monkeypatch, capsys, code):
    from scripts import test_luminal
    monkeypatch.setattr(test_luminal.LuminalConfig, "from_env", lambda: config)
    requests = []

    def respond(request):
        requests.append(request)
        return chat_response("OK") if code == 200 else httpx.Response(code, json={"error": "private response"})

    mock_sdk(monkeypatch, respond)
    assert test_luminal.main([]) == (0 if code == 200 else 1)
    assert len(requests) == 1 and json.loads(requests[0].content)["max_tokens"] == 16
    captured = capsys.readouterr()
    assert "private" not in captured.out + captured.err


def test_connectivity_dry_run_never_constructs_sdk(config, monkeypatch, capsys):
    from scripts import test_luminal
    monkeypatch.setattr(test_luminal.LuminalConfig, "from_env", lambda: config)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("Dry run attempted cloud access"))
    assert test_luminal.main(["--dry-run"]) == 0
    assert "zero API requests" in capsys.readouterr().out


@pytest.mark.parametrize("requested,expected", [(1, 0), (3, 1)])
def test_cli_can_lower_but_not_raise_root_environment_budget(monkeypatch, capsys, requested, expected):
    from agents import __main__ as cli
    monkeypatch.setenv("MAX_AGENT_ROUNDS", "2")
    monkeypatch.setattr("sys.argv", ["agents", "--dry-run", "--rounds", str(requested)])
    captured = []

    def run(**kwargs):
        captured.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(cli, "run_experiments", run)
    assert cli.main() == expected
    assert len(captured) == (1 if expected == 0 else 0)
    if captured:
        assert captured[0]["dry_run"] and captured[0]["limits"].rounds == 1


def test_real_candidate_uses_only_fixed_training_scaler_and_validation(fitted, monkeypatch):
    root, splits, model_path, _, cache = fitted
    rows = load_splits(splits)
    for row in rows:
        if row["split"] == "test":
            (root / row["path"]).unlink()  # Any final-test audio access must fail this test.
    fitted_pipelines = []
    original = runner.Pipeline

    class CapturingPipeline(original):
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            fitted_pipelines.append(self)
            return result

    monkeypatch.setattr(runner, "Pipeline", CapturingPipeline)
    proposal = static_proposal("features", 1)
    common = (root, splits, model_path, cache, file_digest(splits), file_digest(model_path))
    result = runner.fit_candidate(*common, proposal)
    assert result["metrics"]["n_clips"] == sum(row["split"] == "validation" for row in rows)
    assert result["seed"] == load_model(model_path)["seed"]
    values, weights = [], []
    for row in rows:
        if row["split"] == "train":
            features, duration = cached_features(root, row, cache)
            values.extend(features[:, feature_indices("mfcc")])
            weights.extend(duration / duration.sum())
    np.testing.assert_allclose(fitted_pipelines[0].named_steps["scaler"].mean_, np.average(values, axis=0, weights=weights))
    original_environment_path = os.environ.get("ECHOPROOF_MODEL_PATH")
    assert runner.baseline_check(*common)["metrics"] == load_model(model_path)["validation"]["metrics"]
    assert os.environ.get("ECHOPROOF_MODEL_PATH") == original_environment_path
    before = splits.read_bytes()
    runner.fit_candidate(*common, static_proposal("classifier", 2))
    assert splits.read_bytes() == before


def test_changed_split_hash_refuses_candidate_before_fit(fitted):
    root, splits, model_path, _, cache = fitted
    with pytest.raises(ValueError, match="changed"):
        runner.fit_candidate(root, splits, model_path, cache, "different", file_digest(model_path), static_proposal("features", 1))


@pytest.mark.parametrize("changes", [{"rounds": 4}, {"agents_per_round": 4}, {"llm_calls": 21}, {"experiments": 13},
                                     {"rounds": True}, {"run_timeout_s": float("nan")}, {"train_timeout_s": 0}])
def test_hard_caps_cannot_be_raised(changes):
    with pytest.raises(ValueError):
        replace(runner.Limits(), **changes)


def _slow_operation():
    time.sleep(30)
    return {"should_not_return": True}


def test_actual_worker_is_stopped_by_timeout():
    started = time.monotonic()
    result = runner.bounded_call(_slow_operation, (), 0.2)
    assert result["status"] == "timeout" and result["result"] is None
    assert time.monotonic() - started < 3


def _simulated_bounded(monkeypatch, baseline, *, cloud_fails=False):
    def simulated(action, args, timeout_s):
        if action is runner.baseline_check:
            result = {"metrics": baseline, "seed": 42, "n_train": 8, "n_validation": baseline["n_clips"]}
        elif action is runner.fit_candidate:
            result = {"metrics": baseline, "validation_diagnostics": {}, "seed": 42}
        elif cloud_fails:
            return {"status": "failed", "result": None, "reason": "Luminal HTTP 429. Provider rate limit reached; retry later.", "runtime_s": 0.1}
        else:
            result = action(*args)
        return {"status": "ok", "result": result, "reason": None, "runtime_s": 0.1}

    monkeypatch.setattr(orchestrator, "bounded_call", simulated)


def test_dry_run_caps_real_fit_attempts_and_marks_static_source(fitted, tmp_path, monkeypatch):
    root, splits, model_path, _, cache = fitted
    _simulated_bounded(monkeypatch, load_model(model_path)["validation"]["metrics"])
    output = tmp_path / "events.jsonl"
    original_splits, original_model = splits.read_bytes(), model_path.read_bytes()
    result = orchestrator.run_experiments(root=root, splits=splits, model_path=model_path, cache=cache, output=output,
                                         dry_run=True, limits=runner.Limits(experiments=2))
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["reason"] == "experiment_limit" and result["model_experiments"] == 2
    assert result["llm_calls"] == 0 and result["best_proposal"] is None
    assert all(event["dry_run"] and event["mode"] == "dry_run_static_proposals" for event in events)
    completed = [event for event in events if event["event"] == "experiment"]
    assert len(completed) == 2 and all(event["decision"] == "rejected" for event in completed)
    assert splits.read_bytes() == original_splits and model_path.read_bytes() == original_model


def test_failed_cloud_calls_consume_budget_and_never_fallback(fitted, config, tmp_path, monkeypatch):
    root, splits, model_path, _, cache = fitted
    _simulated_bounded(monkeypatch, load_model(model_path)["validation"]["metrics"], cloud_fails=True)
    output = tmp_path / "events.jsonl"
    result = orchestrator.run_experiments(root=root, splits=splits, model_path=model_path, cache=cache, output=output,
        client=LuminalClient(config), limits=runner.Limits(llm_calls=2))
    assert result["llm_calls"] == 2 and result["model_experiments"] == 0
    assert result["reason"] == "llm_call_limit" and result["status"] == "partial_failure"
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(event["mode"] == "cloud" and not event["dry_run"] for event in events)
    assert all(event["proposal"] is None for event in events)
    assert "private-test-key" not in output.read_text()


def test_cloud_summaries_are_aggregate_and_rejected_proposals_never_fit(fitted, config, tmp_path, monkeypatch):
    root, splits, model_path, _, cache = fitted
    summaries = []

    class InvalidClient:
        def __init__(self):
            self.config = config

        def propose(self, role, summary):
            summaries.append(summary)
            return {"shell": "forbidden generated command"}

    _simulated_bounded(monkeypatch, load_model(model_path)["validation"]["metrics"])
    output = tmp_path / "events.jsonl"
    result = orchestrator.run_experiments(root=root, splits=splits, model_path=model_path, cache=cache,
        output=output, client=InvalidClient(), limits=runner.Limits(rounds=1))
    assert result["llm_calls"] == 3 and result["model_experiments"] == 0
    assert len(summaries) == 3
    serialized = json.dumps(summaries)
    assert "flac" not in serialized and "test_label" not in serialized and "sha256" not in serialized
    assert "forbidden generated command" not in output.read_text()


def test_missing_baseline_prevents_even_dry_run(tmp_path):
    with pytest.raises(ValueError, match="baseline"):
        orchestrator.run_experiments(root=tmp_path, splits=tmp_path / "splits.csv", model_path=tmp_path / "absent.joblib",
                                     output=tmp_path / "events.jsonl", dry_run=True)
    assert not (tmp_path / "events.jsonl").exists()


def test_missing_cloud_config_records_failure_without_calls(fitted, tmp_path, monkeypatch):
    root, splits, model_path, _, cache = fitted
    _simulated_bounded(monkeypatch, load_model(model_path)["validation"]["metrics"])
    for key in ("API_KEY", "BASE_URL", "MODEL_ID"):
        monkeypatch.delenv("LUMINAL_" + key, raising=False)
    output = tmp_path / "events.jsonl"
    with pytest.raises(LuminalError, match="Configure LUMINAL"):
        orchestrator.run_experiments(root=root, splits=splits, model_path=model_path, cache=cache, output=output)
    last = json.loads(output.read_text().splitlines()[-1])
    assert last["status"] == "failed" and last["llm_calls"] == 0 and last["model_experiments"] == 0


def test_overall_deadline_stops_additional_calls_and_fits(fitted, tmp_path, monkeypatch):
    root, splits, model_path, _, cache = fitted
    baseline = load_model(model_path)["validation"]["metrics"]
    clock = [0.0]
    durations = []
    monkeypatch.setattr(orchestrator.time, "monotonic", lambda: clock[0])

    def time_limited(action, args, timeout_s):
        durations.append(timeout_s)
        if action is runner.baseline_check:
            clock[0] += 0.5
            return {"status": "ok", "result": {"metrics": baseline, "seed": 42, "n_train": 8, "n_validation": 4},
                    "reason": None, "runtime_s": 0.5}
        clock[0] += timeout_s
        return {"status": "timeout", "result": None, "reason": "operation_timeout", "runtime_s": timeout_s}

    monkeypatch.setattr(orchestrator, "bounded_call", time_limited)
    result = orchestrator.run_experiments(root=root, splits=splits, model_path=model_path, cache=cache,
        output=tmp_path / "events.jsonl", dry_run=True, limits=runner.Limits(run_timeout_s=1, train_timeout_s=10))
    assert result["status"] == "timeout" and result["model_experiments"] == 1
    assert durations == [1, 0.5]


def test_documentation_schema_matches_events_and_has_no_claimed_measurements(fitted, tmp_path, monkeypatch):
    root, splits, model_path, _, cache = fitted
    _simulated_bounded(monkeypatch, load_model(model_path)["validation"]["metrics"])
    result = orchestrator.run_experiments(root=root, splits=splits, model_path=model_path, cache=cache,
        output=tmp_path / "events.jsonl", dry_run=True, limits=runner.Limits(rounds=1))
    directory = Path(__file__).resolve().parents[1] / "agents"
    schema = json.loads((directory / "event.schema.json").read_text())
    fixture = json.loads((directory / "example_event.json").read_text())
    assert set(result) == set(schema["properties"]) == set(fixture)
    assert set(schema["required"]) <= set(result)
    assert fixture["status"] == "not_run" and fixture["metrics"] is None and fixture["best_balanced_accuracy"] is None
