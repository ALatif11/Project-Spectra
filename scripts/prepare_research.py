"""Prepare the expanded profile without cloud calls; benchmark only when approved."""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

from agents.experiment_runner import bounded_call
from agents.inthewild_context import baseline_check, prior_history
from agents.luminal_client import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, LuminalConfig
from agents.research_budget import Pricing, ResearchPolicy
from agents.research_evaluation import cached_evaluation, evaluate_configuration, prepare_bundle, save_json
from agents.research_loop import fit_events
from agents.research_proposals import BASELINE
from scripts.optimize_inthewild import EXPERIMENT, MODEL, protected_hashes
from src.model import file_digest


def source_hashes():
    paths = [*Path("agents").glob("*.py"), Path("agents/research_profile.json"),
             Path("scripts/prepare_research.py"), Path("scripts/optimize_inthewild.py"),
             *[Path("src")/name for name in ("inthewild.py", "model.py", "training.py", "features.py", "audio.py", "spectrogram.py")]]
    return {str(p): file_digest(p) for p in paths}


def prior_overview():
    history = prior_history(Path("runs/experiments.jsonl"), EXPERIMENT/"comparison.json")
    overview = {"different_dataset_history": {"experiments": len(history)-1,
                 "successful_measured_experiments": sum(e.get("metrics") is not None for e in history[:-1]),
                 "caution": "Gary Stafford results use a different dataset/protocol; not directly comparable."},
                "inthewild_validation_candidate": history[-1]}
    measured = []
    for path in sorted(Path("runs").glob("inthewild-luminal-*/experiments.jsonl")):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "experiment" and e.get("metrics"):
                measured.append({"proposal": e["proposal"], "metrics": e["metrics"], "source": "live_luminal",
                                 "comparison_note": "earlier fixed validation protocol; training-only folds not measured"})
    overview["prior_inthewild_live_experiments"] = measured[-8:]
    return overview


def prepare(run_dir, profile_path, benchmark=False):
    if (run_dir/"research-prepared.json").exists() or (run_dir/"live_started.json").exists():
        raise ValueError("Preserve the existing preparation/run; choose a fresh directory.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir/".gitignore").exists():
        (run_dir/".gitignore").write_text("*\n")
    profile = json.loads(profile_path.read_text())
    policy, pricing = ResearchPolicy(**profile["policy"]), Pricing(**profile["pricing"])
    config = LuminalConfig.from_env()  # Parsing only, zero requests.
    if config.base_url.rstrip("/") != DEFAULT_BASE_URL or config.model_id != DEFAULT_MODEL_ID:
        raise ValueError("Use the verified Luminal DeepSeek V4.1 Flash model/endpoint.")
    protected = protected_hashes()
    for path in Path("runs").glob("inthewild-luminal-*/*"):
        if path.is_file():
            protected[str(path)] = file_digest(path)
    common = (Path("data/inthewild"), EXPERIMENT/"optimization.json", MODEL, EXPERIMENT/"features",
              file_digest(EXPERIMENT/"optimization.json"), file_digest(MODEL))
    check = bounded_call(baseline_check, common, 120)
    if check["status"] != "ok":
        save_json(run_dir/"research-preflight-failed.json", check)
        raise ValueError("Existing candidate/cache check failed; zero cloud calls.")
    started = time.monotonic()
    bundle, metadata = prepare_bundle(EXPERIMENT, Path("data/inthewild/research-cache-v2/bundles"))
    read_seconds = time.monotonic()-started
    evaluation_cache = Path("data/inthewild/research-cache-v2/evaluations")
    benchmark_result = None
    if benchmark:
        save_json(run_dir/"benchmark_started.json", {"max_local_fits": 3, "api_calls": 0,
                  "authorization": "User explicitly approved up to three cached-feature local benchmark fits."})
        journal = run_dir/"benchmark-fits.jsonl"
        outcome = bounded_call(evaluate_configuration, (bundle, BASELINE, evaluation_cache, journal, 3), 120)
        events = fit_events(journal)
        attempts = sum(e["event"] == "fit_started" for e in events)
        if attempts > 3:
            raise RuntimeError("Local benchmark exceeded its three-fit approval.")
        benchmark_result = {"status": outcome["status"], "fit_attempts": attempts,
                            "fits_completed": sum(e["event"] == "fit_completed" for e in events),
                            "wall_runtime_s": outcome["runtime_s"], "api_calls": 0,
                            "result": outcome["result"], "reason": outcome["reason"]}
        save_json(run_dir/"benchmark.json", benchmark_result)
        if outcome["status"] != "ok":
            raise ValueError("Local benchmark failed; no automatic fit retry.")
    baseline_dev = cached_evaluation(bundle, BASELINE, evaluation_cache)
    benchmark_seconds = benchmark_result["wall_runtime_s"] if benchmark_result else None
    max_configurations = (policy.experiments-1-(0 if baseline_dev else 3))//3
    max_calls_scheduled = policy.rounds*3
    old_events = [json.loads(line) for line in Path("runs/inthewild-luminal-20261011-v1/experiments.jsonl").read_text().splitlines()]
    provider_times = [e["runtime_s"] for e in old_events if e["event"] == "experiment" and e["status"] == "failed"]
    historical_call_seconds = sum(provider_times)/len(provider_times) if provider_times else 30.
    # Scaling previous 1000-token call times is an estimate, not a new API measurement.
    projected_call = historical_call_seconds * policy.max_output_tokens/1000
    projected_eval = max(1., benchmark_seconds or 10.)*8  # Conservative allowance for bounded nonlinear models.
    estimated_typical = max_configurations*(projected_call+projected_eval)+projected_eval/3
    estimate = {"maximum_scheduled_calls": max_calls_scheduled, "maximum_distinct_new_configurations": max_configurations,
                "maximum_planned_live_fit_attempts": max_configurations*3+1+(0 if baseline_dev else 3),
                "expected_plateau_stop_configurations": 6,
                "runtime_estimate_s": estimated_typical, "hard_runtime_cap_s": policy.run_timeout_s,
                "historical_1000_token_call_mean_s": historical_call_seconds,
                "estimated_2500_token_call_s": projected_call, "conservative_configuration_runtime_s": projected_eval,
                "runtime_estimate_basis": "measured cached logistic 3-fold evaluation x8 for alternatives; old provider latencies scaled by output cap; variable provider latency",
                "scheduled_calls_conservative_cost_envelope_usd": float(pricing.cost(50048, policy.max_output_tokens))*max_calls_scheduled,
                "absolute_60_call_conservative_cost_envelope_usd": float(pricing.cost(50048, policy.max_output_tokens))*policy.llm_calls,
                "feasible_under_estimate": estimated_typical < policy.run_timeout_s}
    sources = source_hashes()
    for name in sources:
        path = run_dir/"source"/name
        path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, path)
    prepared = {"run_id": uuid.uuid4().hex, "run_dir": str(run_dir), "bundle": str(bundle),
                "model": str(MODEL), "splits_hash": common[-2], "model_hash": common[-1],
                "evaluation_cache": str(evaluation_cache), "policy": asdict(policy), "pricing": asdict(pricing),
                "provider_output_limit_verified": profile.get("provider_output_limit_verified", False),
                "protected_hashes": protected, "source_hashes": sources,
                "baseline_validation": check["result"]["metrics"], "baseline_development": baseline_dev,
                "history_overview": prior_overview(), "feature_bundle": metadata,
                "read_only_cache_preparation_s": read_seconds, "benchmark": benchmark_result, "estimate": estimate,
                "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "scikit-learn", "joblib", "openai", "httpx", "threadpoolctl")},
                "api_calls": 0, "live_started": False, "test_accessed": False, "diagnostic_used": False,
                "exact_live_command": f".venv/bin/python -m scripts.optimize_inthewild --research --live --run-dir {run_dir}"}
    save_json(run_dir/"research-prepared.json", prepared)
    return prepared


def load_prepared(run_dir):
    config = json.loads((run_dir/"research-prepared.json").read_text())
    config["policy"] = ResearchPolicy(**config["policy"])
    config["pricing"] = Pricing(**config["pricing"])
    return config
