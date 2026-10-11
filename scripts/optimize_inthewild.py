"""Explicit offline check, then one live run of the existing three-role system."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from agents.experiment_runner import Limits, bounded_call
from agents.inthewild_context import baseline_check, prior_history
from agents.luminal_client import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, LuminalClient, LuminalConfig
from agents.orchestrator import run_experiments
from src.model import file_digest

EXPERIMENT = Path("data/inthewild/experiment-12000-v1")
MODEL = Path("models/inthewild-v1-12000.joblib")


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def protected_hashes():
    # Never open sealed manifests, raw audio, or diagnostic microphone recordings.
    paths = [*Path("models").glob("*.joblib"), Path("runs/experiments.jsonl"), Path("data/splits.csv"),
             EXPERIMENT / "optimization.json", EXPERIMENT / "comparison.json", EXPERIMENT / "report.md",
             Path("app.py"), *Path("ui").rglob("*.py")]
    return {str(p): file_digest(p) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--live", action="store_true", help="One live run; no automatic retries or deployment")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--research", action="store_true", help="Expanded hypothesis/development-fold profile")
    parser.add_argument("--benchmark", action="store_true", help="Requires separate approval: at most three local fold fits, zero API calls")
    parser.add_argument("--profile", type=Path, default=Path("agents/research_profile.json"))
    args = parser.parse_args()
    if args.research:
        from scripts.prepare_research import load_prepared, prepare
        if args.live:
            if args.benchmark:
                raise ValueError("Benchmark preparation and live execution are separate approvals/stages.")
            result = run_experiments(research=load_prepared(args.run_dir))
            print(json.dumps(result, indent=2))
            return
        prepared = prepare(args.run_dir, args.profile, args.benchmark)
        print(json.dumps({k: prepared[k] for k in ("run_id", "policy", "pricing", "baseline_validation", "estimate", "api_calls", "exact_live_command")}, indent=2))
        return
    if args.benchmark:
        raise ValueError("Use --research for an explicitly approved local benchmark.")
    config = LuminalConfig.from_env()  # Local parsing only; no SDK client/request.
    if config.base_url.rstrip("/") != DEFAULT_BASE_URL or config.model_id != DEFAULT_MODEL_ID:
        raise ValueError("This approval requires the verified Luminal DeepSeek V4.1 Flash configuration.")
    configured = Limits.from_env()
    limits = replace(configured, rounds=min(2, configured.rounds), experiments=min(6, configured.experiments),
                     llm_calls=min(12, configured.llm_calls), agents_per_round=3,
                     train_timeout_s=120, run_timeout_s=600)
    if not args.live:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        identity = protected_hashes()
        common = (Path("data/inthewild"), EXPERIMENT / "optimization.json", MODEL, EXPERIMENT / "features",
                  file_digest(EXPERIMENT / "optimization.json"), file_digest(MODEL))
        outcome = bounded_call(baseline_check, common, limits.train_timeout_s)
        if outcome["status"] != "ok":
            save(args.run_dir / "preflight_failure.json", outcome)
            raise ValueError("Offline check failed; no API calls made.")
        history = prior_history(Path("runs/experiments.jsonl"), EXPERIMENT / "comparison.json")
        check = {"run_id": uuid.uuid4().hex, "checked_at": datetime.now(timezone.utc).isoformat(),
                 "status": "ok", "api_calls": 0, "model_fits": 0, "baseline": outcome["result"],
                 "limits": asdict(limits), "provider": {"base_url": config.base_url, "model_id": config.model_id,
                 "max_output_tokens": config.max_output_tokens, "timeout_s": config.timeout_s, "automatic_retries": 0},
                 "protected_hashes": identity, "existing_history": history,
                 "input_paths": [str(p) for p in common[:4]], "test_accessed": False,
                 "diagnostic_used": False, "deployed": False}
        save(args.run_dir / "preflight.json", check)
        print(json.dumps({key: check[key] for key in ("run_id", "status", "api_calls", "model_fits", "baseline", "limits", "provider")}, indent=2))
        return
    check = json.loads((args.run_dir / "preflight.json").read_text())
    if check["status"] != "ok" or asdict(limits) != check["limits"]:
        raise ValueError("Offline check missing or effective budgets changed.")
    expected_provider = {"base_url": config.base_url, "model_id": config.model_id,
                         "max_output_tokens": config.max_output_tokens, "timeout_s": config.timeout_s,
                         "automatic_retries": 0}
    if expected_provider != check["provider"]:
        raise ValueError("Provider configuration changed after offline check.")
    if any(file_digest(Path(p)) != digest for p, digest in check["protected_hashes"].items()):
        raise ValueError("Protected inputs changed after offline check.")
    save(args.run_dir / "live_started.json", {"run_id": check["run_id"], "started_at": datetime.now(timezone.utc).isoformat(),
                                            "authorization": "One user-approved live run; no paid retries"})
    root, splits, model, cache = map(Path, check["input_paths"])
    result = run_experiments(root=root, splits=splits, model_path=model, cache=cache,
                             output=args.run_dir / "experiments.jsonl", limits=limits,
                             client=LuminalClient(config), inthewild=True, prior_history=check["existing_history"],
                             artifact_dir=args.run_dir / "artifacts", run_id=check["run_id"])
    preservation = {p: file_digest(Path(p)) == digest for p, digest in check["protected_hashes"].items()}
    save(args.run_dir / "preservation.json", {"unchanged": preservation, "all_preserved": all(preservation.values()),
                                            "test_accessed": False, "diagnostic_used": False, "deployed": False})
    save(args.run_dir / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
