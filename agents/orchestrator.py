"""Append-only experiment events; hard budgets apply to every run, including failures."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from agents.experiment_runner import Limits, baseline_check, bounded_call, fit_candidate
from agents.luminal_client import LuminalClient, LuminalConfig, LuminalError
from agents.proposals import ROLES, ProposalError, static_proposal, validate_proposal
from src.model import file_digest


def run_experiments(*, root=Path("data/deepfake-audio-detection"), splits=Path("data/splits.csv"),
                    model_path=Path("models/baseline.joblib"), cache=Path("data/features"),
                    output=Path("runs/experiments.jsonl"), dry_run=False, limits=None, client=None,
                    inthewild=False, prior_history=None, artifact_dir=None, run_id=None, research=None) -> dict:
    if research is not None:
        from agents.research_loop import run_research
        return run_research(research, client=client)
    limits = limits or Limits.from_env()
    if not isinstance(limits, Limits):
        raise ValueError("Use validated Limits for experiment budgets.")
    if not inthewild and (limits.rounds > 3 or limits.llm_calls > 20 or limits.experiments > 12):
        raise ValueError("Use the research profile for larger budgets; the legacy frontend event schema is preserved.")
    baseline_action, fit_action = baseline_check, fit_candidate
    if inthewild:
        from agents import inthewild_context as context
        if limits.rounds > 2 or limits.experiments > 6 or limits.llm_calls > 12:
            raise ValueError("InTheWild approval caps: two rounds, six experiments, twelve calls.")
        if output.exists() or artifact_dir is None or artifact_dir.exists():
            raise ValueError("Use separate new history and artifact paths.")
        baseline_action, fit_action = context.baseline_check, context.fit_candidate
    started = time.monotonic()
    deadline = started + limits.run_timeout_s
    if not splits.is_file() or not model_path.is_file():
        raise ValueError("Prepare data/splits.csv and train/verify the Gary Stafford baseline before experiments.")
    splits_hash, model_hash = file_digest(splits), file_digest(model_path)
    run_id = run_id or uuid.uuid4().hex
    mode = "dry_run_static_proposals" if dry_run else "cloud"
    calls = experiments = 0
    best_score, best_proposal, seed = None, None, None
    history = list(prior_history or [])
    best_artifact = str(model_path)
    provider = ({"provider": "Luminal", "base_url": client.config.base_url,
                 "model_id": client.config.model_id, "max_output_tokens": client.config.max_output_tokens,
                 "automatic_retries": 0} if inthewild and client is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(event, *, round_number=None, role=None, proposal=None, status="ok", runtime_s=0.0,
               decision=None, reason=None, measured=None, diagnostics=None, response=None, saved_artifact=None):
        value = {"schema_version": 1, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "event": event, "mode": mode, "dry_run": dry_run, "round": round_number, "role": role,
                 "seed": seed, "proposal": proposal, "status": status, "runtime_s": runtime_s,
                 "decision": decision, "reason": reason, "metrics": measured, "validation_diagnostics": diagnostics,
                 "best_balanced_accuracy": best_score, "best_proposal": best_proposal,
                 "splits_sha256": splits_hash, "baseline_sha256": model_hash,
                 "llm_calls": calls, "model_experiments": experiments, "limits": asdict(limits)}
        if inthewild:
            value.update(experiment_profile="inthewild_logistic_only", provider=provider,
                         provider_response=response, saved_artifact=saved_artifact,
                         best_artifact=best_artifact, test_accessed=False, diagnostic_used=False)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        return value

    record("run_started", reason="Static proposal fixtures; all fits/metrics are real." if dry_run else "Actual provider proposals only.")
    common = (root, splits, model_path, cache, splits_hash, model_hash)
    baseline = bounded_call(baseline_action, common, min(limits.train_timeout_s, deadline - time.monotonic()))
    if baseline["status"] != "ok":
        record("baseline", status=baseline["status"], runtime_s=baseline["runtime_s"], reason=baseline["reason"])
        return record("run_finished", status="failed", reason="baseline_prerequisite_failed", runtime_s=time.monotonic() - started)
    seed = baseline["result"]["seed"]
    best_score = baseline["result"]["metrics"]["balanced_accuracy"]
    record("baseline", measured=baseline["result"]["metrics"], runtime_s=baseline["runtime_s"])
    if not dry_run and client is None:
        try:
            client = LuminalClient(LuminalConfig.from_env())
        except LuminalError as exc:
            record("run_finished", status="failed", reason=str(exc), runtime_s=time.monotonic() - started)
            raise
    seen = set()
    if inthewild:
        seen.add(json.dumps(baseline["result"]["baseline_configuration"], sort_keys=True))
    stop_reason = "round_limit"
    any_failure = False
    for round_number in range(1, limits.rounds + 1):
        for role in ROLES[:limits.agents_per_round]:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or experiments >= limits.experiments or (not dry_run and calls >= limits.llm_calls):
                stop_reason = "run_timeout" if remaining <= 0 else "experiment_limit" if experiments >= limits.experiments else "llm_call_limit"
                break
            if file_digest(splits) != splits_hash or file_digest(model_path) != model_hash:
                any_failure = True
                stop_reason = "frozen_inputs_changed"
                break
            proposal_started = time.monotonic()
            if dry_run:
                proposed = {"status": "ok", "result": static_proposal(role, round_number), "reason": None, "runtime_s": 0.0}
            else:
                calls += 1  # Failed/time-out attempts consume the call budget too.
                summary = {"round": round_number, "seed": seed, "n_train": baseline["result"]["n_train"],
                           "n_validation": baseline["result"]["n_validation"], "baseline_metrics": baseline["result"]["metrics"],
                           "best_validation_balanced_accuracy": best_score, "experiments": history,
                           "limits": asdict(limits), "limitations": "Small English dataset; vendor slices are seen-vendor validation, not independent generalization. Scores are uncalibrated."}
                action = client.propose_recorded if inthewild else client.propose
                if inthewild:
                    summary.update(experiment_profile="inthewild_logistic_only",
                                   baseline_configuration=baseline["result"]["baseline_configuration"],
                                   best_configuration=best_proposal or baseline["result"]["baseline_configuration"],
                                   objective="Maximize balanced accuracy, report FPR/FNR tradeoffs. Threshold selected on validation only.",
                                   limitations=context.LIMITATIONS + " Repeated validation optimization may overfit; speaker reliability is uneven.")
                    # Exact aggregate payload is saved before the one paid attempt.
                    prompt_path = output.parent / f"{run_id}-call-{calls:02d}.json"
                    with prompt_path.open("x") as stream:
                        json.dump({"role": role, "summary": summary, "provider": provider}, stream, indent=2)
                    record("call_started", round_number=round_number, role=role)
                    action = client.propose_recorded
                proposed = bounded_call(action, (role, summary), min(client.config.timeout_s, deadline - time.monotonic()))
            if proposed["status"] != "ok":
                any_failure = True
                record("experiment", round_number=round_number, role=role, status=proposed["status"],
                       decision="rejected", reason=proposed["reason"], runtime_s=proposed["runtime_s"])
                history.append({"role": role, "status": proposed["status"], "reason": proposed["reason"]})
                continue
            try:
                response = None
                value = proposed["result"]
                if inthewild and not dry_run:
                    response = value["provider_response"]
                    value = value["proposal"]
                proposal = validate_proposal(value, role)
                if inthewild:
                    record("proposal", round_number=round_number, role=role, proposal=proposal, response=response)
                    context.validate_logistic(proposal)
            except ProposalError:
                any_failure = True
                record("experiment", round_number=round_number, role=role, status="rejected", decision="rejected",
                       reason="proposal_outside_allowlist", runtime_s=time.monotonic() - proposal_started)
                history.append({"role": role, "status": "rejected", "reason": "proposal_outside_allowlist"})
                continue
            configuration = {key: value for key, value in proposal.items() if key != "role"}
            if inthewild:
                configuration = {**configuration, "parameters": {**proposal["parameters"], "C": float(proposal["parameters"]["C"])}}
            fingerprint = json.dumps(configuration, sort_keys=True)
            if fingerprint in seen:
                record("experiment", round_number=round_number, role=role, proposal=proposal, status="rejected",
                       decision="rejected", reason="duplicate_configuration", runtime_s=time.monotonic() - proposal_started)
                history.append({"role": role, "proposal": proposal, "status": "rejected", "reason": "duplicate_configuration"})
                continue
            seen.add(fingerprint)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop_reason = "run_timeout"
                break
            experiments += 1  # A failed/timed-out fit still consumes an experiment slot.
            fit_args = (*common, proposal)
            if inthewild:
                export_path = artifact_dir / f"experiment-{experiments:02d}.joblib"
                fit_args += (export_path, {"run_id": run_id, "mode": mode, "provider": provider,
                                          "provider_response": response, "round": round_number, "role": role},
                             baseline["result"]["cache_sha256"])
            outcome = bounded_call(fit_action, fit_args, min(limits.train_timeout_s, remaining))
            measured = diagnostics = None
            decision, reason = "rejected", outcome["reason"]
            if outcome["status"] == "ok":
                measured = outcome["result"]["metrics"]
                diagnostics = outcome["result"]["validation_diagnostics"]
                if measured["balanced_accuracy"] > best_score + 1e-12:
                    best_score, best_proposal = measured["balanced_accuracy"], proposal
                    if inthewild:
                        best_artifact = outcome["result"]["saved_artifact"]
                    decision, reason = "accepted", "validation_balanced_accuracy_improved"
                else:
                    reason = "no_validation_balanced_accuracy_improvement"
            else:
                any_failure = True
            record("experiment", round_number=round_number, role=role, proposal=proposal, status=outcome["status"],
                   decision=decision, reason=reason, measured=measured, diagnostics=diagnostics,
                   response=response, saved_artifact=outcome["result"].get("saved_artifact") if inthewild and outcome["status"] == "ok" else None,
                   runtime_s=time.monotonic() - proposal_started)
            history.append({"role": role, "proposal": proposal, "status": outcome["status"], "decision": decision,
                            "metrics": measured, "reason": reason})
        else:
            continue
        break
    if file_digest(splits) != splits_hash or file_digest(model_path) != model_hash:
        any_failure, stop_reason = True, "frozen_inputs_changed"
    elif time.monotonic() >= deadline:
        stop_reason = "run_timeout"
    status = "timeout" if stop_reason == "run_timeout" else "partial_failure" if any_failure else "ok"
    return record("run_finished", status=status, reason=stop_reason, runtime_s=time.monotonic() - started)
