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
                    output=Path("runs/experiments.jsonl"), dry_run=False, limits=None, client=None) -> dict:
    limits = limits or Limits.from_env()
    if not isinstance(limits, Limits):
        raise ValueError("Use validated Limits for experiment budgets.")
    started = time.monotonic()
    deadline = started + limits.run_timeout_s
    if not splits.is_file() or not model_path.is_file():
        raise ValueError("Prepare data/splits.csv and train/verify the Gary Stafford baseline before experiments.")
    splits_hash, model_hash = file_digest(splits), file_digest(model_path)
    run_id = uuid.uuid4().hex
    mode = "dry_run_static_proposals" if dry_run else "cloud"
    calls = experiments = 0
    best_score, best_proposal, seed = None, None, None
    history = []
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(event, *, round_number=None, role=None, proposal=None, status="ok", runtime_s=0.0,
               decision=None, reason=None, measured=None, diagnostics=None):
        value = {"schema_version": 1, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "event": event, "mode": mode, "dry_run": dry_run, "round": round_number, "role": role,
                 "seed": seed, "proposal": proposal, "status": status, "runtime_s": runtime_s,
                 "decision": decision, "reason": reason, "metrics": measured, "validation_diagnostics": diagnostics,
                 "best_balanced_accuracy": best_score, "best_proposal": best_proposal,
                 "splits_sha256": splits_hash, "baseline_sha256": model_hash,
                 "llm_calls": calls, "model_experiments": experiments, "limits": asdict(limits)}
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        return value

    record("run_started", reason="Static proposal fixtures; all fits/metrics are real." if dry_run else "Actual provider proposals only.")
    common = (root, splits, model_path, cache, splits_hash, model_hash)
    baseline = bounded_call(baseline_check, common, min(limits.train_timeout_s, deadline - time.monotonic()))
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
                proposed = bounded_call(client.propose, (role, summary), min(client.config.timeout_s, remaining))
            if proposed["status"] != "ok":
                any_failure = True
                record("experiment", round_number=round_number, role=role, status=proposed["status"],
                       decision="rejected", reason=proposed["reason"], runtime_s=proposed["runtime_s"])
                history.append({"role": role, "status": proposed["status"], "reason": proposed["reason"]})
                continue
            try:
                proposal = validate_proposal(proposed["result"], role)
            except ProposalError:
                any_failure = True
                record("experiment", round_number=round_number, role=role, status="rejected", decision="rejected",
                       reason="proposal_outside_allowlist", runtime_s=time.monotonic() - proposal_started)
                history.append({"role": role, "status": "rejected", "reason": "proposal_outside_allowlist"})
                continue
            fingerprint = json.dumps({key: value for key, value in proposal.items() if key != "role"}, sort_keys=True)
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
            outcome = bounded_call(fit_candidate, (*common, proposal), min(limits.train_timeout_s, remaining))
            measured = diagnostics = None
            decision, reason = "rejected", outcome["reason"]
            if outcome["status"] == "ok":
                measured = outcome["result"]["metrics"]
                diagnostics = outcome["result"]["validation_diagnostics"]
                if measured["balanced_accuracy"] > best_score + 1e-12:
                    best_score, best_proposal = measured["balanced_accuracy"], proposal
                    decision, reason = "accepted", "validation_balanced_accuracy_improved"
                else:
                    reason = "no_validation_balanced_accuracy_improvement"
            else:
                any_failure = True
            record("experiment", round_number=round_number, role=role, proposal=proposal, status=outcome["status"],
                   decision=decision, reason=reason, measured=measured, diagnostics=diagnostics,
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
