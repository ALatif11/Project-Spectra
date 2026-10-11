"""Expanded profile of the existing bounded three-role experiment orchestrator."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from agents.experiment_runner import bounded_call
from agents.luminal_client import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, LuminalClient, LuminalConfig
from agents.proposals import ROLES
from agents.research_budget import Budget
from agents.research_evaluation import (OBJECTIVE, cached_evaluation, evaluate_configuration, known_evaluations,
                                       final_candidate, passes_guardrails, save_json)
from agents.research_events import append, frontend_event
from agents.research_proposals import BASELINE, fingerprint, instructions, proposal_schema, validate_configuration
from src.model import file_digest


def fit_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def assert_frozen(config):
    for name, digest in config["protected_hashes"].items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Protected checkpoint, result or partition changed.")
    bundle = Path(config["bundle"])
    meta = json.loads((bundle/"bundle.json").read_text())
    if file_digest(bundle/"folds.json") != meta["folds_sha256"]:
        raise ValueError("Development folds changed.")
    if any(file_digest(bundle/name) != digest for name, digest in meta["array_hashes"].items()):
        raise ValueError("Prepared feature arrays changed.")
    for name, digest in config.get("source_hashes", {}).items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Research source changed after preflight.")


def round_summary(round_number, config, policy, baseline, best, best_proposal, history, seen):
    # Full local ledger is preserved; cloud receives bounded aggregate feedback.
    worst = sorted(best["heldout_speakers"].items(), key=lambda item: item[1]["balanced_accuracy"]
                   if item[1]["balanced_accuracy"] is not None else 1.)[:6]
    return {"round": round_number, "seed": 42, "dataset": "InTheWild",
            "evaluation_protocol": best["protocol"], "original_validation_reference": config["baseline_validation"],
            "original_validation_use": "Not used for iterative selection; final chosen configuration is scored once at end.",
            "baseline_development_metrics": baseline["metrics"], "baseline_objective": baseline["objective"],
            "best_development_metrics": best["metrics"], "best_objective": best["objective"],
            "best_configuration": best_proposal, "objective_definition": OBJECTIVE,
            "minimum_meaningful_improvement": policy.minimum_improvement,
            "fold_metrics": best["fold_metrics"], "worst_heldout_speaker_groups": dict(worst),
            "feature_and_error_diagnostics": best["diagnostics"], "recent_measured_outcomes": history[-8:],
            "prior_history_overview": config["history_overview"], "tried_configuration_fingerprints": sorted(seen),
            "feature_selection_notes": "All subsets use the existing 52 cached features, same preprocessing and frozen development folds.",
            "limitations": "Development folds and hypotheses are adaptively reused. No independent generalization claim; "
                "unknown session/vendor overlap; English public-figure domain bias. No microphone examples or sealed test are available."}


def run_research(config, *, client=None):
    policy, pricing = config["policy"], config["pricing"]
    budget = Budget(policy, pricing)
    started = time.monotonic()
    run_dir = Path(config["run_dir"]); bundle = Path(config["bundle"])
    cache = Path(config["evaluation_cache"]); run_id = config["run_id"]
    output = run_dir/"research-history.jsonl"; projection = run_dir/"frontend-events.jsonl"
    if output.exists() or (run_dir/"live_started.json").exists():
        raise ValueError("This research run already started; automatic paid reruns are forbidden.")
    assert_frozen(config)
    if client is None:
        from dataclasses import replace
        client = LuminalClient(replace(LuminalConfig.from_env(), max_output_tokens=policy.max_output_tokens,
                                      timeout_s=policy.request_timeout_s))
    if (client.config.base_url.rstrip("/") != DEFAULT_BASE_URL or client.config.model_id != DEFAULT_MODEL_ID
            or client.config.max_output_tokens != policy.max_output_tokens or pricing.model_id != DEFAULT_MODEL_ID):
        raise ValueError("This profile requires the recorded Luminal Flash model and bounded token ceiling.")
    provider = {"provider": "Luminal", "base_url": client.config.base_url, "model_id": client.config.model_id,
                "max_output_tokens": policy.max_output_tokens, "automatic_retries": 0,
                "provider_output_limit_verified": config.get("provider_output_limit_verified", False)}
    save_json(run_dir/"live_started.json", {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat()})

    def record(event, **fields):
        value = {"schema_version": 2, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                 "event": event, "luminal_provenance": provider, "budget": budget.snapshot(),
                 "test_accessed": False, "diagnostic_used": False, "deployed": False, **fields}
        append(output, value)
        return value

    def project(round_number, event, *, calls=0, fits=0, **fields):
        append(projection, frontend_event(parent_id=run_id, round_number=round_number, event=event,
               splits_hash=config["splits_hash"], model_hash=config["model_hash"], calls=calls, fits=fits,
               policy=policy, **fields))

    record("run_started", policy=config["policy"].__dict__, pricing=config["pricing"].__dict__,
           objective=OBJECTIVE, existing_history=config["history_overview"])
    baseline = cached_evaluation(bundle, BASELINE, cache)
    if baseline is None:
        if not budget.can_fit(3):
            return record("run_finished", status="failed", reason="baseline_fit_budget")
        journal = run_dir/"baseline-fits.jsonl"
        outcome = bounded_call(evaluate_configuration, (bundle, BASELINE, cache, journal, 3),
                               min(policy.train_timeout_s, budget.remaining()))
        budget.account_fits(fit_events(journal))
        if outcome["status"] != "ok":
            return record("run_finished", status="failed", reason=outcome["reason"])
        baseline = outcome["result"]["report"]
    record("baseline", status="ok", metrics=baseline["metrics"], development=baseline,
           baseline_validation=config["baseline_validation"])
    best, best_proposal = baseline, BASELINE
    history = []
    seen = {fingerprint(BASELINE)}
    for previous in known_evaluations(bundle, cache):
        proposal = previous["evaluated_proposal"]
        seen.add(fingerprint(proposal))
        history.append({"source": "cached_same_development_protocol", "proposal": proposal,
                        "metrics": previous["metrics"], "objective": previous["objective"]})
        if (previous["objective"]["score"] >= best["objective"]["score"]+policy.minimum_improvement
                and passes_guardrails(previous, baseline)):
            best, best_proposal = previous, proposal
    record("cached_history_loaded", configurations=len(history), best_configuration=best_proposal,
           best_objective=best["objective"])
    stale_rounds = 0; completed_rounds = 0; configurations = 0; any_failure = False
    stop_reason = "round_limit"
    for round_number in range(1, policy.rounds+1):
        meaningful = False; handled = 0
        calls_before, fits_before = budget.calls, budget.fits
        project(round_number, "run_started", reason="training-only speaker development; parent round in child run ID")
        project(round_number, "baseline", measured=best["frontend_fold_metrics"][0],
                best_ba=best["objective"]["mean_fold_balanced_accuracy"], reason="first development fold of incumbent")
        for role in ROLES:
            if not budget.can_fit(3):
                stop_reason = "fit_budget_reserved_for_final_gate" if budget.remaining() else "run_timeout"
                break
            try:
                assert_frozen(config)
            except ValueError:
                stop_reason = "frozen_inputs_changed"; any_failure = True; break
            summary = round_summary(round_number, config, policy, baseline, best, best_proposal, history, seen)
            messages = [{"role": "system", "content": instructions(role)+" Plain JSON only: "+json.dumps(proposal_schema(role))},
                        {"role": "user", "content": json.dumps(summary, allow_nan=False, separators=(",", ":"))}]
            try:
                reservation = budget.reserve_call(messages)
            except ValueError as exc:
                stop_reason = str(exc); break
            save_json(run_dir/f"call-{budget.calls:02d}.json", {"role": role, "messages": messages, "provider": provider,
                                                            "reservation": reservation})
            record("call_started", round=round_number, agent_role=role, reservation=reservation)
            response = bounded_call(client.propose_research, (role, summary), min(policy.request_timeout_s, budget.remaining()))
            data = response["result"] if response["status"] == "ok" else {"status": response["status"], "reason": response["reason"],
                                                                         "provider_response": {}, "proposal": None, "hypothesis": None}
            cost = budget.settle_call(reservation, data.get("provider_response", {}))
            record("call_finished", round=round_number, agent_role=role, status=data["status"],
                   provider_response=data.get("provider_response"), cost=cost, reason=data.get("reason"))
            handled += 1
            if cost["usage_unverified"] and data.get("provider_response"):
                stop_reason = "provider_usage_outside_reservation"; any_failure = True
                record("experiment", round=round_number, agent_role=role, hypothesis=data.get("hypothesis"),
                       proposal=data.get("proposal"), status="rejected", decision="rejected", reason=stop_reason,
                       measured_results=None, previous_best=best["objective"])
                break
            if data["status"] != "ok":
                any_failure = True
                reason = data.get("reason", "provider_failure")
                record("experiment", round=round_number, agent_role=role, hypothesis=None, proposal=None,
                       status=data["status"], decision="rejected", reason=reason, measured_results=None,
                       previous_best=best["objective"])
                project(round_number, "experiment", role=role, status=data["status"], decision="rejected", reason=reason,
                        calls=budget.calls-calls_before, fits=budget.fits-fits_before)
                history.append({"role": role, "status": data["status"], "reason": reason})
                if any(code in reason for code in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "HTTP 422")):
                    stop_reason = "provider_configuration_rejected_no_retry"; break
                continue
            try:
                proposal = validate_configuration(data["proposal"], role)
            except ValueError:
                any_failure = True
                record("experiment", round=round_number, agent_role=role, status="rejected", decision="rejected",
                       reason="proposal_outside_research_allowlist", measured_results=None)
                continue
            hypothesis = data["hypothesis"]
            key = fingerprint(proposal)
            record("proposal", round=round_number, agent_role=role, proposal=proposal, hypothesis=hypothesis,
                   provider_response=data.get("provider_response"), cost=cost)
            if key in seen:
                record("experiment", round=round_number, agent_role=role, proposal=proposal, hypothesis=hypothesis,
                       status="rejected", decision="rejected", reason="duplicate_configuration", measured_results=None,
                       previous_best=best["objective"])
                project(round_number, "experiment", role=role, proposal=proposal, status="rejected", decision="rejected",
                        reason="duplicate_configuration", calls=budget.calls-calls_before, fits=budget.fits-fits_before)
                history.append({"role": role, "proposal": proposal, "status": "rejected", "reason": "duplicate_configuration"})
                continue
            seen.add(key)
            journal = run_dir/f"configuration-{key[:16]}-fits.jsonl"
            before = best["objective"]
            experiment_started = time.monotonic()
            outcome = bounded_call(evaluate_configuration, (bundle, proposal, cache, journal, 3),
                                   min(policy.train_timeout_s, budget.remaining()))
            budget.account_fits(fit_events(journal))
            status, decision, reason = outcome["status"], "rejected", outcome["reason"]
            measured = None; cache_hit = False
            if status == "ok":
                configurations += 1
                measured = outcome["result"]["report"]; cache_hit = outcome["result"]["cache_hit"]
                if measured["objective"]["score"] >= best["objective"]["score"]+policy.minimum_improvement and passes_guardrails(measured, baseline):
                    best, best_proposal = measured, proposal
                    decision, reason, meaningful = "accepted", "development_objective_meaningfully_improved", True
                else:
                    reason = "no_meaningful_objective_improvement_or_quality_guardrail"
                for i, fold_metrics in enumerate(measured["frontend_fold_metrics"]):
                    project(round_number, "experiment", role=role, proposal=proposal, measured=fold_metrics,
                            decision=decision, reason=f"development_fold_{i+1}: "+reason, status="ok",
                            best_ba=best["objective"]["mean_fold_balanced_accuracy"],
                            calls=budget.calls-calls_before, fits=budget.fits-fits_before,
                            runtime_s=measured["fold_training_runtime_s"][i])
            else:
                any_failure = True
                project(round_number, "experiment", role=role, proposal=proposal, status=status, decision="rejected", reason=reason,
                        calls=budget.calls-calls_before, fits=budget.fits-fits_before)
            record("experiment", round=round_number, agent_role=role, proposal=proposal, hypothesis=hypothesis,
                   status=status, decision=decision, reason=reason, measured_results=measured, previous_best=before,
                   best_objective=best["objective"], cache_hit=cache_hit, training_runtime_s=measured["training_runtime_s"] if measured else None,
                   runtime_s=time.monotonic()-experiment_started, provider_response=data.get("provider_response"), cost=cost)
            history.append({"role": role, "proposal": proposal, "hypothesis": hypothesis, "status": status,
                            "decision": decision, "reason": reason,
                            "metrics": measured["metrics"] if measured else None, "objective": measured["objective"] if measured else None})
        project(round_number, "run_finished", status="partial_failure" if any_failure else "ok",
                reason=stop_reason, calls=budget.calls-calls_before, fits=budget.fits-fits_before,
                best_ba=best["objective"]["mean_fold_balanced_accuracy"])
        complete = handled == 3
        if complete:
            completed_rounds += 1
            stale_rounds = 0 if meaningful else stale_rounds+1
        record("round_finished", round=round_number, complete=complete, meaningful_improvement=meaningful,
               complete_rounds_without_improvement=stale_rounds, best_objective=best["objective"])
        if stale_rounds >= policy.patience:
            stop_reason = "early_stop_two_complete_rounds_without_meaningful_improvement"; break
        if not complete or stop_reason != "round_limit":
            break
    final = None
    safe_final = stop_reason not in ("frozen_inputs_changed", "provider_usage_outside_reservation")
    if fingerprint(best_proposal) != fingerprint(BASELINE) and safe_final and budget.can_fit(1, reserve_final=0):
        assert_frozen(config)
        journal = run_dir/"final-fit.jsonl"
        provenance = {"run_id": run_id, "luminal": provider, "development_artifact": best["artifact"],
                      "selection_objective": best["objective"], "splits_sha256": config["splits_hash"]}
        outcome = bounded_call(final_candidate, (bundle, best_proposal, journal, run_dir/"final-candidate.joblib", provenance),
                               min(policy.train_timeout_s, budget.remaining()))
        budget.account_fits(fit_events(journal))
        if outcome["status"] == "ok":
            final = outcome["result"]
            record("final_validation_gate", status="ok", measured_results=final,
                   balanced_accuracy_improved_over_9292=final["metrics"]["balanced_accuracy"] > config["baseline_validation"]["balanced_accuracy"])
            project("final-validation", "run_started", reason="one final existing-validation gate after development selection")
            project("final-validation", "experiment", proposal=best_proposal, role=best_proposal["role"],
                    measured=final["frontend_metrics"], calls=0, fits=1, decision="accepted" if final["metrics"]["balanced_accuracy"] > config["baseline_validation"]["balanced_accuracy"] else "rejected",
                    reason="one final existing-validation comparison; adaptively used development, not untouched test", runtime_s=final["training_runtime_s"])
            project("final-validation", "run_finished", calls=0, fits=1, reason="not_deployed")
        else:
            any_failure = True
            record("final_validation_gate", status=outcome["status"], reason=outcome["reason"])
    assert_frozen(config)
    result = record("run_finished", status="timeout" if budget.remaining() <= 0 else "partial_failure" if any_failure else "ok",
                    reason="run_timeout" if budget.remaining() <= 0 else stop_reason, completed_rounds=completed_rounds,
                    completed_configurations=configurations, best_development_configuration=best_proposal,
                    best_development_objective=best["objective"], final_validation=final,
                    best_artifact=final["artifact"] if final else str(config["model"]), runtime_s=time.monotonic()-started,
                    independent_generalization_verified=False)
    save_json(run_dir/"research-result.json", result)
    return result
