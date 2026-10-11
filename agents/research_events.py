"""Full research ledger plus exact-v1-schema, per-round/fold frontend projections.

The original frontend schema caps rounds at 3 and fits at 12. Each research
round is therefore a separate v1 child run (one round, <=3 calls, <=9 fold fits).
The parent run, true round numbers, global budgets and hypotheses are in the
research ledger. Never clip/fabricate global counts to satisfy the old schema.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json

from agents.experiment_runner import Limits
from agents.proposals import ProposalError, validate_proposal


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False, sort_keys=True)+"\n")
        stream.flush()


def compatible(proposal):
    if proposal is None:
        return None
    try:
        return validate_proposal(proposal, proposal["role"])
    except ProposalError:
        return None  # Exact configuration remains in the research ledger.


def frontend_event(*, parent_id, round_number, event, splits_hash, model_hash, calls, fits,
                   policy, measured=None, proposal=None, role=None, status="ok", decision=None,
                   reason=None, best_ba=None, runtime_s=0.0):
    return {"schema_version": 1, "run_id": f"{parent_id}:development-round-{round_number}",
            "timestamp": datetime.now(timezone.utc).isoformat(), "event": event, "mode": "cloud", "dry_run": False,
            "round": 1 if role is not None else None, "role": role, "seed": 42,
            "proposal": compatible(proposal), "status": status, "runtime_s": runtime_s,
            "decision": decision, "reason": reason, "metrics": measured, "validation_diagnostics": None,
            "best_balanced_accuracy": best_ba, "best_proposal": None,
            "splits_sha256": splits_hash, "baseline_sha256": model_hash,
            "llm_calls": calls, "model_experiments": fits,
            "limits": asdict(Limits(rounds=1, agents_per_round=3, llm_calls=3, experiments=9,
                                   train_timeout_s=policy.train_timeout_s, run_timeout_s=policy.run_timeout_s))}
