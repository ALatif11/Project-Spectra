"""Run with python -m agents --dry-run; without --dry-run requires real Luminal configuration."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from agents.experiment_runner import Limits
from agents.luminal_client import LuminalError
from agents.orchestrator import run_experiments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("data/deepfake-audio-detection"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--cache", type=Path, default=Path("data/features"))
    parser.add_argument("--output", type=Path, default=Path("runs/experiments.jsonl"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--agents-per-round", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--train-timeout", type=float, default=120)
    parser.add_argument("--run-timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        limits = Limits.from_env()
        # CLI may further lower root .env budgets, but may never raise them.
        overrides = {}
        for name, value in (("rounds", args.rounds), ("llm_calls", args.max_llm_calls),
                            ("experiments", args.max_experiments)):
            if value is not None:
                if value > getattr(limits, name):
                    raise ValueError("CLI budget cannot exceed the configured environment cap.")
                overrides[name] = value
        limits = replace(limits, **overrides, agents_per_round=args.agents_per_round,
                         train_timeout_s=args.train_timeout, run_timeout_s=args.run_timeout)
        result = run_experiments(root=args.root, splits=args.splits, model_path=args.model, cache=args.cache,
                                 output=args.output, dry_run=args.dry_run, limits=limits)
    except LuminalError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, OSError):
        print("Cannot run experiments. Verify the local dataset, fixed splits, baseline artifact, output path, and hard budget bounds.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
