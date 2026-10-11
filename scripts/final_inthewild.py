"""Explicit freeze or single final test evaluation; never fit or optimize."""
import argparse
import json

from threadpoolctl import threadpool_limits
from src.inthewild_final import evaluate_once, freeze_selected


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["freeze", "evaluate"])
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            result = freeze_selected() if args.stage == "freeze" else evaluate_once()
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result if args.stage == "evaluate" else
                     {k: result[k] for k in ("model_path", "rollback_path", "model_sha256", "threshold", "validation")}, indent=2))


if __name__ == "__main__":
    main()
