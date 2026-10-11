"""Export a completed run's winner locally; never calls Luminal or accesses final test."""

from pathlib import Path

from scripts._workflow import parser, run
from src.promotion import reproduce


def main() -> int:
    cli = parser(__doc__)
    cli.add_argument("--run-id", required=True)
    cli.add_argument("--history", type=Path, default=Path("runs/experiments.jsonl"))
    cli.add_argument("--baseline", type=Path, default=Path("models/baseline.joblib"))
    cli.set_defaults(model=None, run_dir=None)
    args = cli.parse_args()
    # Restrict identifiers used in default artifact paths.
    if len(args.run_id) != 32 or any(char not in "0123456789abcdef" for char in args.run_id):
        cli.error("run-id must be a 32-character lowercase hexadecimal identifier")
    return run(lambda: reproduce(root=args.root, splits=args.splits, baseline_path=args.baseline,
        history=args.history, run_id=args.run_id,
        model_path=args.model or Path(f"models/optimized-v1-{args.run_id}.joblib"),
        run_dir=args.run_dir or Path(f"runs/optimized-v1-{args.run_id}"), cache=args.cache)["metrics"])


if __name__ == "__main__":
    raise SystemExit(main())
