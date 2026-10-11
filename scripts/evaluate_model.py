"""Evaluate the frozen model on final test once; subsequent calls replay saved metrics."""

from scripts._workflow import parser, run
from src.evaluation import evaluate


def main() -> int:
    args = parser(__doc__).parse_args()
    return run(lambda: evaluate(args.root, args.splits, args.model, args.run_dir, args.cache)["metrics"])


if __name__ == "__main__":
    raise SystemExit(main())
