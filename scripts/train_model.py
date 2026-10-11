"""Train the fast spectral/MFCC baseline and save measured validation metrics."""

from scripts._workflow import parser, run
from src.training import train


def main() -> int:
    args = parser(__doc__).parse_args()
    return run(lambda: train(args.root, args.splits, args.model, args.run_dir, args.cache)["metrics"])


if __name__ == "__main__":
    raise SystemExit(main())
