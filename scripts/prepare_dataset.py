"""Audit every v4 clip and persist approximate 70/15/15 source-group splits."""

from scripts._workflow import parser, run
from src.dataset import PARTITIONS, prepare


def main() -> int:
    cli = parser(__doc__)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()

    def action():
        rows = prepare(args.root, args.splits, args.seed)
        return {split: {"clips": sum(row["split"] == split for row in rows),
                        "real": sum(row["split"] == split and int(row["label"]) == 0 for row in rows),
                        "fake": sum(row["split"] == split and int(row["label"]) == 1 for row in rows),
                        "groups": len({row["group"] for row in rows if row["split"] == split})}
                for split in PARTITIONS}

    return run(action)


if __name__ == "__main__":
    raise SystemExit(main())
