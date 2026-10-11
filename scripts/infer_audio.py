"""Screen a local file, or demonstrate inference on one already-evaluated labeled test clip."""

import json
import os
from pathlib import Path

from scripts._workflow import parser
from src.dataset import load_splits
from src.inference import predict_audio


def main() -> int:
    cli = parser(__doc__)
    choice = cli.add_mutually_exclusive_group(required=True)
    choice.add_argument("--audio", type=Path)
    choice.add_argument("--test-clip", action="store_true")
    args = cli.parse_args()
    try:
        if args.test_clip:
            if not (args.run_dir / "test_metrics.json").is_file():
                raise ValueError("Run final evaluation before inspecting a labeled test prediction.")
            row = next(row for row in load_splits(args.splits) if row["split"] == "test")
            path = args.root / row["path"]
            print(json.dumps({"labeled_test_clip": row["path"], "ground_truth": int(row["label"]),
                              "group": row["group"]}))
        else:
            path = args.audio
        os.environ["ECHOPROOF_MODEL_PATH"] = str(args.model.resolve())
        result = predict_audio(path.read_bytes(), path.name)
    except (ValueError, OSError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
