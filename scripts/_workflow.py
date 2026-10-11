"""Small common CLI configuration; no work happens on import."""

import argparse
import json
from pathlib import Path
import sys

from scripts.download_dataset import DEFAULT_ROOT


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    result.add_argument("--model", type=Path, default=Path("models/baseline.joblib"))
    result.add_argument("--run-dir", type=Path, default=Path("runs/baseline"))
    result.add_argument("--cache", type=Path, default=Path("data/features"))
    return result


def run(action) -> int:
    try:
        report = action()
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0
