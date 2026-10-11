"""Explicit, revision-pinned raw FLAC download. Run with python -m scripts.download_dataset."""

import argparse
import json
from pathlib import Path
import sys

MANIFEST_PATH = Path(__file__).with_name("dataset_manifest.json")
DEFAULT_ROOT = Path("data/deepfake-audio-detection")
ALLOW_PATTERNS = ["real/*.flac", "fake/*.flac", "README.md"]


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def check_inventory(root: Path) -> dict:
    expected = set(manifest()["files"])
    actual = {p.relative_to(root).as_posix() for folder in ("real", "fake")
              for p in (root / folder).glob("*.flac")}
    return {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected),
            "empty": sorted(p for p in expected & actual if (root / p).stat().st_size == 0),
            "readme_missing": not (root / "README.md").is_file()}


def download(root: Path = DEFAULT_ROOT) -> dict:
    # Import only on explicit invocation; no datasets/Parquet or second audio cache.
    from huggingface_hub import snapshot_download

    source = manifest()
    snapshot_download(repo_id=source["repo_id"], repo_type="dataset",
                      revision=source["revision"], allow_patterns=ALLOW_PATTERNS,
                      local_dir=str(root), max_workers=4)
    report = check_inventory(root)
    (root / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(report.values()):
        raise ValueError(f"Incomplete dataset; inspect {root / 'download_report.json'} and rerun download.")
    (root / "provenance.json").write_text(json.dumps(source, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        download(args.root)
    except Exception as exc:
        # Hub errors may contain authentication details; do not echo their payload.
        print(f"Download failed ({type(exc).__name__}). Check network/access and rerun; "
              "completed local files are reusable. No Parquet was requested.", file=sys.stderr)
        return 1
    print(f"Downloaded 1,866 raw clips to {args.root}. Run preparation to check decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
