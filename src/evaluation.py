"""Explicit, one-time source-group held-out test evaluation of a frozen model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.download_dataset import manifest
from src.dataset import load_splits
from src.model import file_digest, load_model, score_rows
from src.training import LIMITATIONS, metrics


def evaluate(root: Path, splits: Path, model_path: Path, run_dir: Path, cache: Path) -> dict:
    rows = load_splits(splits)
    artifact = load_model(model_path)
    if artifact["splits_sha256"] != file_digest(splits):
        raise ValueError("Partitions changed since training; refusing test evaluation.")
    for split, key in (("train", "train_groups"), ("validation", "validation_groups")):
        if sorted({row["group"] for row in rows if row["split"] == split}) != artifact[key]:
            raise ValueError("Model training groups disagree with split file.")
    test = [row for row in rows if row["split"] == "test"]
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"model_sha256": file_digest(model_path), "splits_sha256": file_digest(splits)}
    report_path = run_dir / "test_metrics.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if any(report[key] != value for key, value in identity.items()):
            raise ValueError("Final test was already evaluated for another model. Do not optimize on test.")
        return report  # Replay saved metrics without rescoring or accessing test audio.
    marker = run_dir / "test_started.json"
    try:
        with marker.open("x") as stream:
            json.dump(dict(identity, started_at=datetime.now(timezone.utc).isoformat()), stream)
    except FileExistsError as exc:
        raise ValueError("Final test evaluation already started; inspect run files before any manual recovery.") from exc
    scores = score_rows(artifact["pipeline"], root, test, cache)
    labels = np.array([int(row["label"]) for row in test])
    report = {**identity, "partition": "test", "seed": artifact["seed"],
              "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "model_version": artifact["model_version"],
              "source": manifest()["repo_id"], "revision": manifest()["revision"],
              "metrics": metrics(labels, scores, artifact["threshold"]),
              "groups": sorted({row["group"] for row in test}), "limitations": LIMITATIONS,
              "robustness_test": "Not performed; vendors were not held out of optimization.",
              "clips": [dict(path=row["path"], label=int(row["label"]), group=row["group"],
                             synthetic_score=float(score)) for row, score in zip(test, scores)]}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
