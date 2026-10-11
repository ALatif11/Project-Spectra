"""Offline validation replay of saved live fits: no fitting, audio or test access."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from agents.inthewild_context import cache_snapshot, fixed_context, validate_logistic
from agents.proposals import feature_indices
from src.inthewild import read_features, summary
from src.model import file_digest, segment_scores


def replay(run_dir):
    check = json.loads((run_dir / "preflight.json").read_text())
    root, splits, model, cache = map(Path, check["input_paths"])
    rows, _ = fixed_context(root, splits, model, cache,
                            check["protected_hashes"][str(splits)], check["protected_hashes"][str(model)])
    if cache_snapshot(splits) != check["baseline"]["cache_sha256"]:
        raise ValueError("Caches differ from offline check.")
    validation = [row for row in rows if row["split"] == "validation"]
    features = [read_features(splits.parent, row) for row in validation]
    labels = [row["label"] for row in validation]
    events = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    reports = []
    with threadpool_limits(limits=1):
        for event in events:
            if event["event"] != "experiment" or event["status"] != "ok" or event["metrics"] is None:
                continue
            proposal = validate_logistic(event["proposal"])
            path = Path(event["saved_artifact"])
            if not path.resolve().is_relative_to((run_dir / "artifacts").resolve()):
                raise ValueError("Artifact is outside this run.")
            artifact = joblib.load(path)
            columns = feature_indices(proposal["feature_set"])
            if (artifact["feature_indices"] != columns or artifact["training_configuration"] != proposal
                    or artifact["provenance"]["run_id"] != check["run_id"]
                    or artifact["provenance"]["provider"] != event["provider"]):
                raise ValueError("Artifact settings or provenance differ from live events.")
            scores = [float(np.average(segment_scores(artifact["pipeline"], values[:, columns]), weights=duration))
                      for values, duration in features]
            measured = summary(labels, scores, artifact["threshold"])
            if measured != event["metrics"] or measured != artifact["validation"]["metrics"]:
                raise ValueError("Saved live validation results do not reproduce exactly.")
            reports.append({"artifact": str(path), "sha256": file_digest(path), "metrics": measured,
                            "exact_metric_match": True})
    return {"run_id": check["run_id"], "artifacts": reports, "api_calls": 0, "model_fits": 0,
            "test_accessed": False, "diagnostic_used": False,
            "verifier_sha256": file_digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.run_dir)
    with (args.run_dir / "replay.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
