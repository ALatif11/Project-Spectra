"""Explicit final test comparison of two frozen checkpoints; no fitting or API calls."""

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts._workflow import parser, run
from src.evaluation import evaluate
from src.model import file_digest, load_model


def compare(root, splits, baseline, winner, baseline_run, winner_run, cache, history):
    baseline_model, winner_model = load_model(baseline), load_model(winner)
    identity = {"baseline_sha256": file_digest(baseline), "winner_sha256": file_digest(winner),
                "splits_sha256": file_digest(splits)}
    if (baseline_model["splits_sha256"] != identity["splits_sha256"]
            or winner_model["splits_sha256"] != identity["splits_sha256"]
            or winner_model.get("provenance", {}).get("baseline_sha256") != identity["baseline_sha256"]):
        raise ValueError("Model/split provenance does not agree; refusing test access.")
    if baseline_run.resolve() == winner_run.resolve():
        raise ValueError("Use separate evaluation directories to preserve both models' results.")
    winner_run.mkdir(parents=True, exist_ok=True)
    audit_path = winner_run / "test_history_before_comparison.json"
    if not audit_path.exists():
        prior = []
        paths = {path for directory in (baseline_run.parent, baseline_run, winner_run)
                 for path in directory.rglob("test_*.json")}
        for path in sorted(paths):
            if path.name not in ("test_started.json", "test_metrics.json"):
                continue
            record = json.loads(path.read_text())
            prior.append({"path": str(path), "file_sha256": file_digest(path),
                          "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                          "model_sha256": record.get("model_sha256"),
                          "splits_sha256": record.get("splits_sha256"),
                          "evaluated_at": record.get("evaluated_at"),
                          "kind": "completed_evaluation" if path.name == "test_metrics.json" else "access_started"})
        audit = {**identity, "recorded_at": datetime.now(timezone.utc).isoformat(),
                 "prior_evaluation_records": prior,
                 "interpretation": "Repository records only; absence cannot rule out deleted or unlogged evaluations. "
                    "Dataset preparation decodes all clips for integrity/grouping; that is not model evaluation. "
                    "Both checkpoints and validation thresholds are frozen before this comparison. No test-based tuning."}
        with audit_path.open("x") as stream:
            json.dump(audit, stream, indent=2)
    audit = json.loads(audit_path.read_text())
    if any(audit[key] != value for key, value in identity.items()):
        raise ValueError("Comparison already started with different models/splits; preserve original results.")
    history_hash = file_digest(history)
    baseline_report = evaluate(root, splits, baseline, baseline_run, cache)
    winner_report = evaluate(root, splits, winner, winner_run, cache)
    if any(file_digest(path) != identity[key] for path, key in
           ((baseline, "baseline_sha256"), (winner, "winner_sha256"), (splits, "splits_sha256"))):
        raise ValueError("Frozen model/split changed during comparison.")
    if file_digest(history) != history_hash:
        raise ValueError("Experiment history changed during comparison; inspect concurrent activity.")
    report = {**identity, "baseline": baseline_report["metrics"], "winner": winner_report["metrics"],
              "balanced_accuracy_gain": winner_report["metrics"]["balanced_accuracy"] - baseline_report["metrics"]["balanced_accuracy"],
              "f1_gain": winner_report["metrics"]["f1_synthetic"] - baseline_report["metrics"]["f1_synthetic"],
              "history_before_comparison": audit,
              "experiment_history_sha256": history_hash,
              "thresholds": "Frozen from validation; not retuned on test. Metrics are threshold decisions before UI abstention.",
              "confusion_matrix_order": "rows actual real/fake; columns predicted real/fake",
              "robustness": "No distinct vendor holdout; source/domain bias remains possible.",
              "deployment": "Opt in with ECHOPROOF_MODEL_PATH; baseline remains default and fallback."}
    target = winner_run / "comparison.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved != report:
            raise ValueError("Saved comparison differs; refusing replacement.")
        return saved
    with target.open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


def main() -> int:
    cli = parser(__doc__)
    cli.add_argument("--winner", type=Path, required=True)
    cli.add_argument("--winner-run-dir", type=Path, required=True)
    cli.add_argument("--history", type=Path, default=Path("runs/experiments.jsonl"))
    args = cli.parse_args()
    return run(lambda: compare(args.root, args.splits, args.model, args.winner,
                              args.run_dir, args.winner_run_dir, args.cache, args.history))


if __name__ == "__main__":
    raise SystemExit(main())
