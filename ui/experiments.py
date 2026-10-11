"""Read-only loader for recorded agent experiments (runs/experiments.jsonl).

Follows agents/event.schema.json v1 (backend-owned): one event per line, grouped by
`run_id`; `event` is run_started | baseline | experiment | run_finished. Metrics are
plotted only from status="ok" events. Nothing here is ever invented or filled in.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "runs" / "experiments.jsonl"
METRICS = ("balanced_accuracy", "f1_synthetic", "roc_auc")
ROLES = ("features", "classifier", "robustness")


@dataclass
class Experiments:
    events: list[dict] = field(default_factory=list)
    skipped_lines: int = 0
    exists: bool = False

    @property
    def run_ids(self) -> list[str]:
        """Run ids in first-seen file order (the file is append-only)."""
        return list(dict.fromkeys(e["run_id"] for e in self.events))


@dataclass
class Run:
    run_id: str
    events: list[dict]

    def _first(self, kind: str, last: bool = False) -> dict | None:
        found = [e for e in self.events if e.get("event") == kind]
        return (found[-1] if last else found[0]) if found else None

    @property
    def experiments(self) -> list[dict]:
        return [e for e in self.events if e.get("event") == "experiment"]

    @property
    def baseline(self) -> dict | None:
        return self._first("baseline")

    @property
    def finished(self) -> dict | None:
        return self._first("run_finished", last=True)

    @property
    def latest(self) -> dict:
        return self.events[-1]

    @property
    def dry_run(self) -> bool:
        return any(e.get("dry_run") for e in self.events)

    @property
    def rounds(self) -> list[int]:
        return sorted({e["round"] for e in self.experiments if isinstance(e.get("round"), int)})


def load(path: Path = PATH) -> Experiments:
    """Parse the file; malformed lines are counted, never guessed at."""
    if not path.is_file():
        return Experiments()
    out = Experiments(exists=True)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            e = None
        if isinstance(e, dict) and isinstance(e.get("run_id"), str) and isinstance(e.get("event"), str):
            out.events.append(e)
        else:
            out.skipped_lines += 1
    return out


def run(exp: Experiments, run_id: str) -> Run:
    return Run(run_id, [e for e in exp.events if e["run_id"] == run_id])


def metric(e: dict | None, name: str) -> float | None:
    """A measured value, only from a successful event."""
    if not e or e.get("status") != "ok":
        return None
    v = (e.get("metrics") or {}).get(name)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def available_metrics(events: list[dict]) -> list[str]:
    return [m for m in METRICS if any(metric(e, m) is not None for e in events)]


def trajectory(events: list[dict], name: str) -> list[dict]:
    """Measured points in file order, with `best` marking the highest value (first wins ties)."""
    pts = [{"i": i, "label": f'round {e.get("round", "?")} · {e.get("role") or "?"}', "value": metric(e, name)}
           for i, e in enumerate(events) if metric(e, name) is not None]
    if pts:
        top = max(range(len(pts)), key=lambda k: (pts[k]["value"], -k))
        for k, p in enumerate(pts):
            p["best"] = k == top
    return pts


def outcome_counts(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        key = e.get("status") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts
