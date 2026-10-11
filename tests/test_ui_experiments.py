import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ui import experiments as ex
from ui.lab import card_html, evolution_fig

ROOT = Path(__file__).parents[1]


def _ev(**kw):
    """Test-only events in the agents/event.schema.json v1 shape. dry_run=True: not real agent output."""
    base = json.loads((ROOT / "agents" / "example_event.json").read_text(encoding="utf-8"))
    base.update(run_id="t1", dry_run=True, **kw)
    return base


def _m(bal, f1=0.5):
    return {"threshold": 0.5, "accuracy": bal, "balanced_accuracy": bal, "f1_synthetic": f1, "roc_auc": 0.5,
            "n_clips": 10, "confusion_matrix_real_fake": [[4, 1], [1, 4]]}


FIXTURE = [
    _ev(event="run_started", round=None, role=None, proposal=None),
    _ev(event="baseline", round=None, role=None, proposal=None, status="ok", metrics=_m(0.70)),
    _ev(round=1, role="features", status="ok", decision="accepted", metrics=_m(0.80), reason="<b>gain</b>"),
    _ev(round=1, role="classifier", status="timeout", reason="fit exceeded limit"),
    _ev(round=2, role="robustness", status="ok", decision="rejected", metrics=_m(0.78, 0.7), reason="no improvement"),
    _ev(event="run_finished", round=None, role=None, proposal=None, status="ok", best_balanced_accuracy=0.8),
]


def _write(tmp_path, lines):
    p = tmp_path / "experiments.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_missing_file_is_empty(tmp_path):
    exp = ex.load(tmp_path / "nope.jsonl")
    assert not exp.exists and exp.events == []


def test_load_groups_runs_and_skips_bad_lines(tmp_path):
    other = dict(FIXTURE[2], run_id="t2")
    exp = ex.load(_write(tmp_path, [json.dumps(e) for e in FIXTURE + [other]] + ["{broken", "[1]", '{"run_id": 1}']))
    assert exp.skipped_lines == 3 and exp.run_ids == ["t1", "t2"]
    r = ex.run(exp, "t1")
    assert len(r.experiments) == 3 and r.rounds == [1, 2] and r.dry_run and r.finished and r.baseline
    assert ex.outcome_counts(r.experiments) == {"ok": 2, "timeout": 1}


def test_metrics_only_from_ok_events():
    assert ex.metric(FIXTURE[3], "balanced_accuracy") is None
    assert ex.metric(dict(FIXTURE[3], metrics=_m(0.9)), "balanced_accuracy") is None  # not ok -> never plotted
    pts = ex.trajectory(FIXTURE[2:5], "balanced_accuracy")
    assert [p["value"] for p in pts] == [0.80, 0.78] and [p["best"] for p in pts] == [True, False]
    assert ex.available_metrics([FIXTURE[3]]) == []
    assert evolution_fig(pts, "balanced_accuracy", 0.7).data


def test_card_escapes_and_labels_static_proposal():
    out = card_html(FIXTURE[2])
    assert "&lt;b&gt;" in out and "static proposal" in out and "0.800" in out and "mfcc" in out
    assert "fit exceeded limit" in card_html(FIXTURE[3]) and "err" in card_html(FIXTURE[3])


def test_app_tabs_render_with_and_without_file(monkeypatch, tmp_path):
    for lines in (None, [json.dumps(e) for e in FIXTURE]):
        monkeypatch.setattr(ex, "PATH", _write(tmp_path, lines) if lines else tmp_path / "missing.jsonl")
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not at.exception
