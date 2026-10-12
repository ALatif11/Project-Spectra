"""Evidence layer: read, verify and normalize the project's saved reports (read-only).

Retrieval and validation live here; rendering lives in ui/pages.py. Every value carries its dataset,
partition and source artifact. Missing, corrupt or mismatched sources produce scoped "unavailable"
sections; nothing falls back to a stale or invented number.

When local reports are absent, sections may come from the sanitized export (ui/assets/project_evidence.json,
built by `python -m ui.export_evidence`). Those are labeled provenance="export" and never establish which
checkpoint is active: that always comes from hashing the real local file.
"""

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = ROOT / "ui" / "assets" / "project_evidence.json"
SCHEMA_VERSION = 1

GS = "Gary Stafford deepfake-audio-detection"
ITW = "In-the-Wild (SpeechAntiSpoofingBenchmarks/InTheWild)"
SOURCES = {
    # original development (Gary Stafford)
    "gs_val": "runs/baseline/validation_metrics.json",
    "gs_test": "runs/baseline/test_metrics.json",
    "gs_history": "runs/experiments.jsonl",
    "gs_opt_val": "runs/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e/validation_metrics.json",
    "gs_opt_test": "runs/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e/test_metrics.json",
    "gs_opt_meta": "runs/optimized-v1-ba52f0c8dafd4a1880b60d5c557ec35e/checkpoint_metadata.json",
    # In-the-Wild data expansion
    "itw_prep": "data/inthewild/experiment-12000-v1/preparation.json",
    "itw_cmp": "data/inthewild/experiment-12000-v1/comparison.json",
    "itw_verify": "data/inthewild/experiment-12000-v1/verification.json",
    # final evaluation
    "final_freeze": "runs/inthewild-final-20261011/freeze.json",
    "final_test": "runs/inthewild-final-20261011/test_metrics.json",
    "final_preflight": "runs/inthewild-final-20261011/test_preflight.json",
    "final_canonical": "data/inthewild/experiment-12000-v1/sealed/test_metrics.json",
    # In-the-Wild research sessions
    "itw1_events": "runs/inthewild-luminal-20261011-v1/experiments.jsonl",
    "itw1_summary": "runs/inthewild-luminal-20261011-v1/summary.json",
    "itw2_history": "runs/inthewild-research-preparation-20261011-v2/research-history.jsonl",
    "itw2_result": "runs/inthewild-research-preparation-20261011-v2/research-result.json",
}
METRIC_KEYS = ("balanced_accuracy", "f1_synthetic", "roc_auc", "accuracy", "precision_synthetic",
               "recall_synthetic", "false_positive_rate", "false_negative_rate", "threshold")
ROLE_NAMES = {"features": "Feature Researcher", "classifier": "Classifier Researcher",
              "robustness": "Robustness Researcher"}


# ---------- primitives ----------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(v, lo: float | None = None, hi: float | None = None) -> float | None:
    """A real finite number within [lo, hi], else None (preserves null/undefined)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return float(v)


def read(key: str, root: Path | None = None) -> dict:
    """{key, path, sha256, data, error}. JSONL becomes a list; unreadable lines are counted, not guessed."""
    root = root or ROOT
    rel = SOURCES[key]
    path = root / rel
    out = {"key": key, "path": rel, "sha256": None, "data": None, "error": None, "skipped_lines": 0}
    if not path.is_file():
        out["error"] = "missing"
        return out
    try:
        out["sha256"] = sha256(path)
        text = path.read_text(encoding="utf-8")
        if rel.endswith(".jsonl"):
            rows = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    out["skipped_lines"] += 1
            out["data"] = rows
        else:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            out["data"] = data
    except (OSError, ValueError) as exc:
        out["error"] = f"unreadable ({type(exc).__name__})"
        out["data"] = None
    return out


def signature(model_path: Path | None, root: Path | None = None) -> tuple:
    """Cache key covering every evidence source, the export and the selected checkpoint."""
    root = root or ROOT
    paths = [root / rel for rel in SOURCES.values()] + [EXPORT_PATH]
    if model_path is not None:
        paths.append(Path(model_path))
    sig = []
    for p in paths:
        try:
            st = p.stat()
            sig.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((str(p), None, None))
    return tuple(sig)


def metrics(raw) -> dict | None:
    """Validated metric dict: rates in [0,1], counts as ints, 2x2 matrix consistent with n."""
    if not isinstance(raw, dict):
        return None
    out = {k: finite(raw.get(k), 0, 1) for k in METRIC_KEYS if k in raw}
    n = raw.get("n", raw.get("n_clips"))
    out["n"] = n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None
    cm = raw.get("confusion_matrix_real_fake")
    ok = (isinstance(cm, list) and len(cm) == 2 and all(isinstance(r, list) and len(r) == 2 for r in cm)
          and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for r in cm for x in r))
    out["confusion"] = cm if ok and (out["n"] is None or sum(map(sum, cm)) == out["n"]) else None
    if out["confusion"] is not None:
        (tn, fp), (fn, tp) = out["confusion"]
        out["derived_accuracy"] = (tn + tp) / (tn + fp + fn + tp)
    return out


def section(status: str, sources: list[dict], **data) -> dict:
    return {"status": status, "provenance": "local",
            "sources": [{"path": s["path"], "sha256": s["sha256"], "error": s["error"]} for s in sources], **data}


# ---------- active checkpoint ----------

def dataset_of(artifact: dict) -> dict:
    """Dataset identity, handling both shapes: nested validation (baseline) vs top-level provenance."""
    prov = artifact.get("provenance") if isinstance(artifact.get("provenance"), dict) else {}
    val = artifact.get("validation") if isinstance(artifact.get("validation"), dict) else {}
    repo = prov.get("repo") or prov.get("source") or val.get("source")
    rev = prov.get("revision") or val.get("revision")
    name = ITW if repo == "SpeechAntiSpoofingBenchmarks/InTheWild" else GS if repo == "garystafford/deepfake-audio-detection" else repo
    return {"repo": repo, "revision": rev, "name": name}


def active_model(path: Path | None = None) -> dict:
    """Identity of the checkpoint predict_audio will load. Never inferred from file names or recency."""
    from ui.detector import _model_path
    path = Path(path) if path is not None else _model_path()
    try:
        shown = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = path.name
    out = {"path": shown, "override": bool(os.environ.get("ECHOPROOF_MODEL_PATH")), "exists": path.is_file(),
           "sha256": None, "error": None}
    if not out["exists"]:
        out["error"] = "Checkpoint file not found. The app does not switch to another model."
        return out
    out["sha256"] = sha256(path)
    try:
        from src.model import load_model
        art = load_model(path)
    except Exception as exc:  # incompatible or untrusted artifact
        out["error"] = f"Checkpoint could not be loaded ({type(exc).__name__}). The app does not switch to another model."
        return out
    steps = getattr(art["pipeline"], "named_steps", {})
    clf = steps.get("classifier")
    fs = art.get("feature_settings") or {}
    out.update(version=art.get("model_version"), threshold=finite(art.get("threshold"), 0, 1),
               classifier=type(clf).__name__ if clf is not None else None,
               scaler=type(steps.get("scaler")).__name__ if steps.get("scaler") is not None else None,
               params={k: v for k, v in (clf.get_params() if clf is not None else {}).items()
                       if k in ("C", "class_weight", "solver", "max_iter", "random_state")},
               n_features=len(fs.get("feature_names") or []) or None, feature_settings=fs,
               dataset=dataset_of(art), splits_sha256=art.get("splits_sha256"))
    return out


# ---------- sections ----------

def original(root: Path | None = None) -> dict:
    """Gary Stafford development: baseline vs agent-optimized configuration, validation and test."""
    root = root or ROOT
    src = [read(k, root) for k in ("gs_val", "gs_test", "gs_opt_val", "gs_opt_test", "gs_opt_meta")]
    if any(s["error"] for s in src):
        return section("unavailable", src, reason="One or more original reports are missing or unreadable.")
    val, test, ov, ot, meta = (s["data"] for s in src)
    rows = {"baseline": {"validation": metrics(val.get("metrics")), "test": metrics(test.get("metrics")),
                         "sha256": test.get("model_sha256")},
            "optimized": {"validation": metrics(ov.get("metrics")), "test": metrics(ot.get("metrics")),
                          "sha256": meta.get("model_sha256"),
                          "config": meta.get("training_configuration"),
                          "provenance": {k: (meta.get("provenance") or {}).get(k) for k in ("run_id", "round", "role")}}}
    return section("ok", src, dataset=GS, revision=val.get("revision"), rows=rows,
                   n_train=ov.get("n_train") if isinstance(ov.get("n_train"), int) else None,
                   threshold_policy="Validation-selected threshold, frozen before the one-time test.")


def expansion(root: Path | None = None) -> dict:
    """In-the-Wild data expansion: split counts and same-validation comparison, digest-checked against the freeze."""
    root = root or ROOT
    prep, cmp_, ver, freeze = (read(k, root) for k in ("itw_prep", "itw_cmp", "itw_verify", "final_freeze"))
    warnings = []
    if freeze["data"]:
        protected = {Path(k).name: v for k, v in (freeze["data"].get("protected_sha256") or {}).items()
                     if "experiment-12000" in k}
        for s, name in ((prep, "preparation.json"), (cmp_, "comparison.json")):
            if s["sha256"] and protected.get(name) and s["sha256"] != protected[name]:
                warnings.append(f"{name} does not match the digest recorded in the freeze record.")
                s["error"] = "digest mismatch"
    counts = None
    if prep["data"] and not prep["error"]:
        p = prep["data"]

        def part(d):
            c = (d or {}).get("classes") or {}
            return {"n": d.get("n"), "genuine": c.get("0"), "synthetic": c.get("1"), "speakers": d.get("speakers")}
        counts = {"selected": part(p.get("selected")), **{k: part((p.get("partitions") or {}).get(k)) for k in ("train", "validation", "test")},
                  "revision": p.get("revision"), "repo": p.get("repo"), "limitations": p.get("limitations")}
    comparison = None
    if cmp_["data"] and not cmp_["error"]:
        c = cmp_["data"]
        comparison = {"partition": c.get("partition"), "baseline": metrics(c.get("baseline")),
                      "candidate": metrics(c.get("candidate")), "baseline_sha256": c.get("baseline_sha256"),
                      "luminal_calls": c.get("luminal_calls"), "original_training_included": c.get("original_training_included"),
                      "training_clips": c.get("training_clips"), "limitations": c.get("limitations")}
    diagnostic = None
    if ver["data"] and not ver["error"]:
        inf = ver["data"].get("inference") or {}
        diagnostic = {"baseline_score": finite((inf.get("baseline") or {}).get("synthetic_score"), 0, 1),
                      "candidate_score": finite((inf.get("candidate") or {}).get("synthetic_score"), 0, 1),
                      "candidate_sha256": ver["data"].get("candidate_model_sha256"),
                      "label_source": "previous audit/user attestation"}
    status = "ok" if counts and comparison else "partial" if counts or comparison or diagnostic else "unavailable"
    return section(status, [prep, cmp_, ver], dataset=ITW, counts=counts, comparison=comparison,
                   diagnostic=diagnostic, warnings=warnings)


def final(root: Path | None = None) -> dict:
    """Frozen-model identity and the one-time held-out test, cross-checked between freeze and test report."""
    root = root or ROOT
    freeze, test, pre = (read(k, root) for k in ("final_freeze", "final_test", "final_preflight"))
    if freeze["error"] or test["error"]:
        return section("unavailable", [freeze, test, pre], reason="Final evaluation reports are missing or unreadable.")
    fz, t = freeze["data"], test["data"]
    warnings = []
    if t.get("model_sha256") != fz.get("model_sha256"):
        warnings.append("The test report's model digest does not match the freeze record.")
    m = metrics(t.get("metrics"))
    if m is None or m.get("confusion") is None:
        warnings.append("The final confusion matrix is missing or inconsistent with n.")
    if t.get("threshold_tuned_on_test") is not False:
        warnings.append("The report does not confirm the threshold was fixed before testing.")
    if any(v is None for k, v in (m or {}).items() if k in METRIC_KEYS):
        warnings.append("One or more final metrics are not finite numbers in [0, 1].")
    prov = t.get("provenance") or {}
    dist = (pre["data"] or {}).get("test_distribution") or {}
    return section("invalid" if warnings else "ok", [freeze, test, pre], dataset=ITW,
                   partition="local speaker-disjoint held-out test", model_sha256=fz.get("model_sha256"),
                   baseline_sha256=fz.get("baseline_sha256"), threshold=finite(fz.get("threshold"), 0, 1),
                   feature_settings=fz.get("feature_settings"), repo=prov.get("repo"), revision=prov.get("revision"),
                   model_version=t.get("model_version"), metrics=m, validation=metrics(fz.get("validation")),
                   speakers=dist.get("speakers"), classes=dist.get("classes"),
                   speakers_disjoint=(pre["data"] or {}).get("speakers_disjoint"),
                   decision_policy=t.get("decision_policy"), luminal_calls=t.get("luminal_calls"),
                   model_fits=t.get("model_fits"), warnings=warnings)


def attribution(active: dict, fin: dict) -> dict:
    """Is the final evaluation about the checkpoint that is actually loaded?"""
    if fin.get("status") not in ("ok",) or not active.get("sha256"):
        return {"active": False, "reasons": ["No verified final evaluation or no loaded checkpoint."]}
    reasons = []
    if active["sha256"] != fin.get("model_sha256"):
        reasons.append("Loaded checkpoint digest differs from the evaluated model.")
    if active.get("threshold") is not None and fin.get("threshold") is not None and abs(active["threshold"] - fin["threshold"]) > 1e-12:
        reasons.append("Threshold differs.")
    if fin.get("feature_settings") and active.get("feature_settings") and fin["feature_settings"] != active["feature_settings"]:
        reasons.append("Feature settings differ.")
    ds = active.get("dataset") or {}
    if fin.get("revision") and ds.get("revision") and fin["revision"] != ds["revision"]:
        reasons.append("Dataset revision differs.")
    return {"active": not reasons, "reasons": reasons}


def role_of(active: dict, fin: dict) -> str:
    """Name the loaded checkpoint by digest only."""
    sha = active.get("sha256")
    if not sha:
        return "missing"
    if sha == fin.get("model_sha256"):
        return "final"
    if sha == fin.get("baseline_sha256"):
        return "rollback" if "rollback" in active.get("path", "") else "baseline"
    return "other"


# ---------- research sessions (three formats) ----------

def _proposal_config(p) -> dict | None:
    if not isinstance(p, dict):
        return None
    return {"feature_set": p.get("feature_set"), "classifier": p.get("classifier"), "parameters": p.get("parameters") or {}}


def session_from_v1(events: list[dict], *, dataset: str, protocol: str, run_id: str | None = None) -> dict:
    """Original and first In-the-Wild sessions (schema_version 1). Counters are cumulative: take the last value, never sum."""
    evs = [e for e in events if run_id is None or e.get("run_id") == run_id]
    if not evs:
        return {}
    last = evs[-1]
    fin = next((e for e in reversed(evs) if e.get("event") == "run_finished"), None)
    calls = max((e.get("llm_calls") for e in evs if isinstance(e.get("llm_calls"), int)), default=0)
    fits = max((e.get("model_experiments") for e in evs if isinstance(e.get("model_experiments"), int)), default=0)
    dry = any(e.get("dry_run") for e in evs)
    kind = "dry_run" if dry else "config_failure" if calls == 0 else "live"
    base = next((e for e in evs if e.get("event") == "baseline"), None)
    proposals = []
    for e in evs:
        if e.get("event") != "experiment":
            continue
        m = metrics(e.get("metrics")) if e.get("status") == "ok" else None
        proposals.append({"round": e.get("round"), "role": e.get("role"), "hypothesis": None,
                          "config": _proposal_config(e.get("proposal")), "status": e.get("status"),
                          "decision": e.get("decision"), "reason": e.get("reason"), "metrics": m,
                          "folds": None, "objective": None})
    return {"run_id": last.get("run_id"), "dataset": dataset, "protocol": protocol, "kind": kind, "format": "v1",
            "calls": calls, "fits": fits, "rounds": len({p["round"] for p in proposals if p["round"] is not None}),
            "status": (fin or {}).get("status"), "stop_reason": (fin or {}).get("reason"), "finished": fin is not None,
            "limits": last.get("limits") if isinstance(last.get("limits"), dict) else None,
            "baseline": metrics((base or {}).get("metrics")) if base and base.get("status") == "ok" else None,
            "incumbent_ba": finite(last.get("best_balanced_accuracy"), 0, 1),
            "selected": (fin or last).get("best_proposal"), "proposals": proposals,
            "failures": sum(1 for p in proposals if p["status"] not in ("ok",))}


def session_from_v2(history: list[dict]) -> dict:
    """Expanded research ledger (schema_version 2). Parent run_finished and budget snapshots are authoritative;
    fold folds are parts of one configuration, never separate proposals."""
    if not history:
        return {}
    fin = next((e for e in reversed(history) if e.get("event") == "run_finished"), None)
    started = next((e for e in history if e.get("event") == "run_started"), {})
    budget = (fin or history[-1]).get("budget") or {}
    base = next((e for e in history if e.get("event") == "baseline"), None)
    base_dev = (base or {}).get("development") or {}
    proposals = []
    for e in history:
        if e.get("event") != "experiment":
            continue
        mr = e.get("measured_results") if e.get("status") == "ok" else None
        folds = [metrics(f) for f in (mr or {}).get("fold_metrics") or []] or None
        obj = (mr or {}).get("objective") or {}
        proposals.append({"round": e.get("round"), "role": e.get("agent_role"), "hypothesis": e.get("hypothesis"),
                          "config": _proposal_config(e.get("proposal")), "status": e.get("status"),
                          "decision": e.get("decision"), "reason": e.get("reason"),
                          "metrics": metrics((mr or {}).get("metrics")), "folds": folds,
                          "objective": {k: finite(obj.get(k)) for k in ("score", "mean_fold_balanced_accuracy",
                                                                       "fold_balanced_accuracy_std", "speaker_macro_balanced_accuracy")} if obj else None})
    limits = budget.get("limits") if isinstance(budget.get("limits"), dict) else (started.get("policy") or None)
    bobj = base_dev.get("objective") or {}
    return {"run_id": (fin or history[-1]).get("run_id"), "dataset": ITW,
            "protocol": "Training-speaker development folds (3 speaker folds, inner calibration)",
            "kind": "live" if isinstance(budget.get("api_calls"), int) and budget["api_calls"] > 0 else "config_failure",
            "format": "v2", "calls": budget.get("api_calls"), "fits": budget.get("ml_fits_completed"),
            "fit_attempts": budget.get("ml_fit_attempts"), "rounds": (fin or {}).get("completed_rounds"),
            "configurations": (fin or {}).get("completed_configurations"),
            "status": (fin or {}).get("status"), "stop_reason": (fin or {}).get("reason"), "finished": fin is not None,
            "limits": limits, "estimated_cost_usd": finite(budget.get("estimated_cost_usd"), 0),
            "baseline": metrics(base_dev.get("metrics")),
            "baseline_folds": [metrics(f) for f in base_dev.get("fold_metrics") or []] or None,
            "baseline_objective": {k: finite(bobj.get(k)) for k in ("score", "mean_fold_balanced_accuracy",
                                                                   "fold_balanced_accuracy_std", "speaker_macro_balanced_accuracy")} if bobj else None,
            "final_validation": (fin or {}).get("final_validation"),
            "selected": (fin or {}).get("best_development_configuration"), "proposals": proposals,
            "failures": sum(1 for p in proposals if p["status"] != "ok")}


def sessions(root: Path | None = None) -> dict:
    root = root or ROOT
    gs, itw1, itw2 = read("gs_history", root), read("itw1_events", root), read("itw2_history", root)
    out = []
    if gs["data"]:
        for rid in dict.fromkeys(e.get("run_id") for e in gs["data"] if e.get("run_id")):
            s = session_from_v1(gs["data"], dataset=GS, protocol="Fixed validation split (250 clips)", run_id=rid)
            if s:
                s["source"] = gs["path"]
                out.append(s)
    if itw1["data"]:
        for rid in dict.fromkeys(e.get("run_id") for e in itw1["data"] if e.get("run_id")):
            s = session_from_v1(itw1["data"], dataset=ITW, protocol="Fixed speaker-disjoint validation (1,800 recordings)", run_id=rid)
            if s:
                s["source"] = itw1["path"]
                out.append(s)
    if itw2["data"]:
        s = session_from_v2(itw2["data"])
        if s:
            s["source"] = itw2["path"]
            out.append(s)
    missing = [r["path"] for r in (itw1, itw2) if r["error"] == "missing"]
    return section("ok" if out else "unavailable", [gs, itw1, itw2], sessions=out, missing=missing,
                   skipped_lines=sum(r["skipped_lines"] for r in (gs, itw1, itw2)))


# ---------- assembly with optional sanitized export ----------

def load_export(path: Path | None = None) -> dict | None:
    path = path or EXPORT_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("schema_version") == SCHEMA_VERSION else None
    except (OSError, ValueError):
        return None


def evidence(root: Path | None = None, export_path: Path | None = None) -> dict:
    """All report sections. A section missing locally may be filled from the sanitized export, labeled as such."""
    root = root or ROOT
    ev = {"original": original(root), "expansion": expansion(root), "final": final(root), "sessions": sessions(root)}
    exp = load_export(export_path)
    if exp:
        for name, sec in ev.items():
            other = (exp.get("sections") or {}).get(name)
            if sec["status"] == "unavailable" and other and other.get("status") in ("ok", "partial"):
                ev[name] = {**other, "provenance": "export", "export_generated_at": exp.get("generated_at")}
        # sessions are merged per run: local runs win, missing runs may come from the export
        local_ids = {s.get("run_id") for s in ev["sessions"].get("sessions", []) if ev["sessions"]["provenance"] == "local"}
        extra = [{**s, "provenance": "export"} for s in ((exp.get("sections") or {}).get("sessions") or {}).get("sessions", [])
                 if s.get("run_id") not in local_ids]
        if extra and ev["sessions"]["provenance"] == "local":
            ev["sessions"] = {**ev["sessions"], "status": "ok", "sessions": ev["sessions"].get("sessions", []) + extra,
                              "export_generated_at": exp.get("generated_at")}
    return ev
