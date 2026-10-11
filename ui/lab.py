"""Agent Lab and Model Evolution views. Render recorded experiments only."""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import experiments as ex, theme
from ui.viz import _BASE, _GRID, MINT, MUTE, VIOLET

ROLE_TITLES = {"features": "Feature agent", "classifier": "Classifier agent", "robustness": "Robustness agent"}
TONE = {"ok": "ok", "failed": "bad", "timeout": "bad", "partial_failure": "bad", "rejected": "warn", "not_run": "warn"}
METRIC_NAMES = {"balanced_accuracy": "balanced accuracy", "f1_synthetic": "F1 (synthetic)", "roc_auc": "ROC AUC"}
PATH_HINT = "runs/experiments.jsonl"


def _esc(v) -> str:
    return html.escape("—" if v is None else str(v))


def _pills(e: dict) -> str:
    status = e.get("status") or "unknown"
    out = f'<span class="pill {TONE.get(status, "")}">{_esc(status).replace("_", " ")}</span>'
    if e.get("decision"):
        out += f'<span class="pill">{_esc(e["decision"])}</span>'
    if e.get("dry_run"):
        out += '<span class="pill warn">static proposal</span>'
    return out


def _proposal_kv(p: dict | None) -> str:
    if not p:
        return ""
    items = [("feature set", p.get("feature_set")), ("classifier", p.get("classifier"))]
    items += list((p.get("parameters") or {}).items())
    return '<dl class="kv">' + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in items) + "</dl>"


def card_html(e: dict) -> str:
    metrics = "".join(f'<div class="stat"><b>{v:.3f}</b><span>{METRIC_NAMES[m]}</span></div>'
                      for m in ex.METRICS if (v := ex.metric(e, m)) is not None)
    meta = " · ".join(x for x in (f'seed {e["seed"]}' if e.get("seed") is not None else "",
                                  f'{e["runtime_s"]:.1f}s' if isinstance(e.get("runtime_s"), (int, float)) else "") if x)
    bad = TONE.get(e.get("status")) == "bad"
    return (
        f'<div class="acard"><div class="card-h"><i></i>{_esc(ROLE_TITLES.get(e.get("role"), e.get("role")))}</div>'
        f"<div>{_pills(e)}</div>{_proposal_kv(e.get('proposal'))}"
        + (f'<div class="stats">{metrics}</div>' if metrics else "")
        + (f'<p class="note{" err" if bad else ""}">{_esc(e["reason"])}</p>' if e.get("reason") else "")
        + (f'<p class="note mono">{_esc(meta)}</p>' if meta else "")
        + "</div>"
    )


def _pct(v: float) -> str:
    return f"{v * 100:.4f}%"


def _pts(v: float) -> str:
    return f"{v * 100:+.4f} pts"


def comparison_html(c: dict) -> str:
    """Validation vs one-time final test, built only from the backend's saved reports."""
    rows, gains = "", {}
    for split, label in (("validation", "Validation"), ("test", "Final test (one-time)")):
        b, o = c[split]["baseline"], c[split]["optimized"]
        gains[split] = o["balanced_accuracy"] - b["balanced_accuracy"]
        rows += (f'<tr><td>{label}<span class="sub">{b.get("n_clips", "—")} clips</span></td>'
                 f'<td>{_pct(b["balanced_accuracy"])}</td><td>{_pct(o["balanced_accuracy"])}</td>'
                 f'<td class="{"up" if gains[split] > 0 else "down"}">{_pts(gains[split])}</td></tr>')
    who = f'round {_esc(c["winner"].get("round"))} {_esc(c["winner"].get("role"))} proposal'
    if gains["validation"] > 0 and gains["test"] <= 0:
        verdict = (f"The optimized configuration ({who}) improved validation balanced accuracy, "
                   "but did not improve final test performance. The saved baseline stays the default detector.")
    elif gains["validation"] > 0 and gains["test"] > 0:
        verdict = f"The optimized configuration ({who}) improved both validation and final test balanced accuracy."
    else:
        verdict = f"The optimized configuration ({who}) did not improve validation balanced accuracy."
    notes = "".join(f'<p class="note">{_esc(n)}</p>' for n in c["notes"])
    return (f'<p class="verdict-sm">{verdict}</p>'
            '<div class="tbl"><table class="cmp"><tr><th>Balanced accuracy</th><th>Baseline</th><th>Optimized</th>'
            f'<th>Change</th></tr>{rows}</table></div>{notes}')


def _comparison_card(run_id: str, key: str) -> None:
    c = ex.comparison(run_id)
    if c:
        with st.container(key=f"bezel_cmp_{key}"):
            theme.card_title("Validation vs final test")
            st.markdown(comparison_html(c), unsafe_allow_html=True)


def _empty(exp: ex.Experiments, what: str, key: str) -> bool:
    if exp.events:
        return False
    body = (f"{PATH_HINT} exists but has no readable events ({exp.skipped_lines} unreadable lines)." if exp.exists
            else f"No recorded experiments. {what} will appear after the agent loop writes {PATH_HINT}.")
    with st.container(key=f"bezel_empty_{key}"):
        theme.empty_state("Nothing recorded yet", body)
    return True


def _pick_run(exp: ex.Experiments, key: str) -> ex.Run:
    ids = exp.run_ids[::-1]  # newest first, but open on the run with saved final-test results if any
    default = next((i for i, rid in enumerate(ids) if ex.comparison(rid)), 0)
    rid = ids[0] if len(ids) == 1 else st.selectbox("Run", ids, index=default, key=f"run_{key}")
    r = ex.run(exp, rid)
    if r.dry_run:
        st.markdown('<span class="pill warn">Dry run: static proposals, not cloud agent output. '
                    "Any metrics shown were still measured locally.</span>", unsafe_allow_html=True)
    if exp.skipped_lines:
        st.markdown(f'<span class="pill warn">{exp.skipped_lines} unreadable lines skipped</span>', unsafe_allow_html=True)
    return r


def render_lab(exp: ex.Experiments) -> None:
    if _empty(exp, "Agent proposals and results", "lab"):
        return
    r = _pick_run(exp, "lab")
    last, fin = r.latest, r.finished
    with st.container(key="bezel_run"):
        theme.card_title(f"Run {r.run_id}")
        limits = last.get("limits") or {}
        st.markdown(
            '<div class="stats">'
            f'<div class="stat"><b>{_esc(last.get("llm_calls"))} / {_esc(limits.get("llm_calls", "—"))}</b><span>LLM calls</span></div>'
            f'<div class="stat"><b>{_esc(last.get("model_experiments"))} / {_esc(limits.get("experiments", "—"))}</b><span>Model fits</span></div>'
            "</div>"
            + (f'<p class="note">Baseline balanced accuracy {v:.3f}.</p>' if (v := ex.metric(r.baseline, "balanced_accuracy")) is not None else "")
            + ("" if fin else '<p class="note">This run has no finish event yet; it may still be running or may have stopped early.</p>')
            + (f'<p class="note">Selected configuration</p>{_proposal_kv(fin.get("best_proposal"))}' if fin and fin.get("best_proposal") else ""),
            unsafe_allow_html=True,
        )
    _comparison_card(r.run_id, "lab")
    for rnd in r.rounds:
        evs = sorted((e for e in r.experiments if e.get("round") == rnd),
                     key=lambda e: ex.ROLES.index(e["role"]) if e.get("role") in ex.ROLES else 9)
        with st.container(key=f"bezel_round{rnd}"):
            theme.card_title(f"Round {rnd} · {len(evs)} proposals")
            st.markdown(f'<div class="agrid">{"".join(card_html(e) for e in evs)}</div>', unsafe_allow_html=True)
    with st.container(key="bezel_history"):
        theme.card_title("Experiment history")
        rows = [{"round": e.get("round"), "role": e.get("role"), "status": e.get("status"), "decision": e.get("decision"),
                 **{METRIC_NAMES[m]: ex.metric(e, m) for m in ex.METRICS}, "seed": e.get("seed"),
                 "runtime s": e.get("runtime_s"), "reason": e.get("reason")} for e in r.experiments]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown('<p class="note">No experiment events in this run.</p>', unsafe_allow_html=True)


def evolution_fig(pts: list[dict], name: str, baseline: float | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[p["i"] + 1 for p in pts], y=[p["value"] for p in pts], mode="lines+markers",
                               line=dict(color=VIOLET, width=1.5), marker=dict(color=VIOLET, size=8),
                               text=[p["label"] for p in pts], hovertemplate="%{text}<br>%{y:.3f}<extra></extra>"))
    best = [p for p in pts if p.get("best")]
    fig.add_trace(go.Scatter(x=[p["i"] + 1 for p in best], y=[p["value"] for p in best], mode="markers",
                             marker=dict(color=MINT, size=16, symbol="star", line=dict(color="#050505", width=1)),
                             hoverinfo="skip"))
    if baseline is not None:
        fig.add_hline(y=baseline, line=dict(color=MUTE, dash="dot", width=1),
                      annotation_text="baseline", annotation_font_color=MUTE)
    fig.update_layout(**_BASE, height=320, showlegend=False,
                      xaxis=dict(title="experiment order in run", dtick=1, **_GRID),
                      yaxis=dict(title=METRIC_NAMES.get(name, name), range=[0, 1], **_GRID))
    return fig


def render_evolution(exp: ex.Experiments) -> None:
    if _empty(exp, "Validation metrics across experiments", "evo"):
        return
    r = _pick_run(exp, "evo")
    names = ex.available_metrics(r.experiments)
    with st.container(key="bezel_evo_chart"):
        theme.card_title("Validation metric across executed experiments")
        if not names:
            st.markdown('<p class="note">No experiment in this run has a measured validation metric, so there is nothing to plot.</p>',
                        unsafe_allow_html=True)
        else:
            name = st.radio("Metric", names, horizontal=True, label_visibility="collapsed",
                            format_func=lambda m: METRIC_NAMES[m])
            pts = ex.trajectory(r.experiments, name)
            st.plotly_chart(evolution_fig(pts, name, ex.metric(r.baseline, name)), use_container_width=True,
                            config={"displayModeBar": False})
            best = next(p for p in pts if p.get("best"))
            st.markdown(f'<p class="note">Star: highest measured value, {best["value"]:.3f} ({_esc(best["label"])}). '
                        "Fixed validation partition only; this is not a final test result.</p>", unsafe_allow_html=True)
    _comparison_card(r.run_id, "evo")
    with st.container(key="bezel_evo_out"):
        theme.card_title("Outcomes, including failures")
        counts = ex.outcome_counts(r.experiments)
        st.markdown("".join(f'<span class="pill {TONE.get(k, "")}">{html.escape(k).replace("_", " ")} · {v}</span>'
                            for k, v in counts.items())
                    + '<p class="note">Failed, rejected and unselected experiments stay in the record. '
                      "Only measured validation gains count as improvement.</p>", unsafe_allow_html=True)
