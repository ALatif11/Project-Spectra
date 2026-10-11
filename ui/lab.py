"""Agent Lab and Model Evolution views. Render recorded experiments only."""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import experiments as ex, theme
from ui.theme import TOKENS, term, tip
from ui.viz import _BASE, _GRID, INK, MUTE, SIGNAL

ROLE_TITLES = {"features": "Feature agent", "classifier": "Classifier agent", "robustness": "Robustness agent"}
ROLE_TIPS = {
    "features": "Chooses which measurements of the sound the model uses, for example only voice-timbre measurements.",
    "classifier": "Chooses the type of model and its settings, such as how strongly it avoids memorizing the training clips.",
    "robustness": "Tries to make the model lean less on things like recording loudness, so it holds up on other recordings.",
}
TONE = {"ok": "ok", "failed": "bad", "timeout": "bad", "partial_failure": "bad", "rejected": "warn", "not_run": "warn"}
STATUS_TIPS = {
    "ok": "The suggestion was trained and measured on this computer.",
    "failed": "Something went wrong, so nothing was trained for this suggestion.",
    "timeout": "Training took longer than the time limit and was stopped.",
    "partial_failure": "Part of this step failed; see the reason below.",
    "rejected": "Refused before training: outside the allowed settings, or a repeat of an earlier suggestion.",
    "not_run": "Recorded but never executed.",
}
DECISION_TIPS = {
    "accepted": "It measured better than the best result so far, so it became the new best.",
    "rejected": "It did not beat the best result so far, so it was not kept.",
}
METRIC_NAMES = {"balanced_accuracy": "balanced accuracy", "f1_synthetic": "F1 (synthetic)", "roc_auc": "ROC AUC"}
METRIC_TIPS = {
    "balanced_accuracy": "How often the model is right, averaged over real clips and fake clips so neither side "
                         "dominates. 50% is a coin flip, 100% is perfect.",
    "f1_synthetic": "A 0 to 1 score for catching fakes without crying wolf on real voices. Higher is better.",
    "roc_auc": "How well the scores separate real from fake overall, before any threshold is chosen. "
               "0.5 is guessing, 1.0 is perfect.",
}
VALIDATION_TIP = ("Practice exam: clips used to compare suggestions and pick a winner. Checking many suggestions "
                  "against it can flatter the winner.")
TEST_TIP = "Sealed exam: clips kept untouched until the end and scored once. The fairest measure of the two."
REASON_TEXT = {
    "no_validation_balanced_accuracy_improvement": "Did not beat the best practice-exam score so far.",
    "validation_balanced_accuracy_improved": "Beat the best practice-exam score so far, so it became the new best.",
}
RUN_TIP = "Each run is one session of the agent loop. Dry runs use pre-written suggestions instead of asking the AI."
PATH_HINT = "runs/experiments.jsonl"


def _esc(v) -> str:
    return html.escape("—" if v is None else str(v))


def _chip(text: str, cls: str = "", explain: str | None = None) -> str:
    t = f' tabindex="0" data-tip="{html.escape(explain, quote=True)}"' if explain else ""
    return f'<span class="chip {cls}"{t}>{_esc(text)}</span>'


def _pills(e: dict) -> str:
    status = e.get("status") or "unknown"
    out = _chip(status.replace("_", " "), TONE.get(status, ""), STATUS_TIPS.get(status))
    if e.get("decision"):
        out += _chip(e["decision"], "plain", DECISION_TIPS.get(e["decision"]))
    if e.get("dry_run"):
        out += _chip("static proposal", "warn", "A pre-written suggestion used to test the loop, not one from the AI.")
    return out


def _proposal_kv(p: dict | None) -> str:
    if not p:
        return ""
    items = [("feature set", p.get("feature_set")), ("classifier", p.get("classifier"))]
    items += list((p.get("parameters") or {}).items())
    return '<dl class="kv">' + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in items) + "</dl>"


def card_html(e: dict) -> str:
    metrics = "".join(f'<div class="stat"><b>{v:.3f}</b><span>{term(METRIC_NAMES[m], METRIC_TIPS[m])}</span></div>'
                      for m in ex.METRICS if (v := ex.metric(e, m)) is not None)
    meta = " · ".join(x for x in (f'seed {e["seed"]}' if e.get("seed") is not None else "",
                                  f'{e["runtime_s"]:.1f}s' if isinstance(e.get("runtime_s"), (int, float)) else "") if x)
    bad = TONE.get(e.get("status")) == "bad"
    role = e.get("role")
    return (
        f'<div class="acard"><h4>{_esc(ROLE_TITLES.get(role, role))} {tip(ROLE_TIPS[role]) if role in ROLE_TIPS else ""}</h4>'
        f"<div>{_pills(e)}</div>{_proposal_kv(e.get('proposal'))}"
        + (f'<div class="stats">{metrics}</div>' if metrics else "")
        + (f'<p class="note{" err" if bad else ""}">{_esc(REASON_TEXT.get(e["reason"], e["reason"]))}</p>'
           if e.get("reason") else "")
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
    for split, label, explain in (("validation", "Practice exam", VALIDATION_TIP), ("test", "Final test", TEST_TIP)):
        b, o = c[split]["baseline"], c[split]["optimized"]
        gains[split] = g = o["balanced_accuracy"] - b["balanced_accuracy"]
        bars = "".join(
            f'<div class="bar {cls}"><span>{name}</span><i><b style="width:{m["balanced_accuracy"] * 100:.2f}%"></b></i>'
            f'<em>{_pct(m["balanced_accuracy"])}</em></div>' for cls, name, m in (("base", "Baseline", b), ("opt", "Optimized", o)))
        rows += (f'<div class="exam-row"><div class="lbl"><b>{term(label, explain, "below")}</b>'
                 f'<span>{"validation" if split == "validation" else "one-time"} · {b.get("n_clips", "—")} clips · '
                 f'<span class="delta {"up" if g > 0 else "down"}">{_pts(g)}</span></span></div>'
                 f'<div class="bars">{bars}</div></div>')
    who = f'round {_esc(c["winner"].get("round"))} {_esc(c["winner"].get("role"))} proposal'
    if gains["validation"] > 0 and gains["test"] <= 0:
        verdict = (f"The optimized configuration ({who}) improved validation balanced accuracy, "
                   "but did not improve final test performance. The saved baseline stays the default detector.")
    elif gains["validation"] > 0 and gains["test"] > 0:
        verdict = f"The optimized configuration ({who}) improved both validation and final test balanced accuracy."
    else:
        verdict = f"The optimized configuration ({who}) did not improve validation balanced accuracy."
    notes = "".join(f'<p class="note">{_esc(n)}</p>' for n in c["notes"])
    return (f'<p class="verdict-sm">{verdict}</p><div class="exam">{rows}</div>'
            f'<p class="note">Bars run from 0% to 100% of {term("balanced accuracy", METRIC_TIPS["balanced_accuracy"])}, '
            f"so you can see how small the differences are.</p>{notes}")


def _comparison_card(run_id: str, key: str) -> None:
    c = ex.comparison(run_id)
    if c:
        with st.container(key=f"card_cmp_{key}"):
            theme.card_title("Did the agents' pick hold up?", "Compares the original model with the agents' favourite "
                             "on the practice exam they optimized for and on the sealed final test.")
            st.markdown(comparison_html(c), unsafe_allow_html=True)


def _empty(exp: ex.Experiments, what: str, key: str) -> bool:
    if exp.events:
        return False
    body = (f"{PATH_HINT} exists but has no readable events ({exp.skipped_lines} unreadable lines)." if exp.exists
            else f"No recorded experiments. {what} will appear after the agent loop writes {PATH_HINT}.")
    with st.container(key=f"card_empty_{key}"):
        theme.empty_state("Nothing recorded yet", body)
    return True


def _pick_run(exp: ex.Experiments, key: str) -> ex.Run:
    ids = exp.run_ids[::-1]  # newest first, but open on the run with saved final-test results if any
    default = next((i for i, rid in enumerate(ids) if ex.comparison(rid)), 0)
    if len(ids) == 1:
        rid = ids[0]
    else:
        rid = st.selectbox("Run", ids, index=default, key=f"run_{key}", help=RUN_TIP,
                           format_func=lambda i: f"{i[:8]} · {'dry run' if ex.run(exp, i).dry_run else 'Luminal'}")
    r = ex.run(exp, rid)
    if r.dry_run:
        st.markdown(_chip("Dry run: static proposals, not cloud agent output. Any metrics shown were still measured "
                          "locally.", "warn"), unsafe_allow_html=True)
    if exp.skipped_lines:
        st.markdown(_chip(f"{exp.skipped_lines} unreadable lines skipped", "warn"), unsafe_allow_html=True)
    return r


def render_lab(exp: ex.Experiments) -> None:
    if _empty(exp, "Agent proposals and results", "lab"):
        return
    st.markdown('<p class="intro">Three AI agents, run through Luminal, each suggested one change per round. '
                "This computer trained every suggestion and kept it only if it measurably beat the best so far. "
                "The agents only ever saw summary numbers, never audio.</p>", unsafe_allow_html=True)
    r = _pick_run(exp, "lab")
    last, fin = r.latest, r.finished
    with st.container(key="card_run"):
        theme.card_title(f"Run {r.run_id[:8]}", f"Full run id: {r.run_id}")
        limits = last.get("limits") or {}
        base = ex.metric(r.baseline, "balanced_accuracy")
        st.markdown(
            '<div class="stats big-stats">'
            f'<div class="stat"><b>{_esc(last.get("llm_calls"))} / {_esc(limits.get("llm_calls", "—"))}</b><span>'
            f'{term("LLM calls", "How many times the agents asked the AI for a suggestion. The hard limit is 20.")}</span></div>'
            f'<div class="stat"><b>{_esc(last.get("model_experiments"))} / {_esc(limits.get("experiments", "—"))}</b><span>'
            f'{term("Model fits", "How many suggestions were actually trained and measured here. The hard limit is 12.")}</span></div>'
            + (f'<div class="stat"><b>{base * 100:.1f}%</b><span>{term("Starting point", "Balanced accuracy of the original model on the practice exam, before any agent suggestion.", "end")}</span></div>'
               if base is not None else "")
            + "</div>"
            + ("" if fin else '<p class="note">This run has no finish event yet; it may still be running or may have stopped early.</p>')
            + (f'<p class="note">Selected configuration</p>{_proposal_kv(fin.get("best_proposal"))}' if fin and fin.get("best_proposal") else ""),
            unsafe_allow_html=True,
        )
    _comparison_card(r.run_id, "lab")
    for rnd in r.rounds:
        evs = sorted((e for e in r.experiments if e.get("round") == rnd),
                     key=lambda e: ex.ROLES.index(e["role"]) if e.get("role") in ex.ROLES else 9)
        with st.container(key=f"card_round{rnd}"):
            theme.card_title(f"{len(evs)} suggestions", "In each round every agent proposes one change, informed by "
                             "the results of earlier rounds.", n=f"Round {rnd} ·")
            st.markdown(f'<div class="agrid">{"".join(card_html(e) for e in evs)}</div>', unsafe_allow_html=True)
    with st.container(key="card_history"):
        theme.card_title("Every experiment", "The same results as the cards above, as one sortable table.")
        rows = [{"round": e.get("round"), "role": e.get("role"), "status": e.get("status"), "decision": e.get("decision"),
                 **{METRIC_NAMES[m]: ex.metric(e, m) for m in ex.METRICS}, "seed": e.get("seed"),
                 "runtime s": e.get("runtime_s"), "reason": e.get("reason")} for e in r.experiments]
        if rows:
            cfg = {METRIC_NAMES[m]: st.column_config.NumberColumn(format="%.3f", help=METRIC_TIPS[m]) for m in ex.METRICS}
            cfg |= {"status": st.column_config.TextColumn(help="ok means trained and measured; failed or rejected means it was not."),
                    "decision": st.column_config.TextColumn(help="accepted: became the new best. rejected: did not beat it."),
                    "seed": st.column_config.NumberColumn(help="Fixed random seed, so the result can be reproduced exactly.")}
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config=cfg)
        else:
            st.markdown('<p class="note">No experiment events in this run.</p>', unsafe_allow_html=True)


def evolution_fig(pts: list[dict], name: str, baseline: float | None = None) -> go.Figure:
    label = METRIC_NAMES.get(name, name)
    fig = go.Figure(go.Scatter(x=[p["i"] + 1 for p in pts], y=[p["value"] for p in pts], mode="lines+markers",
                               line=dict(color=SIGNAL, width=2),
                               marker=dict(color=TOKENS["surface"], size=10, line=dict(color=SIGNAL, width=2)),
                               text=[p["label"] for p in pts],
                               hovertemplate="%{text}<br>" + label + " %{y:.3f}<extra></extra>"))
    best = [p for p in pts if p.get("best")]
    fig.add_trace(go.Scatter(x=[p["i"] + 1 for p in best], y=[p["value"] for p in best], mode="markers",
                             marker=dict(color=SIGNAL, size=18, symbol="star", line=dict(color=TOKENS["surface"], width=1.5)),
                             hoverinfo="skip"))
    if baseline is not None:
        fig.add_hline(y=baseline, line=dict(color=MUTE, dash="dot", width=1),
                      annotation_text="starting point", annotation_font_color=MUTE)
    fig.update_layout(**_BASE, height=320, showlegend=False,
                      xaxis=dict(title="experiment, in the order it ran", dtick=1, **_GRID),
                      yaxis=dict(title=label, range=[0, 1], **_GRID))
    return fig


def render_evolution(exp: ex.Experiments) -> None:
    if _empty(exp, "Validation metrics across experiments", "evo"):
        return
    st.markdown('<p class="intro">How each agent suggestion scored on the practice exam, in the order they were '
                "tried, and whether the favourite still won on the sealed final test.</p>", unsafe_allow_html=True)
    r = _pick_run(exp, "evo")
    names = ex.available_metrics(r.experiments)
    with st.container(key="card_evo_chart"):
        theme.card_title("Practice-exam score per experiment", VALIDATION_TIP)
        if not names:
            st.markdown('<p class="note">No experiment in this run has a measured validation metric, so there is nothing to plot.</p>',
                        unsafe_allow_html=True)
        else:
            name = st.radio("Metric", names, horizontal=True, label_visibility="collapsed",
                            format_func=lambda m: METRIC_NAMES[m])
            st.markdown(f'<p class="note">{html.escape(METRIC_TIPS[name])}</p>', unsafe_allow_html=True)
            pts = ex.trajectory(r.experiments, name)
            st.plotly_chart(evolution_fig(pts, name, ex.metric(r.baseline, name)), use_container_width=True,
                            config={"displayModeBar": False})
            best = next(p for p in pts if p.get("best"))
            st.markdown(f'<p class="note">Star: highest measured value, {best["value"]:.3f} ({_esc(best["label"])}). '
                        "Fixed validation partition only; this is not a final test result.</p>", unsafe_allow_html=True)
    _comparison_card(r.run_id, "evo")
    with st.container(key="card_evo_out"):
        theme.card_title("Outcomes, including failures", "Failures stay on the record. Hiding them would make the "
                         "agents look better than they were.")
        counts = ex.outcome_counts(r.experiments)
        st.markdown("".join(_chip(f"{k.replace('_', ' ')} · {v}", TONE.get(k, ""), STATUS_TIPS.get(k))
                            for k, v in counts.items())
                    + '<p class="note">Failed, rejected and unselected experiments stay in the record. '
                      "Only measured validation gains count as improvement.</p>", unsafe_allow_html=True)
