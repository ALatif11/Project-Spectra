"""Model, Project Evolution, Agent Lab and architecture views, rendered from ui.reporting evidence only."""

import html

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui import reporting as rp, theme
from ui.viz import _BASE, _GRID, HUMAN, MUTE, SIGNAL, SYNTH

FINAL_CAPTION = "In-the-Wild — local speaker-disjoint held-out test, 1,800 recordings."
KIND_LABEL = {"live": "Live Luminal session", "dry_run": "Dry run · static proposals, real local fits",
              "config_failure": "Cloud configuration failure · 0 calls"}
KIND_TONE = {"live": "ok", "dry_run": "warn", "config_failure": "bad"}
LIMITATIONS = [
    "Speaker separation does not establish independence of source recordings, sessions, shared text or voice generators.",
    "The data is public-figure English web audio with uneven speaker counts.",
    "How it generalizes to other microphones, languages, codecs and voice generators is unknown.",
    "There is no independent text-to-speech vendor holdout.",
    "The upstream test split was repurposed into local partitions; this is not an untouched official benchmark.",
    "Scores are uncalibrated model outputs and not forensic proof.",
]


def esc(v) -> str:
    return html.escape("—" if v is None else str(v))


def num(v) -> str:
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else "—"


def pct(v, d: int = 2) -> str:
    return "—" if v is None else f"{v * 100:.{d}f}%"


def sources_note(sec: dict) -> str:
    items = []
    for s in sec.get("sources", []):
        tag = f" · sha256 {s['sha256'][:12]}" if s.get("sha256") else f" · {s.get('error')}" if s.get("error") else ""
        items.append(f"<code>{esc(s['path'])}</code>{esc(tag)}")
    prov = (" · from sanitized export generated " + esc(sec.get("export_generated_at"))) if sec.get("provenance") == "export" else ""
    return f'<p class="note mono">Source: {"; ".join(items)}{prov}</p>' if items else ""


def unavailable(title: str, sec: dict) -> None:
    missing = [s["path"] for s in sec.get("sources", []) if s.get("error")]
    body = sec.get("reason") or "The saved report needed for this section is not on this computer."
    if missing:
        body += " Missing or unreadable: " + ", ".join(missing) + "."
    theme.empty_state(title, body)


def kv(rows) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows) + "</dl>"


def kpis(items) -> str:
    return '<div class="stats big-stats">' + "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{theme.term(k, t) if t else esc(k)}</span></div>' for k, v, t in items) + "</div>"


# ---------- Model ----------

def confusion_fig(cm) -> go.Figure:
    (tn, fp), (fn, tp) = cm
    z = [[tn, fp], [fn, tp]]
    text = [[f"{tn:,}<br>correct", f"{fp:,}<br>human flagged as synthetic"],
            [f"{fn:,}<br>synthetic missed", f"{tp:,}<br>correct"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted human", "Predicted synthetic"], y=["Actual human", "Actual synthetic"],
                               text=text, texttemplate="%{text}", textfont=dict(size=13, color="#0b0c0e"),
                               colorscale=[[0, "#2a3550"], [1, "#8fb3ff"]], showscale=False,
                               hovertemplate="%{y}, %{x}: %{z} recordings<extra></extra>"))
    fig.update_layout(**_BASE, height=260, yaxis=dict(autorange="reversed"))
    return fig


def render_model(ev: dict, active: dict) -> None:
    fin = ev["final"]
    role = rp.role_of(active, fin)
    attr = rp.attribution(active, fin)
    theme.page_heading("The model", "Which checkpoint is checking your clips, what it was trained on, and how it "
                       "scored on recordings it never saw.")
    with st.container(key="card_active"):
        theme.card_title("Model in use", "The checkpoint every check on this computer runs through. Identified by "
                         "hashing the file, never by its name or date.")
        if active.get("error"):
            st.markdown(f'<span class="chip bad">{esc(active["error"])}</span>', unsafe_allow_html=True)
        role_text = {"final": "Frozen final model (evaluated below)", "baseline": "Original Gary Stafford baseline",
                     "rollback": "Rollback copy of the original baseline", "other": "Unrecognized checkpoint",
                     "missing": "No checkpoint loaded"}[role]
        ds = active.get("dataset") or {}
        rows = [("Identity", role_text), ("Checkpoint", active.get("path")),
                ("Selected by", "ECHOPROOF_MODEL_PATH" if active.get("override") else "backend default (models/baseline.joblib)"),
                ("Model version", active.get("version")), ("Fingerprint (sha256)", (active.get("sha256") or "")[:16] or None),
                ("Training dataset", ds.get("name")), ("Dataset revision", (ds.get("revision") or "")[:12] or None),
                ("Classifier", " → ".join(x for x in (active.get("scaler"), active.get("classifier")) if x) or None),
                ("Settings", ", ".join(f"{k}={v}" for k, v in (active.get("params") or {}).items()) or None),
                ("Measurements per chunk", active.get("n_features")),
                ("Decision threshold", None if active.get("threshold") is None else f"{active['threshold']:.4f}")]
        st.markdown(kv(rows), unsafe_allow_html=True)
        if attr["active"]:
            st.markdown('<span class="chip ok">Matches the saved final evaluation identity: digest, threshold, '
                        "features and dataset revision</span>", unsafe_allow_html=True)
        elif active.get("sha256"):
            st.markdown('<span class="chip warn">The final evaluation below is historical: it is not about the loaded '
                        "checkpoint</span>", unsafe_allow_html=True)
            st.markdown('<p class="note">' + esc(" ".join(attr["reasons"])) + "</p>", unsafe_allow_html=True)

    with st.container(key="card_final"):
        theme.card_title("Held-out test result" + ("" if attr["active"] else " (historical, different checkpoint)"),
                         "Measured once on speakers the model never saw during training or tuning. The threshold "
                         "was fixed beforehand.")
        if fin["status"] != "ok":
            if fin["status"] == "invalid":
                st.markdown('<span class="chip bad">' + esc(" ".join(fin.get("warnings", []))) + "</span>", unsafe_allow_html=True)
            unavailable("Final evaluation unavailable", fin)
        else:
            m = fin["metrics"]
            st.markdown(f'<p class="note">{esc(FINAL_CAPTION)} Model {esc(fin.get("model_version"))} · '
                        f'sha256 {esc((fin.get("model_sha256") or "")[:12])}.</p>', unsafe_allow_html=True)
            st.markdown(kpis([
                ("Balanced accuracy", pct(m.get("balanced_accuracy")), "Average of how often genuine and synthetic recordings are classified correctly. 50% is a coin flip."),
                ("Precision (synthetic)", pct(m.get("precision_synthetic")), "Of the recordings flagged as synthetic, the share that really were."),
                ("Recall (synthetic)", pct(m.get("recall_synthetic")), "Of the synthetic recordings, the share that were caught."),
                ("F1 (synthetic)", f'{m["f1_synthetic"]:.3f}' if m.get("f1_synthetic") is not None else "—", "A single 0-1 score balancing precision and recall for synthetic speech."),
                ("False positive rate", pct(m.get("false_positive_rate")), "Share of genuine recordings wrongly flagged as synthetic."),
                ("False negative rate", pct(m.get("false_negative_rate")), "Share of synthetic recordings the model missed."),
            ]), unsafe_allow_html=True)
            a, b = st.columns([3, 2], gap="medium")
            with a:
                st.plotly_chart(confusion_fig(m["confusion"]), width="stretch", config={"displayModeBar": False})
            with b:
                (tn, fp), (fn, tp) = m["confusion"]
                st.markdown(f'<p class="note">Rows are the actual class, columns the prediction. {tn + fp:,} genuine and '
                            f"{fn + tp:,} synthetic recordings; {m['n']:,} in total.</p>"
                            f'<p class="note">Ordinary accuracy, computed from this matrix: {pct(m.get("derived_accuracy"))}. '
                            "It equals balanced accuracy here only because the test is exactly balanced.</p>"
                            '<p class="note">These are binary decisions at the frozen threshold. The app\'s "can\'t tell" band '
                            "(within 0.05 of the threshold) is not applied in these benchmark numbers.</p>"
                            f'<p class="note">Speakers in the test: {esc(fin.get("speakers"))}, disjoint from training: '
                            f'{esc(fin.get("speakers_disjoint"))}. Luminal calls: {esc(fin.get("luminal_calls"))}; '
                            f"model fits during testing: {esc(fin.get('model_fits'))}.</p>", unsafe_allow_html=True)
            st.markdown(sources_note(fin), unsafe_allow_html=True)

    exp = ev["expansion"]
    with st.container(key="card_data"):
        theme.card_title("Training data", "How the 12,000 selected In-the-Wild recordings were split. Whole speakers "
                         "go to one partition only.")
        c = exp.get("counts")
        if not c:
            unavailable("Split counts unavailable", exp)
        else:
            rows = "".join("<tr><td>" + esc(name) + "</td><td>" + esc(num(d.get("n"))) + "</td><td>" + esc(num(d.get("genuine")))
                           + "</td><td>" + esc(num(d.get("synthetic"))) + "</td><td>" + esc(d.get("speakers")) + "</td></tr>"
                           for name, d in (("Selected", c["selected"]), ("Train", c["train"]),
                                           ("Validation", c["validation"]), ("Test", c["test"])))
            st.markdown('<div class="tbl"><table class="cmp"><tr><th>Partition</th><th>Recordings</th><th>Genuine</th>'
                        f"<th>Synthetic</th><th>Speakers</th></tr>{rows}</table></div>"
                        '<p class="note">Only the 8,400 training recordings were used to fit the model; validation chose '
                        "the threshold; the test was scored once. No Gary Stafford recordings were used in this fit.</p>",
                        unsafe_allow_html=True)
            st.markdown(sources_note(exp), unsafe_allow_html=True)

    with st.container(key="card_how_model"):
        theme.card_title("What the model actually is")
        st.markdown('<p class="note">Each 4-second chunk is reduced to 52 numbers: averages and variation of 20 MFCC '
                    "coefficients, plus mean and variation of brightness (centroid), bandwidth, rolloff, flatness, "
                    "zero-crossing rate and loudness (RMS). A standard scaler and a logistic-regression classifier score "
                    "each chunk, and the clip score is the duration-weighted mean. The spectrogram you see on the check "
                    "page is for people; it is not the model's input. This is not a neural network, transformer or "
                    "end-to-end spectrogram model.</p>", unsafe_allow_html=True)

    with st.container(key="card_limits"):
        theme.card_title("Limitations")
        st.markdown('<ul class="warns">' + "".join(f"<li>{esc(t)}</li>" for t in LIMITATIONS) + "</ul>", unsafe_allow_html=True)


# ---------- Project Evolution ----------

def _bar_pair(fig, row, col, labels, values, colors, n_label):
    fig.add_trace(go.Bar(x=labels, y=[v * 100 if v is not None else None for v in values], marker_color=colors,
                         text=[pct(v) for v in values], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}: %{y:.2f}%<extra>" + n_label + "</extra>"), row=row, col=col)


def chart_original(o: dict) -> go.Figure:
    rows = o["rows"]
    fig = make_subplots(1, 2, subplot_titles=(f"Validation · {GS_SHORT} (250 clips)", f"One-time test · {GS_SHORT} (252 clips)"))
    for col, part in ((1, "validation"), (2, "test")):
        _bar_pair(fig, 1, col, ["Baseline", "Agent-optimized"], [rows["baseline"][part]["balanced_accuracy"],
                  rows["optimized"][part]["balanced_accuracy"]], ["#5b5e68", SIGNAL], part)
    fig.update_layout(**_BASE, height=300, showlegend=False)
    fig.update_yaxes(range=[90, 100], title="balanced accuracy (%)", **_GRID)
    fig.update_xaxes(**_GRID)
    return fig


GS_SHORT = "Gary Stafford"


def chart_expansion(c: dict) -> go.Figure:
    b, k = c["baseline"]["balanced_accuracy"], c["candidate"]["balanced_accuracy"]
    fig = go.Figure(go.Bar(x=["Original model", "Retrained on In-the-Wild"], y=[b * 100, k * 100],
                           marker_color=["#5b5e68", SIGNAL], text=[pct(b), pct(k)], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    fig.add_annotation(x=1, y=k * 100, yshift=34, showarrow=False, text=f"{(k - b) * 100:+.2f} percentage points",
                       font=dict(color=HUMAN))
    fig.update_layout(**_BASE, height=300, showlegend=False, title=dict(text="Same In-the-Wild validation set (1,800 recordings)", font=dict(size=13)),
                      yaxis=dict(range=[0, 110], title="balanced accuracy (%)", **_GRID), xaxis=_GRID)
    return fig


def chart_final(m: dict) -> go.Figure:
    names = ["Balanced accuracy", "Precision (synthetic)", "Recall (synthetic)", "F1 (synthetic)"]
    vals = [m.get("balanced_accuracy"), m.get("precision_synthetic"), m.get("recall_synthetic"), m.get("f1_synthetic")]
    fig = go.Figure(go.Bar(x=names, y=[v * 100 if v is not None else None for v in vals], marker_color=SIGNAL,
                           text=[pct(v) for v in vals], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    fig.update_layout(**_BASE, height=300, showlegend=False, title=dict(text=FINAL_CAPTION, font=dict(size=13)),
                      yaxis=dict(range=[0, 110], title="%", **_GRID), xaxis=_GRID)
    return fig


def chart_folds(s: dict) -> go.Figure | None:
    """Expanded session: development-fold balanced accuracy per configuration, with mean and spread."""
    series = []
    if s.get("baseline_folds"):
        series.append(("Incumbent baseline", s["baseline_folds"], s.get("baseline_objective") or {}))
    for p in s["proposals"]:
        if p["folds"]:
            series.append((f"R{p['round']} {rp.ROLE_NAMES.get(p['role'], p['role'])}", p["folds"], p.get("objective") or {}))
    if not series:
        return None
    fig = go.Figure()
    def fold_ba(folds, i):
        v = (folds[i] or {}).get("balanced_accuracy") if i < len(folds) else None
        return None if v is None else v * 100
    for i in range(max(len(f) for _, f, _ in series)):
        fig.add_trace(go.Bar(name=f"Fold {i + 1}", x=[n for n, _, _ in series], y=[fold_ba(f, i) for _, f, _ in series],
                             hovertemplate="%{x}, fold " + str(i + 1) + ": %{y:.2f}%<extra></extra>"))
    means = [(o.get("mean_fold_balanced_accuracy"), o.get("fold_balanced_accuracy_std")) for _, _, o in series]
    fig.add_trace(go.Scatter(name="Mean ± std", x=[n for n, _, _ in series], y=[m * 100 if m is not None else None for m, _ in means],
                             mode="markers", marker=dict(color=MUTE, size=11, symbol="diamond"),
                             error_y=dict(type="data", array=[(sd or 0) * 100 for _, sd in means], visible=True, color=MUTE),
                             hovertemplate="%{x}: mean %{y:.2f}%<extra></extra>"))
    fig.update_layout(**_BASE, height=340, barmode="group", legend=dict(orientation="h", y=-0.25),
                      title=dict(text="In-the-Wild training-speaker development folds (not the validation or test set)", font=dict(size=13)),
                      yaxis=dict(range=[50, 100], title="balanced accuracy (%)", **_GRID), xaxis=_GRID)
    return fig


def milestone(n: int, title: str, problem: str, change: str, measurement: str, outcome: str, lesson: str,
              evidence: list[str], key: str) -> None:
    with st.container(key=f"card_ms{n}"):
        theme.card_title(title, n=f"{n:02d} · ")
        st.markdown(kv([("Problem", problem), ("Change", change), ("Measurement", measurement),
                        ("Outcome", outcome), ("Lesson", lesson)]), unsafe_allow_html=True)
        if evidence:
            with st.expander("Evidence"):
                st.markdown("".join(f'<p class="note mono">{esc(e)}</p>' for e in evidence), unsafe_allow_html=True)


def render_evolution(ev: dict) -> None:
    theme.page_heading("How EchoProof learned", "Six steps from a first prototype to the frozen model, each with what "
                       "went wrong, what changed and what was measured. Different datasets are never joined in one curve.")
    o, x, f, ss = ev["original"], ev["expansion"], ev["final"], ev["sessions"]
    sess = {s.get("run_id"): s for s in ss.get("sessions", [])}
    live_gs = next((s for s in ss.get("sessions", []) if s["dataset"] == rp.GS and s["kind"] == "live"), None)
    itw_sessions = [s for s in ss.get("sessions", []) if s["dataset"] == rp.ITW and s["kind"] == "live"]

    if o["status"] == "ok":
        r = o["rows"]
        milestone(1, "Original local prototype", "Build a voice check that runs entirely on the user's computer.",
                  f"52 acoustic measurements + logistic regression, trained on {o.get('n_train') or '—'} {GS_SHORT} clips with speaker-grouped splits.",
                  f"Validation balanced accuracy {pct(r['baseline']['validation']['balanced_accuracy'])}; one-time test {pct(r['baseline']['test']['balanced_accuracy'])}.",
                  "A working local detector.", "Strong numbers inside one dataset say little about other recordings.",
                  [s["path"] for s in o["sources"]], "o")
        if live_gs:
            milestone(2, "First Luminal research run", "Can AI agents find a better configuration?",
                      f"Three researcher roles proposed changes in live run {live_gs['run_id'][:8]}: {live_gs['calls']} calls, "
                      f"{live_gs['fits']} local fits, {live_gs['rounds']} rounds, {live_gs['failures']} failed response.",
                      f"Best validation {pct(r['optimized']['validation']['balanced_accuracy'])} vs {pct(r['baseline']['validation']['balanced_accuracy'])}; "
                      f"test {pct(r['optimized']['test']['balanced_accuracy'])} vs {pct(r['baseline']['test']['balanced_accuracy'])}.",
                      "The optimized configuration did not improve the test score, so the baseline was kept.",
                      "A validation gain is not a real gain until it holds on unseen data.",
                      [live_gs["source"], *[s["path"] for s in o["sources"][2:]]], "a")
        st.plotly_chart(chart_original(o), width="stretch", config={"displayModeBar": False})
        st.markdown(f'<p class="note">Chart A · {GS_SHORT} only. Both models had the same ordinary test accuracy '
                    f"({pct(r['baseline']['test'].get('derived_accuracy'))}).</p>", unsafe_allow_html=True)
    else:
        unavailable("Original development reports unavailable", o)

    d = x.get("diagnostic") or {}
    milestone(3, "A real microphone recording failed",
              "One previously observed genuine recording, attested by the user, scored as almost certainly synthetic.",
              "An inference audit checked decoding, labels and scoring and found them consistent: the cause was domain shift.",
              f"Original model score {d['baseline_score']:.4f} (threshold 0.33)." if d.get("baseline_score") is not None else "Diagnostic score unavailable.",
              "Retrain on broader, real-world speech.", "In-dataset accuracy does not transfer to everyday recordings.",
              ["src/inference_audit.md", *[s["path"] for s in x.get("sources", []) if "verification" in s["path"]]], "d")

    cmp_ = x.get("comparison")
    if cmp_ and cmp_.get("baseline") and cmp_.get("candidate"):
        b, k = cmp_["baseline"]["balanced_accuracy"], cmp_["candidate"]["balanced_accuracy"]
        milestone(4, "In-the-Wild data and conventional retraining", "The original training data was too narrow.",
                  f"Selected {num(((x.get('counts') or {}).get('selected') or {}).get('n'))} In-the-Wild recordings in speaker-disjoint splits and "
                  f"retrained the same classifier on the {num(cmp_.get('training_clips'))} training recordings. Luminal calls: {cmp_.get('luminal_calls')}.",
                  f"Same In-the-Wild validation set: {pct(b)} → {pct(k)} ({(k - b) * 100:+.2f} percentage points). "
                  + (f"The microphone recording now scores {d['candidate_score']:.4f}." if d.get("candidate_score") is not None else ""),
                  "A much stronger candidate, from new data rather than agent tuning.", "Better data beat parameter search.",
                  [s["path"] for s in x["sources"]], "x")
        st.plotly_chart(chart_expansion(cmp_), width="stretch", config={"displayModeBar": False})
        st.markdown('<p class="note">Chart B · In-the-Wild validation only. The gain comes from new training data and '
                    "retraining, with zero Luminal calls.</p>", unsafe_allow_html=True)
    else:
        unavailable("Data expansion reports unavailable", x)

    if itw_sessions:
        lines = []
        for s in itw_sessions:
            fv = "no final validation gate ran" if s["format"] == "v2" and not s.get("final_validation") else ""
            lines.append(f"{s['run_id'][:8]}: {s.get('calls')} calls, {s.get('configurations') or sum(1 for p in s['proposals'] if p['metrics'])} measured configurations, "
                         f"status {s.get('status')}, stop {s.get('stop_reason')}{'; ' + fv if fv else ''}.")
        milestone(5, "Two In-the-Wild research sessions", "Could agents improve the retrained model further?",
                  "A first live session on the fixed validation set, then an expanded session using training-speaker folds.",
                  " ".join(lines), "No configuration beat the incumbent; the retrained model was kept.",
                  "Recording failures and rejections honestly is part of the result.", [s["source"] for s in itw_sessions], "s")
        exp2 = next((s for s in itw_sessions if s["format"] == "v2"), None)
        if exp2 and (fig := chart_folds(exp2)):
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown('<p class="note">Chart D · development folds only. Mean fold balanced accuracy differs from pooled '
                        "and speaker-macro scores, which the session also recorded.</p>", unsafe_allow_html=True)
    else:
        with st.container(key="card_ms5"):
            theme.card_title("Two In-the-Wild research sessions", n="05 · ")
            theme.empty_state("Session records not on this computer",
                              "Add runs/inthewild-luminal-20261011-v1/ and runs/inthewild-research-preparation-20261011-v2/ "
                              "(or a sanitized export) to show both sessions.")

    if f["status"] == "ok":
        m = f["metrics"]
        milestone(6, "Frozen model and final held-out test", "Measure the chosen model once on speakers it never saw.",
                  "Froze the retrained checkpoint and its threshold, then scored the sealed test once.",
                  f"Balanced accuracy {pct(m['balanced_accuracy'])} on {m['n']:,} recordings; false positive rate "
                  f"{pct(m['false_positive_rate'])}, false negative rate {pct(m['false_negative_rate'])}.",
                  "This frozen model is what the demo launcher selects.", "One honest number beats many tuned ones.",
                  [s["path"] for s in f["sources"]], "f")
        st.plotly_chart(chart_final(m), width="stretch", config={"displayModeBar": False})
        st.markdown('<p class="note">Chart C · ' + esc(FINAL_CAPTION) + "</p>", unsafe_allow_html=True)
    else:
        unavailable("Final evaluation unavailable", f)


# ---------- Agent Lab ----------

def _limits_text(s: dict) -> str:
    lim = s.get("limits") or {}
    keys = [("rounds", "rounds"), ("llm_calls", "calls"), ("api_calls", "calls"), ("experiments", "fits"),
            ("ml_fits", "fits"), ("max_output_tokens", "output tokens/call")]
    parts = [f"{lim[k]} {label}" for k, label in keys if isinstance(lim.get(k), (int, float))]
    return ", ".join(parts) or "not recorded"


def proposal_html(p: dict) -> str:
    cfg = p.get("config") or {}
    params = ", ".join(f"{k}={v}" for k, v in (cfg.get("parameters") or {}).items())
    m, o = p.get("metrics") or {}, p.get("objective") or {}
    tone = "ok" if p.get("status") == "ok" else "bad"
    measured = []
    if o.get("mean_fold_balanced_accuracy") is not None:
        measured.append(f"mean fold BA {pct(o['mean_fold_balanced_accuracy'])}")
    if o.get("score") is not None:
        measured.append(f"objective {o['score']:.4f}")
    if m.get("balanced_accuracy") is not None and not o:
        measured.append(f"BA {pct(m['balanced_accuracy'])}")
    for k, label in (("false_positive_rate", "FPR"), ("false_negative_rate", "FNR")):
        if m.get(k) is not None:
            measured.append(f"{label} {pct(m[k])}")
    return (f'<div class="acard"><h4>R{esc(p.get("round"))} · {esc(rp.ROLE_NAMES.get(p.get("role"), p.get("role")))}</h4>'
            f'<span class="chip {tone}">{esc(p.get("status"))}</span><span class="chip plain">{esc(p.get("decision"))}</span>'
            + (f'<p class="note">Hypothesis: {esc(p["hypothesis"])}</p>' if p.get("hypothesis") else "")
            + (kv([("feature set", cfg.get("feature_set")), ("classifier", cfg.get("classifier")), ("parameters", params or None)]) if cfg else
               '<p class="note">No configuration (the response failed before a proposal was made).</p>')
            + (f'<p class="note">{esc(" · ".join(measured))}</p>' if measured else "")
            + f'<p class="note mono">{esc(p.get("reason"))}</p></div>')


def render_agents(ev: dict) -> None:
    theme.page_heading("Agent Lab", "Every recorded research session: what the AI researchers proposed, what was "
                       "measured locally, and what was kept. Failures stay in the record.")
    ss = ev["sessions"]
    sessions = ss.get("sessions", [])
    if not sessions:
        unavailable("No research sessions recorded", ss)
        return
    if ss.get("missing"):
        st.markdown('<span class="chip warn">Not on this computer: ' + esc(", ".join(ss["missing"])) + "</span>",
                    unsafe_allow_html=True)
    datasets = list(dict.fromkeys(s["dataset"] for s in sessions))
    a, b = st.columns(2)
    ds = a.selectbox("Dataset", datasets, index=len(datasets) - 1, key="lab_ds")
    pool = [s for s in sessions if s["dataset"] == ds]
    default = next((i for i, s in enumerate(pool) if s["kind"] == "live"), 0)
    s = pool[b.selectbox("Run", range(len(pool)), index=default, key="lab_run",
                         format_func=lambda i: f"{pool[i]['run_id'][:8]} · {KIND_LABEL[pool[i]['kind']]}")]
    with st.container(key="card_session"):
        theme.card_title(f"Run {s['run_id'][:8]}", f"Full run id: {s['run_id']}. Protocol: {s['protocol']}.")
        st.markdown(f'<span class="chip {KIND_TONE[s["kind"]]}">{esc(KIND_LABEL[s["kind"]])}</span>'
                    + ('<span class="chip warn">from sanitized export</span>' if s.get("provenance") == "export" else ""),
                    unsafe_allow_html=True)
        fits_label = "Fold fits completed" if s["format"] == "v2" else "Model fits"
        items = [("LLM calls", s.get("calls"), "Requests to the Luminal-hosted model for a proposal, including failed ones."),
                 (fits_label, s.get("fits"), "Local training runs on cached features. Each expanded-session configuration needs three fold fits."),
                 ("Rounds", s.get("rounds"), None), ("Status", s.get("status"), None)]
        if s["format"] == "v2":
            items.insert(2, ("Configurations measured", s.get("configurations"), "Distinct proposals that were fully measured."))
        st.markdown(kpis(items), unsafe_allow_html=True)
        rows = [("Dataset", s["dataset"]), ("Protocol", s["protocol"]), ("Stop reason", s.get("stop_reason")),
                ("Limits for this run", _limits_text(s)), ("Source", s.get("source"))]
        if s.get("estimated_cost_usd") is not None:
            rows.append(("Estimated cost", f"${s['estimated_cost_usd']:.6f} (rate-card estimate, not an invoice)"))
        if s["format"] == "v2":
            rows.append(("Final validation gate", "ran" if s.get("final_validation") else
                         "did not run: no configuration beat the incumbent, so no new validation result exists"))
        st.markdown(kv(rows), unsafe_allow_html=True)
    rounds = sorted({p["round"] for p in s["proposals"] if p["round"] is not None})
    for rnd in rounds:
        with st.container(key=f"card_round_{rnd}"):
            theme.card_title(f"Round {rnd}")
            st.markdown('<div class="agrid">' + "".join(proposal_html(p) for p in s["proposals"] if p["round"] == rnd) + "</div>",
                        unsafe_allow_html=True)
    if s["format"] == "v2" and (fig := chart_folds(s)):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------- Architecture and sponsors ----------

def flow(steps: list[str]) -> str:
    return '<div class="flow">' + '<span class="arrow">→</span>'.join(f"<span class='node'>{esc(s)}</span>" for s in steps) + "</div>"


def render_architecture(ev: dict) -> None:
    st.markdown('<div class="prose"><h3>Architecture</h3></div>', unsafe_allow_html=True)
    with st.container(key="card_arch_local"):
        theme.card_title("Local detection (every check)")
        st.markdown(flow(["Your audio", "Local decoding · 16 kHz mono", "4-second chunks", "52 acoustic measurements",
                          "Fitted local classifier", "Chunk and recording scores"]), unsafe_allow_html=True)
        st.markdown('<p class="note">No network request is made to check a clip.</p>', unsafe_allow_html=True)
    with st.container(key="card_arch_research"):
        theme.card_title("Research loop (offline, during development)")
        st.markdown(flow(["Feature Researcher · Classifier Researcher · Robustness Researcher (Luminal)",
                          "Constrained proposal validation", "Deterministic experiments on cached features",
                          "Metrics and guardrails", "Round feedback or early stop"]), unsafe_allow_html=True)
        st.markdown('<p class="note">The researchers only see summary metrics, never audio. Python, not the model, '
                    "runs every experiment.</p>", unsafe_allow_html=True)
    live = [s for s in ev["sessions"].get("sessions", []) if s["kind"] == "live"]
    calls = sum(s.get("calls") or 0 for s in live)
    with st.container(key="card_sponsors"):
        theme.card_title("How sponsor technology is used")
        st.markdown(
            '<div class="prose"><h3>Luminal (Transpose)</h3><p>'
            f"{len(live)} recorded live research session(s), {calls} recorded calls to a DeepSeek model served by Luminal "
            "across three researcher roles with competing proposals, each measured locally, with round feedback, "
            "recorded failures and early stopping. See Agent Lab.</p>"
            "<h3>On-device intelligence</h3><p>Decoding, feature extraction and classification run locally, and "
            "ordinary detection works offline with no cloud request. This describes the implementation; it is not an "
            "SDK integration or an endorsement.</p></div>", unsafe_allow_html=True)
