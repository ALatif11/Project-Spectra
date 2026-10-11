"""EchoProof frontend. Run: streamlit run app.py"""

import hashlib
import html

import streamlit as st

from src.audio import AudioDecodeError, decode_audio
from ui import explain, theme, viz
from ui.detector import checkpoint_label, headline, model_html, result_html, run_detector
from ui.experiments import PATH, load as load_experiments
from ui.lab import render_evolution, render_lab

st.set_page_config(page_title="EchoProof · Check a voice clip", page_icon="🎧", layout="wide")
theme.inject()
theme.header()

SEG_TIP = ("The model listens in 4-second chunks and scores each one. Red bars are at or above the threshold "
           "(AI-like), green bars are below it (human-like). The overall result is a length-weighted average.")
WAVE_TIP = "Loudness over time. Tall peaks are syllables; flat stretches are pauses."
SPEC_TIP = ("A picture of the sound. Left to right is time, bottom to top is pitch, brighter is louder. "
            "Odd-looking spots are not proof of anything.")
NEXT_TITLES = {"ai": "Before you do anything", "human": "A human-sounding voice can still be a scam",
               "unsure": "When in doubt, verify another way", "off": "If a call or voice note feels wrong"}


@st.cache_data(show_spinner=False, max_entries=4)
def analyze(digest: str, data: bytes, name: str):
    """Decode + figures, keyed by content hash. Returns (error, recording, wave, spec)."""
    try:
        rec = decode_audio(data, name)
    except AudioDecodeError as exc:
        return str(exc), None, None, None
    return None, rec, viz.waveform_fig(rec), viz.spectrogram_fig(rec)


@st.cache_data(show_spinner=False, max_entries=8)
def why_for(digest: str, checkpoint: str, data: bytes, name: str):
    return explain.factors(data, name)


def next_steps_html(tone: str) -> str:
    items = "".join(f"<li><b>{html.escape(t)}</b><span>{html.escape(d)}</span></li>" for t, d in explain.NEXT_STEPS)
    links = "".join(f'<a href="{u}" target="_blank" rel="noopener">{html.escape(n)} ↗</a>' for n, u in explain.REPORT_LINKS)
    return (f'<div class="card-h"><span class="n">{html.escape(NEXT_TITLES[tone])}</span></div>'
            f'<ol class="steps-list">{items}</ol><div class="links">Lost money or shared details? Call your bank now, '
            f"then report it. {links}</div>")


def model_card() -> None:
    try:
        from src.model import load_model
        from ui.detector import _model_path
        art = load_model(_model_path())
        clf = type(art["pipeline"].named_steps.get("classifier")).__name__
        rows = [("Version", art.get("model_version", "—")), ("Checkpoint", checkpoint_label()),
                ("Decision threshold", f'{art["threshold"]:.3f}'), ("Classifier", clf),
                ("Measurements per chunk", len((art.get("feature_settings") or {}).get("feature_names") or []) or "—")]
    except Exception as exc:  # missing or incompatible checkpoint
        rows = [("Checkpoint", checkpoint_label()), ("Status", f"could not be loaded ({type(exc).__name__})")]
    kv = "".join(f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows)
    with st.container(key="card_model"):
        theme.card_title("Model in use", "The checkpoint every check on this computer runs through. A new model "
                         "is only used once someone deliberately points the app at it.")
        st.markdown(f'<dl class="kv">{kv}</dl><p class="note">The saved baseline stays the default until a new '
                    "checkpoint is intentionally selected.</p>", unsafe_allow_html=True)


tab_check, tab_how, tab_model = st.tabs(["Check a clip", "How it works", "Model"])

with tab_check:
    theme.page_heading("Check a voice clip", "Got a voice note or call recording that doesn't feel right? See "
                       "whether it sounds like AI-generated speech before you act on it.")
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(key="card_upload"):
            theme.card_title("Your clip", "WAV, FLAC or OGG, up to 25 MB and 120 seconds. Analyzed in memory on "
                             "this computer and never uploaded.")
            up = st.file_uploader("Upload audio", type=["wav", "flac", "ogg"], label_visibility="collapsed")

    data = up.getvalue() if up else None
    err = rec = wave = spec = res = None
    tone = "off"
    if data:
        digest = hashlib.sha256(data).hexdigest()
        err, rec, wave, spec = analyze(digest, data, up.name)
        if err is None:
            with st.spinner("Checking…"):
                res = run_detector(data, up.name)  # runs every script pass, so a fresh upload is never stale
        else:
            res = {"status": "invalid", "label": "inconclusive", "synthetic_score": None, "threshold": None,
                   "segments": [], "model_version": None, "message": err}
        tone = headline(res)[0]

    with left:
        if data and not err:
            with st.container(key="card_clip"):
                theme.card_title(f"{up.name} · {rec.duration_s:.1f} s")
                st.audio(data)
                if rec.is_silent:
                    st.markdown('<span class="chip warn">This recording is almost silent</span>', unsafe_allow_html=True)

    with right:
        with st.container(key="card_result"):
            theme.card_title("Result")
            if not data:
                st.markdown('<div class="answer off"><span class="dot"></span><div><h2>Add a clip to check it</h2>'
                            "<p>You'll get a plain answer, the reasons behind it, and what to do next.</p></div></div>"
                            '<ul class="tips"><li><span><b>Five seconds or more</b> of one person speaking works best.</span></li>'
                            "<li><span><b>Avoid music</b> or several people talking at once.</span></li>"
                            "<li><span><b>Use the original file</b> if you can; re-recording from a speaker changes the sound.</span></li></ul>",
                            unsafe_allow_html=True)
            else:
                ok = res["status"] == "ok"
                why = why_for(digest, checkpoint_label(), data, up.name) if ok else None
                warnings = explain.quality_checks(rec.samples, rec.duration_s) if rec is not None else []
                st.markdown(result_html(res, why, warnings), unsafe_allow_html=True)
                if ok:
                    report = explain.report_markdown(up.name, headline(res)[1], res, why, warnings, checkpoint_label())
                    st.download_button("Save report", report, file_name=f"echoproof-{digest[:8]}.md",
                                       mime="text/markdown", key="dl_report")
            st.markdown(f'<div style="margin-top:.9rem">{model_html()}</div>', unsafe_allow_html=True)

    with st.container(key="card_next"):
        st.markdown(next_steps_html(tone), unsafe_allow_html=True)

    if data and not err:
        with st.expander("Details: chunk scores, waveform and spectrogram"):
            if res["status"] == "ok" and res["segments"]:
                theme.card_title("Score per 4-second chunk", SEG_TIP)
                st.plotly_chart(viz.segments_fig(res["segments"], res["threshold"]), use_container_width=True,
                                config={"displayModeBar": False})
            a, b = st.columns([5, 7], gap="medium")
            with a:
                theme.card_title("Waveform", WAVE_TIP)
                st.plotly_chart(wave, use_container_width=True, config={"displayModeBar": False})
            with b:
                theme.card_title("Spectrogram", SPEC_TIP)
                st.plotly_chart(spec, use_container_width=True, config={"displayModeBar": False})

with tab_how:
    theme.page_heading("How EchoProof works", "What happens to your clip, what the result means, and where it can be wrong.")
    st.markdown("""<div class="prose">
<h3>1. Your clip stays on this computer</h3>
<p>The file is decoded in memory and converted to 16 kHz mono audio. It is never uploaded, stored on a server or sent to an AI service.</p>
<h3>2. It listens in 4-second chunks</h3>
<p>For each chunk the model takes acoustic measurements, such as loudness, brightness and the tone color of the voice, and compares them with what it learned from examples of real and AI-generated speech.</p>
<h3>3. You get a score, a threshold and the reasons</h3>
<p>The model score runs from 0 (resembles real speech it has studied) to 1 (resembles AI-generated speech). Above the team's threshold the clip is labeled AI-like; very close to it, we say we can't tell. The "Why" list shows which measurements moved the score most.</p>
<h3>Where it can be wrong</h3>
<ul>
<li>Real voices recorded on unfamiliar microphones, in echoey rooms, in other languages or with long pauses can score as AI-like.</li>
<li>Newer voice generators the model has never heard can score as human-like.</li>
<li>The score is not a probability and not forensic proof. Treat it as one signal and always verify through another channel.</li>
</ul>
<h3>If something feels wrong</h3>
<p>Call the person back on a number you already know, ask something only they would know, and never send gift cards, wire transfers, crypto or cash because of a phone call.</p>
</div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load(mtime: float):
    return load_experiments()


with tab_model:
    theme.page_heading("The model", "Which model is running, how it was trained and tested, and the experiments "
                       "behind it. Every number here comes from saved reports.")
    model_card()
    experiments = _load(PATH.stat().st_mtime if PATH.is_file() else 0.0)
    sub_runs, sub_history = st.tabs(["Agent experiments", "Results over time"])
    with sub_runs:
        render_lab(experiments)
    with sub_history:
        render_evolution(experiments)
