"""EchoProof AI frontend. Run: streamlit run app.py"""

import hashlib

import streamlit as st

from src.audio import AudioDecodeError, decode_audio
from ui import theme, viz
from ui.detector import model_html, result_html, run_detector
from ui.experiments import PATH, load as load_experiments
from ui.lab import render_evolution, render_lab

st.set_page_config(page_title="EchoProof AI", page_icon="🎧", layout="wide")
theme.inject()
theme.hero()

WAVE_TIP = ("The waveform shows loudness over time. Tall peaks are syllables or loud sounds; flat stretches are "
            "pauses. It helps you check the recording is what you expect.")
SPEC_TIP = ("A spectrogram is a picture of sound. Left to right is time, bottom to top is pitch (low voice tones at "
            "the bottom, hissy s and f sounds near the top). Brighter means louder. Bright or odd-looking spots are "
            "not proof of anything.")
SEG_TIP = ("The model listens in 4-second chunks and scores each one. Red bars are at or above the threshold "
           "(synthetic-like), green bars are below it (human-like). The overall result is a length-weighted average.")
STEPS = [
    ("Decode on this computer", "Your file is converted to 16 kHz mono audio. Nothing is uploaded anywhere."),
    ("Listen in 4-second chunks", "The model measures 52 properties of each chunk, like timbre, brightness and loudness."),
    ("Score each chunk", "Each chunk gets a model score from 0 (human-like) to 1 (synthetic-like)."),
]


@st.cache_data(show_spinner=False, max_entries=4)
def analyze(digest: str, data: bytes, name: str):
    """Decode + figures, keyed by content hash. Returns (error, recording, wave, spec)."""
    try:
        rec = decode_audio(data, name)
    except AudioDecodeError as exc:
        return str(exc), None, None, None
    return None, rec, viz.waveform_fig(rec), viz.spectrogram_fig(rec)


def steps_html() -> str:
    return '<div class="steps">' + "".join(
        f'<div class="step"><span class="n">0{i}</span><b>{t}</b><p>{d}</p></div>' for i, (t, d) in enumerate(STEPS, 1)
    ) + "</div>"


tab_detective, tab_lab, tab_evo = st.tabs(["Audio Detective", "Agent Lab", "Model Evolution"])

with tab_detective:
    left, right = st.columns([7, 5], gap="medium")
    with left:
        with st.container(key="card_upload"):
            theme.card_title("Your recording", "WAV, FLAC or OGG up to 25 MB and 120 seconds. "
                             "The file is analyzed in memory on this computer and never sent to a server.")
            up = st.file_uploader("Upload audio", type=["wav", "flac", "ogg"], label_visibility="collapsed")
            if not up:
                st.markdown(steps_html(), unsafe_allow_html=True)

    data = up.getvalue() if up else None
    err = rec = wave = spec = res = None
    if data:
        err, rec, wave, spec = analyze(hashlib.sha256(data).hexdigest(), data, up.name)
        if err is None:
            with st.spinner("Listening…"):
                res = run_detector(data, up.name)  # runs every script pass, so a fresh upload is never stale
    with right:
        with st.container(key="card_result"):
            theme.card_title("Result", "What the trained model makes of your recording. Hover the small i marks for "
                             "what each number means.")
            st.markdown(model_html(), unsafe_allow_html=True)
            if not data:
                st.markdown('<div class="verdict off">Waiting for audio</div><p class="note">Drop a clip on the left. '
                            "A few seconds of someone talking works best.</p>", unsafe_allow_html=True)
            elif err:
                st.markdown(result_html({"status": "invalid", "label": "inconclusive", "synthetic_score": None,
                                         "threshold": None, "segments": [], "model_version": None, "message": err}),
                            unsafe_allow_html=True)
            else:
                st.markdown(result_html(res), unsafe_allow_html=True)

    if data and not err:
        if rec.is_silent:
            st.markdown('<span class="chip warn">This recording is almost silent, so the analysis is unreliable</span>',
                        unsafe_allow_html=True)
        if res["status"] == "ok" and res["segments"]:
            with st.container(key="card_segments"):
                theme.card_title("Chunk by chunk", SEG_TIP)
                st.plotly_chart(viz.segments_fig(res["segments"], res["threshold"]), use_container_width=True,
                                config={"displayModeBar": False})
        a, b = st.columns([5, 7], gap="medium")
        with a:
            with st.container(key="card_player"):
                theme.card_title(f"Listen back · {rec.duration_s:.1f} s", WAVE_TIP)
                st.audio(data)
                st.plotly_chart(wave, use_container_width=True, config={"displayModeBar": False})
        with b:
            with st.container(key="card_spec"):
                theme.card_title("What the sound looks like", SPEC_TIP)
                st.plotly_chart(spec, use_container_width=True, config={"displayModeBar": False})
                st.markdown('<p class="note">Technically a log-Mel spectrogram, closely related to what the model '
                            "measures. Hover anywhere on it to read time, pitch and loudness.</p>",
                            unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load(mtime: float):
    return load_experiments()


experiments = _load(PATH.stat().st_mtime if PATH.is_file() else 0.0)

with tab_lab:
    render_lab(experiments)

with tab_evo:
    render_evolution(experiments)
