"""EchoProof AI frontend. Run: streamlit run app.py"""

import hashlib

import streamlit as st

from src.audio import AudioDecodeError, decode_audio
from ui import theme, viz
from ui.experiments import PATH, load as load_experiments
from ui.lab import render_evolution, render_lab
from ui.detector import model_html, result_html, run_detector

st.set_page_config(page_title="EchoProof AI", page_icon="🎧", layout="wide")
theme.inject()
theme.hero()


@st.cache_data(show_spinner=False, max_entries=4)
def analyze(digest: str, data: bytes, name: str):
    """Decode + figures, keyed by content hash. Returns (error, recording, wave, spec)."""
    try:
        rec = decode_audio(data, name)
    except AudioDecodeError as exc:
        return str(exc), None, None, None
    return None, rec, viz.waveform_fig(rec), viz.spectrogram_fig(rec)


def result_card(res: dict) -> None:
    st.markdown(result_html(res), unsafe_allow_html=True)


tab_detective, tab_lab, tab_evo = st.tabs(["Audio Detective", "Agent Lab", "Model Evolution"])

with tab_detective:
    left, right = st.columns([7, 5], gap="medium")
    with left:
        with st.container(key="bezel_upload"):
            theme.card_title("Input · processed locally")
            up = st.file_uploader("Upload audio", type=["wav", "flac", "ogg"], label_visibility="collapsed")
            st.caption("WAV or FLAC works best. Max 25 MB / 120 s. Nothing leaves this machine.")

    data = up.getvalue() if up else None
    err = rec = wave = spec = res = None
    if data:
        err, rec, wave, spec = analyze(hashlib.sha256(data).hexdigest(), data, up.name)
        if err is None:
            res = run_detector(data, up.name)  # runs every script pass, so a fresh upload is never stale
    with right:
        with st.container(key="bezel_result"):
            theme.card_title("Detector")
            st.markdown(model_html(), unsafe_allow_html=True)
            if not data:
                st.markdown('<div class="verdict off">Awaiting audio</div><div class="note">'
                            "Upload a recording to decode it and run the local detector.</div>", unsafe_allow_html=True)
            elif err:
                result_card({"status": "invalid", "label": "inconclusive", "synthetic_score": None, "threshold": None,
                             "segments": [], "model_version": None, "message": err})
            else:
                result_card(res)

    if data and not err:
        if rec.is_silent:
            st.markdown('<span class="pill warn">Near-silent recording — analysis is unreliable</span>', unsafe_allow_html=True)
        a, b = st.columns([5, 7], gap="medium")
        with a:
            with st.container(key="bezel_player"):
                theme.card_title(f"Playback · {rec.duration_s:.1f}s · 16 kHz mono")
                st.audio(data)
                st.plotly_chart(wave, use_container_width=True, config={"displayModeBar": False})
        with b:
            with st.container(key="bezel_spec"):
                theme.card_title("Log-Mel spectrogram")
                st.plotly_chart(spec, use_container_width=True, config={"displayModeBar": False})
                st.caption("Visual aid only. Bright regions are not evidence of manipulation.")
        if res["status"] == "ok" and res["segments"]:
            with st.container(key="bezel_segments"):
                theme.card_title("Model score per 4 s segment")
                st.plotly_chart(viz.segments_fig(res["segments"], res["threshold"]), use_container_width=True,
                                config={"displayModeBar": False})

@st.cache_data(show_spinner=False)
def _load(mtime: float):
    return load_experiments()


experiments = _load(PATH.stat().st_mtime if PATH.is_file() else 0.0)

with tab_lab:
    render_lab(experiments)

with tab_evo:
    render_evolution(experiments)
