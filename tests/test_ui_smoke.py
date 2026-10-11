import io
from pathlib import Path

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest


def test_app_renders_without_upload():
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Audio Detective", "Agent Lab", "Model Evolution"]


def test_detector_unavailable_without_backend_inference():
    from ui.detector import run_detector
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    res = run_detector(buf.getvalue(), "x.wav")
    assert res["status"] == "unavailable" and res["synthetic_score"] is None and res["segments"] == []


def test_figures_build_for_tone():
    from src.audio import decode_audio
    from ui import viz
    buf = io.BytesIO()
    t = np.arange(16000 * 5) / 16000
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, format="WAV")
    rec = decode_audio(buf.getvalue(), "tone.wav")
    assert viz.waveform_fig(rec).data and viz.spectrogram_fig(rec).data
