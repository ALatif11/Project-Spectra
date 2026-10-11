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


def _wav(seconds=1.0, amp=0.3):
    buf = io.BytesIO()
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(buf, (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, format="WAV")
    return buf.getvalue()


def test_real_backend_without_model_is_unavailable(monkeypatch, tmp_path):
    from ui.detector import run_detector
    monkeypatch.setenv("ECHOPROOF_MODEL_PATH", str(tmp_path / "missing.joblib"))
    res = run_detector(_wav(), "tone.wav")
    assert res["status"] == "unavailable" and res["synthetic_score"] is None and res["label"] == "inconclusive"


def test_invalid_audio_passes_through():
    from ui.detector import run_detector
    assert run_detector(b"not audio", "x.wav")["status"] == "invalid"


def test_malformed_or_raising_backend_is_unavailable(monkeypatch):
    import src.inference as inf
    from ui.detector import run_detector
    monkeypatch.setattr(inf, "predict_audio", lambda b, f=None: {"status": "ok"})
    assert run_detector(b"x", None)["status"] == "unavailable"

    def boom(b, f=None):
        raise RuntimeError("x")
    monkeypatch.setattr(inf, "predict_audio", boom)
    assert run_detector(b"x", None)["status"] == "unavailable"


def _stub(label, score):
    return lambda b, f=None: {
        "status": "ok", "label": label, "synthetic_score": score, "threshold": 0.5,
        "segments": [{"start_s": 0.0, "end_s": 1.0, "synthetic_score": score}],
        "model_version": "test-stub", "message": "stub"}


def test_result_html_states():
    """Stub lives only in tests: checks rendering paths, not model behavior."""
    from ui.detector import result_html
    for label, text in [("likely_human", "Likely human"), ("likely_synthetic", "Likely synthetic"), ("inconclusive", "Inconclusive")]:
        out = result_html(_stub(label, 0.52)(b"", None))
        assert text in out and "0.520" in out and "calibrated probability" in out and "Not forensic proof" in out
    assert "close to the decision threshold" in result_html(_stub("inconclusive", 0.52)(b"", None))
    out = result_html({"status": "unavailable", "label": "inconclusive", "synthetic_score": None, "threshold": None,
                       "segments": [], "model_version": None, "message": "<b>m</b>"})
    assert "Detector unavailable" in out and "—" in out and "&lt;b&gt;" in out and "meter" not in out
